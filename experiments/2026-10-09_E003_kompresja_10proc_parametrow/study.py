"""E003: kalibracja i rzeczywista kompresja/ocena całego CNN na czterech GPU.
ID 2026-10-09_E003_kompresja_10proc_parametrow; protokół PROTOKOL_v1.md i OPIS.md.
Wejścia: frozen config/plan, własny checkpoint, A/B train oraz dev E001.
Wyjścia: momenty A, grupy, mniejsze checkpointy, predykcje, błędy i czasy.
Bez treningu, rekalibracji BN, testu i rezerwy. Każdy obraz oceny liczony raz.
"""
import argparse
import gc
import os
import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from support import (HERE, ROOT, SOURCE, digest, read_json, write_json, require, model_digest,
                     verify_all_inputs, load_data, loader, base_model, seed_all, rng_state)
from folding import fold_model, materialize_model, count_parameters
from selection import choose_groups, selection_settings


class Capture:
    def __init__(self, model, rows, kind):
        self.values, self.counts, self.handles = {}, {}, []
        def record(path, key, value):
            self.values.setdefault(path, {})[key] = value.detach().clone()
            self.counts[(path, key)] = self.counts.get((path, key), 0) + 1
        for row in rows:
            path = row['block']; block = model.get_submodule(path)
            if kind == 'calibration':
                se = block.block[2]
                self.handles.extend([
                    se.register_forward_pre_hook(lambda m, x, p=path: record(p, 'h', x[0])),
                    se.avgpool.register_forward_hook(lambda m, x, y, p=path: record(p, 'g', y)),
                    se.register_forward_hook(lambda m, x, y, p=path: record(p, 'z', y))])
            else:
                self.handles.extend([
                    block.register_forward_pre_hook(lambda m, x, p=path: record(p, 'input', x[0])),
                    block.block.register_forward_hook(lambda m, x, y, p=path: record(p, 'branch', y)),
                    block.register_forward_hook(lambda m, x, y, p=path: record(p, 'block', y))])
        self.kind = kind; self.rows = rows

    def reset(self):
        self.values.clear(); self.counts.clear()

    def check(self, batch):
        keys = ('h', 'g', 'z') if self.kind == 'calibration' else ('input', 'branch', 'block')
        require(all(self.counts.get((row['block'], key)) == 1 for row in self.rows for key in keys),
                'Hook invocation mismatch')
        for row in self.rows:
            values = self.values[row['block']]
            require(all(v.dtype == torch.float32 and v.shape[0] == batch and bool(torch.isfinite(v).all())
                        for v in values.values()), 'Invalid capture')
            if self.kind == 'calibration':
                require(values['h'].shape == values['z'].shape
                        and values['h'].shape[1] == row['expanded_channels']
                        and tuple(values['g'].shape) == (batch, row['expanded_channels'], 1, 1), 'SE shape mismatch')

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.reset()


def moment_file(run, rank):
    return run / f'moments_rank{rank}.npz'


def calibrate(model, data_loader, records, rows, run, rank, device):
    capture = Capture(model, rows, 'calibration')
    moments = {row['block']: {key: np.zeros((row['expanded_channels'],)*2, dtype=np.float64)
                              for key in ('h', 'z', 'g')} for row in rows}
    ids_seen, targets_seen = [], []
    try:
        with torch.inference_mode():
            for x, y, ids in data_loader:
                capture.reset(); output = model(x.to(device))
                require(bool(torch.isfinite(output).all()), 'Non-finite calibration output')
                capture.check(len(ids))
                for row in rows:
                    values = capture.values[row['block']]
                    for key in ('h', 'z'):
                        array = values[key].cpu().numpy().astype(np.float64)
                        matrix = array.transpose(1, 0, 2, 3).reshape(row['expanded_channels'], -1)
                        moments[row['block']][key] += matrix @ matrix.T
                    g = values['g'][:, :, 0, 0].cpu().numpy().astype(np.float64)
                    moments[row['block']]['g'] += g.T @ g
                ids_seen.extend(ids.tolist()); targets_seen.extend(y.tolist())
        require(ids_seen == [r['id'] for r in records[rank::4]]
                and targets_seen == [r['target'] for r in records[rank::4]], 'Calibration IDs mismatch')
        arrays = {f'b{i}_{key}': moments[row['block']][key] for i, row in enumerate(rows) for key in ('h','z','g')}
        np.savez(moment_file(run, rank), **arrays)
        write_json(run / f'calibration_rank{rank}.json', {'rank':rank, 'ids':ids_seen, 'targets':targets_seen,
                   'count':len(ids_seen), 'moments_sha256':digest(moment_file(run,rank))})
    finally:
        capture.close()
    del moments
    gc.collect()


def aggregate_moments(rows, run):
    total = {row['block']: {k:np.zeros((row['expanded_channels'],)*2, dtype=np.float64)
                            for k in ('h','z','g')} for row in rows}
    for rank in range(4):
        with np.load(moment_file(run,rank), allow_pickle=False) as data:
            for i,row in enumerate(rows):
                for key in ('h','z','g'):
                    total[row['block']][key] += data[f'b{i}_{key}']
    np.savez(run / 'moments_merged.npz', **{f'b{i}_{key}':total[row['block']][key]
              for i,row in enumerate(rows) for key in ('h','z','g')})
    return total


def save_models(cpu_base, rows, candidates, cfg, run):
    (run / 'models').mkdir()
    base_hash = model_digest(cpu_base)
    files = {}
    for mode in cfg['modes']:
        model = cpu_base if mode == 'BASE' else fold_model(cpu_base, rows, candidates[mode])
        model.eval()
        expected = cfg['base_parameters'] if mode == 'BASE' else cfg['expected_compressed_parameters']
        require(count_parameters(model) == expected, 'Physical parameter budget mismatch')
        require(all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()), 'Non-finite model state')
        path = run / 'models' / f'{mode}.pt'
        torch.save({'variant':mode, 'compression_applied':mode!='BASE', 'structure_rows':rows,
                    'groups_sha256':digest(run / 'candidate_manifest.json'), 'parameters':expected,
                    'model_sha256':model_digest(model), 'state_dict':model.state_dict()}, path)
        files[mode] = {'bytes':path.stat().st_size, 'sha256':digest(path), 'model_sha256':model_digest(model)}
        if mode != 'BASE':
            del model
    require(model_digest(cpu_base) == base_hash, 'Base altered by compression')
    write_json(run / 'model_files.json', files)


def load_models(cpu_base, rows, cfg, run, device):
    models, hashes = {}, {}
    for mode in cfg['modes']:
        state = torch.load(run / 'models' / f'{mode}.pt', map_location='cpu', weights_only=True)
        require(state['compression_applied'] == (mode!='BASE') and state['variant']==mode
                and state['groups_sha256']==digest(run / 'candidate_manifest.json'), 'Artifact metadata mismatch')
        if mode == 'BASE':
            import copy
            model = copy.deepcopy(cpu_base)
        else:
            model = materialize_model(cpu_base, rows)
        model.load_state_dict(state['state_dict'], strict=True)
        model.to(device).eval()
        require(count_parameters(model) == state['parameters'] and model_digest(model) == state['model_sha256'],
                'Strict reloaded model differs')
        models[mode] = model; hashes[mode] = state['model_sha256']
    return models, hashes


def diagnostics(models, data_loader, records, rows, rank, device):
    capture = Capture(models['BASE'], rows, 'diagnostic')
    modes = ('PRE','POST','RANDOM')
    sums = np.zeros((3,len(rows),4), dtype=np.float64)
    ids_seen=[]
    try:
        with torch.inference_mode():
            for x,y,ids in data_loader:
                capture.reset(); models['BASE'](x.to(device)); capture.check(len(ids))
                for i,row in enumerate(rows):
                    target = capture.values[row['block']]
                    for j,mode in enumerate(modes):
                        block = models[mode].get_submodule(row['block'])
                        # Evaluate on common teacher input and clone branch before inplace residual.
                        branch = block.block(target['input']).clone()
                        output = branch + target['input'] if block.use_res_connect else branch
                        require(bool(torch.isfinite(branch).all()) and bool(torch.isfinite(output).all()),
                                'Non-finite block output')
                        for offset,(candidate,truth) in enumerate(((branch,target['branch']),(output,target['block']))):
                            a,b = candidate.double(),truth.double()
                            sums[j,i,2*offset] += float(((a-b)**2).sum())
                            sums[j,i,2*offset+1] += float((b*b).sum())
                ids_seen.extend(ids.tolist())
        require(ids_seen==[r['id'] for r in records[rank::4]], 'Diagnostic coverage mismatch')
    finally:
        capture.close()
    return sums,ids_seen


def evaluate(models, data_loader, records, cfg, rank, device):
    stats = {mode:[0.0,0,0] for mode in cfg['modes']}
    logits = {mode:[] for mode in cfg['modes']}; ids_seen=[]; targets_seen=[]
    with torch.inference_mode():
        for x,y,ids in data_loader:
            x,y=x.to(device),y.to(device)
            for mode in cfg['modes']:
                output=models[mode](x)
                require(output.dtype==torch.float32 and bool(torch.isfinite(output).all()), 'Non-finite dev logits')
                stats[mode][0] += float(nn.functional.cross_entropy(output,y,reduction='sum').double())
                stats[mode][1] += int((output.argmax(1)==y).sum()); stats[mode][2] += len(ids)
                logits[mode].append(output.cpu().numpy().copy())
            ids_seen.extend(ids.tolist()); targets_seen.extend(y.tolist())
    require(ids_seen==[r['id'] for r in records[rank::4]] and targets_seen==[r['target'] for r in records[rank::4]],
            'Dev rank coverage mismatch')
    return stats, {'ids':np.asarray(ids_seen),'targets':np.asarray(targets_seen),
                   **{f'logits_{m}':np.concatenate(v) for m,v in logits.items()}}


def benchmark(models,cfg,rank,device):
    x=torch.randn(cfg['latency_batch'],3,cfg['image_size'],cfg['image_size'],device=device)
    entries=[]
    with torch.inference_mode():
        for mode in cfg['modes']:
            for _ in range(cfg['latency_warmup']): models[mode](x)
            torch.cuda.synchronize(device)
            times=[]
            for _ in range(cfg['latency_measurements']):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record(); models[mode](x); end.record(); end.synchronize()
                times.append(float(start.elapsed_time(end)))
            entries.append({'rank':rank,'gpu':torch.cuda.get_device_name(device),'mode':mode,'milliseconds':times})
    return entries


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args(); run=args.run_dir.resolve()
    rank,local_rank=int(os.environ['RANK']),int(os.environ['LOCAL_RANK'])
    require(int(os.environ['WORLD_SIZE'])==4 and torch.cuda.device_count()==4,'Exactly 4 GPUs/processes required')
    torch.cuda.set_device(local_rank); device=torch.device('cuda',local_rank)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(minutes=30),device_id=device)
    try:
        cfg=read_json(HERE/'config.json'); rows=read_json(HERE/cfg['structure_plan'])['blocks']
        require(cfg['status']=='frozen','Unfrozen config')
        if rank==0: write_json(run/'integrity.json',verify_all_inputs(cfg))
        dist.barrier()
        images,labels,samples,records=load_data(cfg)
        if rank==0:
            write_json(run/'SAMPLES.json',samples)
            write_json(run/'dev_records.json',records['dev'])
        seed_all(cfg['seed']+rank)
        cpu_base=base_model(cfg); base_hash=model_digest(cpu_base)
        import copy
        teacher=copy.deepcopy(cpu_base).to(device).eval()
        torch.save(rng_state(device),run/f'rng_before_rank{rank}.pt')
        print(f'rank={rank} GPU={torch.cuda.get_device_name(local_rank)} A=125 B=125 dev=1250 phase=calibration',flush=True)
        calibrate(teacher,loader(images,labels,records['A'],cfg,rank),records['A'],rows,run,rank,device)
        require(model_digest(teacher)==base_hash,'Teacher changed in calibration')
        del teacher; torch.cuda.empty_cache(); dist.barrier()
        if rank==0:
            all_cal=[read_json(run/f'calibration_rank{r}.json') for r in range(4)]
            require(sorted(sum([v['ids'] for v in all_cal],[]))==[r['id'] for r in records['A']], 'Global A coverage')
            moments=aggregate_moments(rows,run); groups=choose_groups(moments,rows,cfg)
            mappings={mode:{row['block']:[i for i in range(row['expanded_channels'])
                         if i not in {g['d'] for g in groups[mode][row['block']]}] for row in rows}
                      for mode in ('PRE','POST','RANDOM')}
            write_json(run/'candidate_manifest.json',{'groups':groups,'kept_indices':mappings,
                       'settings':selection_settings(cfg),'calibration_half':'A','calibration_count':500,
                       'moments_sha256':digest(run/'moments_merged.npz'),'before_B_dev_evaluation':True})
            del moments; gc.collect()
            save_models(cpu_base,rows,groups,cfg,run)
        dist.barrier()
        models,before=load_models(cpu_base,rows,cfg,run,device)
        print(f'rank={rank} phase=diagnostics_and_dev physical_parameters={count_parameters(models["POST"])}',flush=True)
        errors,diag_ids=diagnostics(models,loader(images,labels,records['B'],cfg,rank),records['B'],rows,rank,device)
        np.save(run/f'block_errors_rank{rank}.npy',errors)
        stats,preds=evaluate(models,loader(images,labels,records['dev'],cfg,rank),records['dev'],cfg,rank,device)
        np.savez_compressed(run/f'dev_rank{rank}.npz',**preds)
        times=benchmark(models,cfg,rank,device)
        after={mode:model_digest(model) for mode,model in models.items()}
        require(before==after and model_digest(cpu_base)==base_hash,'Model state changed during evaluation')
        torch.save(rng_state(device),run/f'rng_after_rank{rank}.pt')
        result={'rank':rank,'local_rank':local_rank,'gpu':torch.cuda.get_device_name(local_rank),
                'A_count':125,'B_count':len(diag_ids),'dev_count':len(preds['ids']),
                'diagnostic_ids':diag_ids,'stats':stats,'model_before':before,'model_after':after,
                'latency':times,'peak_memory_bytes':torch.cuda.max_memory_allocated(device)}
        write_json(run/f'rank{rank}.json',result)
        all_results=[None]*4; dist.all_gather_object(all_results,result)
        if rank==0:
            require(sorted(sum([v['diagnostic_ids'] for v in all_results],[]))==[r['id'] for r in records['B']], 'Global B coverage')
            require(all(v['model_before']==all_results[0]['model_before'] for v in all_results),'Replica mismatch')
            merged=[]
            for r in range(4):
                with np.load(run/f'dev_rank{r}.npz',allow_pickle=False) as f: merged.append({k:f[k].copy() for k in f.files})
            joined={k:np.concatenate([v[k] for v in merged]) for k in merged[0]}
            order=np.argsort(joined['ids']); joined={k:v[order] for k,v in joined.items()}
            require(joined['ids'].tolist()==[r['id'] for r in records['dev']]
                    and len(np.unique(joined['ids']))==5000
                    and joined['targets'].tolist()==[r['target'] for r in records['dev']], 'Global dev coverage')
            np.savez_compressed(run/'dev_predictions.npz',**joined)
            reference=read_json(SOURCE/'best_dev_predictions.json')
            baseline=np.stack([joined['ids'],joined['targets'],joined['logits_BASE'].argmax(1)],axis=1).tolist()
            require(baseline==reference,'Baseline prediction mismatch')
            metrics={}
            for mode in cfg['modes']:
                loss=sum(v['stats'][mode][0] for v in all_results)
                correct=sum(v['stats'][mode][1] for v in all_results); count=sum(v['stats'][mode][2] for v in all_results)
                require(count==5000 and correct==int((joined[f'logits_{mode}'].argmax(1)==joined['targets']).sum()), 'Metric mismatch')
                metrics[mode]={'correct':correct,'count':count,'top1':correct/count,'nll':loss/count,
                               'parameters':count_parameters(models[mode]),'model_sha256':before[mode]}
            oldmetrics=[__import__('json').loads(line) for line in (SOURCE/'metrics.jsonl').read_text().splitlines()]
            reference_nll=oldmetrics[176]['dev_nll']
            require(metrics['BASE']['correct']==cfg['baseline_correct']
                    and abs(metrics['BASE']['nll']-reference_nll)<=cfg['baseline_nll_atol'],'Baseline NLL/accuracy mismatch')
            require(digest(ROOT/cfg['checkpoint'])==cfg['checkpoint_sha256'],'Source checkpoint changed during study')
            allerrors=sum((np.load(run/f'block_errors_rank{r}.npy',allow_pickle=False) for r in range(4)),
                          np.zeros((3,len(rows),4),dtype=np.float64))
            errout={mode:{row['block']:{key:float(allerrors[j,i,n]/allerrors[j,i,n+1])
                         if allerrors[j,i,n+1]>cfg['norm_zero_threshold']**2 else None
                         for key,n in (('branch_relative_mse',0),('block_relative_mse',2))}
                         for i,row in enumerate(rows)} for j,mode in enumerate(('PRE','POST','RANDOM'))}
            write_json(run/'block_errors.json',errout)
            write_json(run/'latency.json',sum([v['latency'] for v in all_results],[]))
            write_json(run/'raw_metrics.json',{'modes':metrics,'count':5000,'unique_ids':5000,
                       'baseline_reproduced':True,'calibration_count':500,'diagnostic_count':500,
                       'model_files':read_json(run/'model_files.json'),'source_checkpoint_unchanged':
                       digest(ROOT/cfg['checkpoint'])==cfg['checkpoint_sha256'], 'reference_base_nll':reference_nll})
            print(f'E003_METRICS={metrics}',flush=True)
        dist.barrier()
    except Exception:
        (run/f'error_rank{rank}.txt').write_text(traceback.format_exc()); raise
    finally:
        dist.destroy_process_group()


if __name__=='__main__': main()
