"""E008 etap 1: kalibracja i pomiar szkody kompresji 10%/30% na 4 GPU (bez naprawy).
Eksperyment: 2026-10-12_E008_kompresja_224; protokol: OPIS.md i PROTOKOL_v1.md.
Wejscia: frozen config, plany PLAN_STRUKTURY_10/30.json, baseline E007, A/B train, dev.
Wyjscia: momenty A, grupy per budzet, skompresowane modele, predykcje dev, bledy blokow.
Bez treningu, rekalibracji BN, testu i rezerwy; kazdy obraz oceny liczony raz.
"""
import argparse
import gc
import json
import os
import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from support_e008 import (HERE, ROOT, digest, read_json, write_json, require, model_digest,
                          verify_inputs, load_data, loader, base_model, seed_all, rng_state)
from folding_e008 import fold_model, materialize_model, count_parameters
from selection import choose_groups, selection_settings


class Capture:
    def __init__(self, model, rows, kind):
        self.values, self.counts, self.handles = {}, {}, []

        def record(path, key, value):
            self.values.setdefault(path, {})[key] = value.detach().clone()
            self.counts[(path, key)] = self.counts.get((path, key), 0) + 1

        for row in rows:
            path = row['block']
            block = model.get_submodule(path)
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
        self.kind = kind
        self.rows = rows

    def reset(self):
        self.values.clear()
        self.counts.clear()

    def check(self, batch):
        keys = ('h', 'g', 'z') if self.kind == 'calibration' else ('input', 'branch', 'block')
        require(all(self.counts.get((row['block'], key)) == 1
                    for row in self.rows for key in keys), 'Hook invocation mismatch')
        for row in self.rows:
            values = self.values[row['block']]
            require(all(v.dtype == torch.float32 and v.shape[0] == batch
                        and bool(torch.isfinite(v).all()) for v in values.values()),
                    'Invalid capture')
            if self.kind == 'calibration':
                require(values['h'].shape == values['z'].shape
                        and values['h'].shape[1] == row['expanded_channels']
                        and tuple(values['g'].shape) == (batch, row['expanded_channels'], 1, 1),
                        'SE shape mismatch')

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.reset()


def moment_file(run, rank):
    return run / f'moments_rank{rank}.npz'


def calibrate(model, data_loader, records, rows, run, rank, device):
    capture = Capture(model, rows, 'calibration')
    moments = {row['block']: {key: np.zeros((row['expanded_channels'],) * 2, dtype=np.float64)
                              for key in ('h', 'z', 'g')} for row in rows}
    ids_seen, targets_seen = [], []
    try:
        with torch.inference_mode():
            for x, y, ids in data_loader:
                capture.reset()
                output = model(x.to(device))
                require(bool(torch.isfinite(output).all()), 'Non-finite calibration output')
                capture.check(len(ids))
                for row in rows:
                    values = capture.values[row['block']]
                    for key in ('h', 'z'):
                        matrix = values[key].transpose(1, 0).reshape(
                            row['expanded_channels'], -1).double()
                        moments[row['block']][key] += (matrix @ matrix.t()).cpu().numpy()
                    g = values['g'][:, :, 0, 0].double()
                    moments[row['block']]['g'] += (g.t() @ g).cpu().numpy()
                ids_seen.extend(ids.tolist())
                targets_seen.extend(y.tolist())
        require(ids_seen == [r['id'] for r in records[rank::4]]
                and targets_seen == [r['target'] for r in records[rank::4]],
                'Calibration IDs mismatch')
        arrays = {f'b{i}_{key}': moments[row['block']][key]
                  for i, row in enumerate(rows) for key in ('h', 'z', 'g')}
        np.savez(moment_file(run, rank), **arrays)
        write_json(run / f'calibration_rank{rank}.json',
                   {'rank': rank, 'ids': ids_seen, 'targets': targets_seen,
                    'count': len(ids_seen), 'moments_sha256': digest(moment_file(run, rank))})
    finally:
        capture.close()
    del moments
    gc.collect()


def aggregate_moments(rows, run):
    total = {row['block']: {k: np.zeros((row['expanded_channels'],) * 2, dtype=np.float64)
                            for k in ('h', 'z', 'g')} for row in rows}
    for rank in range(4):
        with np.load(moment_file(run, rank), allow_pickle=False) as data:
            for i, row in enumerate(rows):
                for key in ('h', 'z', 'g'):
                    total[row['block']][key] += data[f'b{i}_{key}']
    np.savez(run / 'moments_merged.npz',
             **{f'b{i}_{key}': total[row['block']][key]
                for i, row in enumerate(rows) for key in ('h', 'z', 'g')})
    return total


def variant_names(cfg):
    return ['BASE'] + [f'{budget}_{mode}' for budget in cfg['budgets'] for mode in cfg['modes']]


def save_models(cpu_base, cfg, rows_by_budget, groups_by_budget, run):
    (run / 'models').mkdir()
    base_hash = model_digest(cpu_base)
    files = {}
    for name in variant_names(cfg):
        if name == 'BASE':
            model = cpu_base
            expected = cfg['base_parameters']
            budget = None
        else:
            budget, mode = name.split('_', 1)
            model = fold_model(cpu_base, rows_by_budget[budget], groups_by_budget[budget][mode])
            expected = cfg['plans'][budget]['compressed_parameters']
        model.eval()
        require(count_parameters(model) == expected, f'Physical parameter budget mismatch: {name}')
        require(all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()),
                'Non-finite model state')
        path = run / 'models' / f'{name}.pt'
        payload = {'variant': name, 'compression_applied': name != 'BASE', 'budget': budget,
                   'parameters': expected, 'model_sha256': model_digest(model),
                   'state_dict': model.state_dict()}
        if budget is not None:
            payload['plan_file'] = cfg['plans'][budget]['file']
            payload['plan_sha256'] = cfg['plans'][budget]['sha256']
            payload['groups_sha256'] = digest(run / f'candidate_manifest_{budget}.json')
            payload['structure_rows'] = rows_by_budget[budget]
        torch.save(payload, path)
        files[name] = {'bytes': path.stat().st_size, 'sha256': digest(path),
                       'model_sha256': model_digest(model)}
        if name != 'BASE':
            del model
    require(model_digest(cpu_base) == base_hash, 'Base altered by compression')
    write_json(run / 'model_files.json', files)
    return files


def load_models(cpu_base, cfg, rows_by_budget, run, device):
    import copy
    models, hashes = {}, {}
    for name in variant_names(cfg):
        state = torch.load(run / 'models' / f'{name}.pt', map_location='cpu', weights_only=True)
        require(state['variant'] == name and state['compression_applied'] == (name != 'BASE'),
                'Artifact metadata mismatch')
        if name == 'BASE':
            model = copy.deepcopy(cpu_base)
        else:
            budget, _ = name.split('_', 1)
            require(state['groups_sha256'] == digest(run / f'candidate_manifest_{budget}.json'),
                    'Groups hash mismatch')
            model = materialize_model(cpu_base, rows_by_budget[budget])
        model.load_state_dict(state['state_dict'], strict=True)
        model.to(device).eval()
        require(count_parameters(model) == state['parameters']
                and model_digest(model) == state['model_sha256'],
                'Strict reloaded model differs')
        models[name] = model
        hashes[name] = state['model_sha256']
    return models, hashes


def diagnostics(models, cfg, rows, data_loader, records, rank, device):
    capture = Capture(models['BASE'], rows, 'diagnostic')
    sums = np.zeros((len(cfg['budgets']) * len(cfg['modes']), len(rows), 4), dtype=np.float64)
    ids_seen = []
    try:
        with torch.inference_mode():
            for x, y, ids in data_loader:
                capture.reset()
                models['BASE'](x.to(device))
                capture.check(len(ids))
                for i, row in enumerate(rows):
                    target = capture.values[row['block']]
                    for bi, budget in enumerate(cfg['budgets']):
                        for mi, mode in enumerate(cfg['modes']):
                            block = models[f'{budget}_{mode}'].get_submodule(row['block'])
                            branch = block.block(target['input']).clone()
                            output = branch + target['input'] if block.use_res_connect else branch
                            require(bool(torch.isfinite(branch).all())
                                    and bool(torch.isfinite(output).all()),
                                    'Non-finite block output')
                            index = bi * len(cfg['modes']) + mi
                            for offset, (candidate, truth) in enumerate(
                                    ((branch, target['branch']), (output, target['block']))):
                                a, b = candidate.double(), truth.double()
                                sums[index, i, 2 * offset] += float(((a - b) ** 2).sum())
                                sums[index, i, 2 * offset + 1] += float((b * b).sum())
                ids_seen.extend(ids.tolist())
        require(ids_seen == [r['id'] for r in records[rank::4]], 'Diagnostic coverage mismatch')
    finally:
        capture.close()
    return sums, ids_seen


def evaluate(models, names, data_loader, records, rank, device):
    stats = {name: [0.0, 0, 0] for name in names}
    logits = {name: [] for name in names}
    ids_seen, targets_seen = [], []
    with torch.inference_mode():
        for x, y, ids in data_loader:
            x, y = x.to(device), y.to(device)
            for name in names:
                output = models[name](x)
                require(output.dtype == torch.float32 and bool(torch.isfinite(output).all()),
                        'Non-finite dev logits')
                stats[name][0] += float(nn.functional.cross_entropy(
                    output, y, reduction='sum').double())
                stats[name][1] += int((output.argmax(1) == y).sum())
                stats[name][2] += len(ids)
                logits[name].append(output.cpu().numpy().copy())
            ids_seen.extend(ids.tolist())
            targets_seen.extend(y.tolist())
    require(ids_seen == [r['id'] for r in records[rank::4]]
            and targets_seen == [r['target'] for r in records[rank::4]],
            'Dev rank coverage mismatch')
    return stats, {'ids': np.asarray(ids_seen), 'targets': np.asarray(targets_seen),
                   **{f'logits_{m}': np.concatenate(v) for m, v in logits.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    require(int(os.environ['WORLD_SIZE']) == 4 and torch.cuda.device_count() == 4,
            'Exactly 4 GPUs/processes required')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(minutes=30), device_id=device)
    try:
        cfg = read_json(HERE / 'config.json')
        require(cfg['status'] == 'frozen', 'Unfrozen config')
        if rank == 0:
            write_json(run / 'integrity.json', verify_inputs(cfg))
        dist.barrier()
        images, labels, records = load_data(cfg)
        if rank == 0:
            write_json(run / 'SAMPLES.json', {
                'a_per_class': cfg['data']['a_per_class'], 'sample_seed': cfg['data']['sample_seed'],
                'A': [r['id'] for r in records['A']], 'B': [r['id'] for r in records['B']],
                'dev_count': len(records['dev'])})
        rows_by_budget = {b: read_json(HERE / cfg['plans'][b]['file'])['blocks']
                          for b in cfg['budgets']}
        paths = [row['block'] for row in rows_by_budget[cfg['budgets'][0]]]
        require(all([row['block'] for row in rows_by_budget[b]] == paths for b in cfg['budgets']),
                'Plans disagree on block list')
        seed_all(cfg['seed'] + rank)
        cpu_base = base_model(cfg)
        base_hash = model_digest(cpu_base)
        import copy
        teacher = copy.deepcopy(cpu_base).to(device).eval()
        torch.save(rng_state(device), run / f'rng_before_rank{rank}.pt')
        a_count = len(records['A'][rank::4])
        b_count = len(records['B'][rank::4])
        dev_count = len(records['dev'][rank::4])
        print(f'rank={rank} GPU={torch.cuda.get_device_name(local_rank)} A={a_count} '
              f'B={b_count} dev={dev_count} phase=calibration', flush=True)
        calibrate(teacher, loader(images, labels, records['A'], cfg, rank,
                                  cfg['calibration_batch_per_gpu']),
                  records['A'], rows_by_budget[cfg['budgets'][0]], run, rank, device)
        require(model_digest(teacher) == base_hash, 'Teacher changed in calibration')
        del teacher
        torch.cuda.empty_cache()
        dist.barrier()
        if rank == 0:
            all_cal = [read_json(run / f'calibration_rank{r}.json') for r in range(4)]
            require(sorted(sum([v['ids'] for v in all_cal], [])) == [r['id'] for r in records['A']],
                    'Global A coverage')
            moments = aggregate_moments(rows_by_budget[cfg['budgets'][0]], run)
            groups_by_budget = {}
            for budget in cfg['budgets']:
                rows = rows_by_budget[budget]
                groups = choose_groups(moments, rows, cfg)
                groups_by_budget[budget] = groups
                mappings = {mode: {row['block']: [i for i in range(row['expanded_channels'])
                             if i not in {g['d'] for g in groups[mode][row['block']]}] for row in rows}
                            for mode in cfg['modes']}
                write_json(run / f'candidate_manifest_{budget}.json', {
                    'budget': budget, 'plan_file': cfg['plans'][budget]['file'],
                    'plan_sha256': cfg['plans'][budget]['sha256'],
                    'groups': groups, 'kept_indices': mappings,
                    'settings': selection_settings(cfg), 'calibration_half': 'A',
                    'calibration_count': len(records['A']),
                    'moments_sha256': digest(run / 'moments_merged.npz'),
                    'before_B_dev_evaluation': True})
            del moments
            gc.collect()
            save_models(cpu_base, cfg, rows_by_budget, groups_by_budget, run)
        dist.barrier()
        models, before = load_models(cpu_base, cfg, rows_by_budget, run, device)
        names = variant_names(cfg)
        print(f'rank={rank} phase=diagnostics_and_dev base_parameters={count_parameters(models["BASE"])}',
              flush=True)
        errors, diag_ids = diagnostics(models, cfg, rows_by_budget[cfg['budgets'][0]],
                                       loader(images, labels, records['B'], cfg, rank,
                                              cfg['b_batch_per_gpu']),
                                       records['B'], rank, device)
        np.save(run / f'block_errors_rank{rank}.npy', errors)
        stats, preds = evaluate(models, names,
                                loader(images, labels, records['dev'], cfg, rank,
                                       cfg['eval_batch_per_gpu']),
                                records['dev'], rank, device)
        np.savez_compressed(run / f'dev_rank{rank}.npz', **preds)
        after = {name: model_digest(model) for name, model in models.items()}
        require(before == after and model_digest(cpu_base) == base_hash,
                'Model state changed during evaluation')
        torch.save(rng_state(device), run / f'rng_after_rank{rank}.pt')
        result = {'rank': rank, 'local_rank': local_rank,
                  'gpu': torch.cuda.get_device_name(local_rank),
                  'A_count': a_count, 'B_count': len(diag_ids), 'dev_count': len(preds['ids']),
                  'diagnostic_ids': diag_ids, 'stats': stats,
                  'model_before': before, 'model_after': after,
                  'peak_memory_bytes': torch.cuda.max_memory_allocated(device)}
        write_json(run / f'rank{rank}.json', result)
        all_results = [None] * 4
        dist.all_gather_object(all_results, result)
        if rank == 0:
            require(sorted(sum([v['diagnostic_ids'] for v in all_results], []))
                    == [r['id'] for r in records['B']], 'Global B coverage')
            require(all(v['model_before'] == all_results[0]['model_before'] for v in all_results),
                    'Replica mismatch')
            merged = []
            for r in range(4):
                with np.load(run / f'dev_rank{r}.npz', allow_pickle=False) as f:
                    merged.append({k: f[k].copy() for k in f.files})
            joined = {k: np.concatenate([v[k] for v in merged]) for k in merged[0]}
            order = np.argsort(joined['ids'])
            joined = {k: v[order] for k, v in joined.items()}
            require(joined['ids'].tolist() == [r['id'] for r in records['dev']]
                    and len(np.unique(joined['ids'])) == cfg['data']['dev_count']
                    and joined['targets'].tolist() == [r['target'] for r in records['dev']],
                    'Global dev coverage')
            np.savez_compressed(run / 'dev_predictions.npz', **joined)
            metrics = {}
            for name in names:
                loss = sum(v['stats'][name][0] for v in all_results)
                correct = sum(v['stats'][name][1] for v in all_results)
                count = sum(v['stats'][name][2] for v in all_results)
                require(count == cfg['data']['dev_count']
                        and correct == int((joined[f'logits_{name}'].argmax(1)
                                            == joined['targets']).sum()),
                        f'Metric mismatch for {name}')
                metrics[name] = {'correct': correct, 'count': count, 'top1': correct / count,
                                 'nll': loss / count, 'parameters': count_parameters(models[name]),
                                 'model_sha256': before[name]}
            require(metrics['BASE']['correct'] == cfg['baseline_gate']['correct']
                    and abs(metrics['BASE']['nll'] - cfg['baseline_gate']['nll'])
                    <= cfg['baseline_gate']['nll_atol'], 'Baseline gate mismatch')
            all_errors = sum((np.load(run / f'block_errors_rank{r}.npy', allow_pickle=False)
                              for r in range(4)),
                             np.zeros((len(cfg['budgets']) * len(cfg['modes']),
                                       len(rows_by_budget[cfg['budgets'][0]]), 4),
                                      dtype=np.float64))
            index = {f'{b}_{m}': bi * len(cfg['modes']) + mi
                     for bi, b in enumerate(cfg['budgets']) for mi, m in enumerate(cfg['modes'])}
            errout = {name: {row['block']: {
                'branch_relative_mse': float(all_errors[index[name], i, 0]
                                             / all_errors[index[name], i, 1])
                if all_errors[index[name], i, 1] > cfg['norm_zero_threshold'] ** 2 else None,
                'block_relative_mse': float(all_errors[index[name], i, 2]
                                            / all_errors[index[name], i, 3])
                if all_errors[index[name], i, 3] > cfg['norm_zero_threshold'] ** 2 else None}
                for i, row in enumerate(rows_by_budget[cfg['budgets'][0]])} for name in index}
            write_json(run / 'block_errors.json', errout)
            write_json(run / 'stage1_metrics.json', {
                'modes': metrics, 'dev_count': cfg['data']['dev_count'],
                'baseline_reproduced': True, 'calibration_count': len(records['A']),
                'diagnostic_count': len(records['B']),
                'model_files': read_json(run / 'model_files.json'),
                'source_checkpoint_unchanged': digest(ROOT / cfg['baseline']['run'] / 'last.pt')
                == cfg['baseline']['sha256']})
            print('E008_STAGE1_METRICS=' + json.dumps(
                {k: {'top1': v['top1'], 'nll': v['nll']} for k, v in metrics.items()}), flush=True)
        dist.barrier()
    except Exception:
        (run / f'error_rank{rank}.txt').write_text(traceback.format_exc())
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
