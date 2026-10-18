"""E010: scalanie nowych kryteriow i zero-shot dev na 4 GPU.

Eksperyment: 2026-10-18_E010_benchmark_kryteriow; protokol: OPIS.md i PROTOKOL_v1.md.
Wejscia: config frozen, model bazowy, plany per ratio, grupy z criteria_e010.py,
modele etapu 1 E008/E003 (reuzyte).
Wyjscia: merge/models/<dataset>_<ratio>_<kryterium>.pt, merge/dev/*.npz,
merge/zero_metrics.json. Bramki: parametry, pokrycie dev, odtworzenie
top-1/NLL modeli reuzytych i BASE (top-1 dokladnie, |dNLL| <= 1e-5).
"""
import argparse
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from support_e010 import (dataset_cfg, plan_rows, plan_spec, base_model, load_dataset,
                          loader, materialize_model, fold_model, count_parameters,
                          model_digest, digest, read_json, write_json, require,
                          load_config, DATASET_KEYS)


def evaluate(model, data_loader, records, rank, device):
    model.eval()
    stats = [0.0, 0, 0]
    logits, ids_seen, targets_seen = [], [], []
    with torch.inference_mode():
        for x, y, ids in data_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            require(output.dtype == torch.float32 and bool(torch.isfinite(output).all()),
                    'Non-finite dev logits')
            stats[0] += float(nn.functional.cross_entropy(output, y, reduction='sum').double())
            stats[1] += int((output.argmax(1) == y).sum())
            stats[2] += len(ids)
            logits.append(output.cpu().numpy().copy())
            ids_seen.extend(ids.tolist())
            targets_seen.extend(y.tolist())
    require(ids_seen == [r['id'] for r in records[rank::4]]
            and targets_seen == [r['target'] for r in records[rank::4]],
            'Dev rank coverage mismatch')
    return stats, {'ids': np.asarray(ids_seen), 'targets': np.asarray(targets_seen),
                   'logits': np.concatenate(logits)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run = args.run_dir.resolve()
    (run / 'merge' / 'models').mkdir(parents=True, exist_ok=True)
    (run / 'merge' / 'dev').mkdir(parents=True, exist_ok=True)
    rank = int(__import__('os').environ['RANK'])
    local_rank = int(__import__('os').environ['LOCAL_RANK'])
    require(int(__import__('os').environ['WORLD_SIZE']) == 4 and torch.cuda.device_count() == 4,
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
    zero_metrics = {}
    try:
        for key in DATASET_KEYS:
            ds = dataset_cfg(cfg, key)
            images, labels, records, _train = load_dataset(ds)
            base = base_model(ds)
            if key == 'tiny':
                reference = read_json(Path(ds['run']['file']) / 'stage1' / 'stage1_metrics.json')['modes']
                ref_base = reference['BASE']
            else:
                reference = read_json(Path(ds['run']['file']) / 'raw_metrics.json')['modes']
                ref_base = reference['BASE']
            dev_batch = int(ds['repair']['eval_batch_per_gpu'])

            def run_cell(name, model, ref=None):
                model = model.to(device).eval()
                started = time.time()
                stats, preds = evaluate(model, loader(images, labels, records['dev'], ds, rank, dev_batch),
                                        records['dev'], rank, device)
                np.savez_compressed(run / 'merge' / 'dev' / f'{name}_rank{rank}.npz', **preds)
                dist.barrier()
                if rank == 0:
                    merged = []
                    for r in range(4):
                        with np.load(run / 'merge' / 'dev' / f'{name}_rank{r}.npz',
                                     allow_pickle=False) as f:
                            merged.append({k: f[k].copy() for k in f.files})
                    joined = {k: np.concatenate([v[k] for v in merged]) for k in merged[0]}
                    order = np.argsort(joined['ids'])
                    joined = {k: v[order] for k, v in joined.items()}
                    require(joined['ids'].tolist() == [r['id'] for r in records['dev']]
                            and len(np.unique(joined['ids'])) == int(ds['data']['dev_count']),
                            f'Global dev coverage mismatch for {name}')
                    np.savez_compressed(run / 'merge' / 'dev' / f'{name}.npz', **joined)
                    total = stats[2]
                    row = {'dataset': key, 'name': name, 'correct': int(stats[1]),
                           'count': int(total), 'nll': float(stats[0] / total),
                           'top1': float(stats[1] / total), 'seconds': time.time() - started}
                    if ref is not None:
                        require(int(stats[1]) == int(ref['correct']),
                                f'{name}: top-1 does not reproduce reference')
                        require(abs(float(stats[0] / total) - float(ref['nll'])) <= 1e-5,
                                f'{name}: NLL does not reproduce reference')
                        row['reference_top1'] = float(ref['top1'])
                    zero_metrics[name] = row
                    print(f'E010_ZERO={name} top1={row["top1"]:.4f} '
                          f'nll={row["nll"]:.4f}', flush=True)
                dist.barrier()
                return model

            for ratio, spec in ds['criteria']['ratios'].items():
                rows = plan_rows(ds, ratio)
                plan = plan_spec(ds, ratio)
                for criterion in spec['criteria']:
                    name = f'{key}_{ratio}_{criterion}'
                    if criterion in ds['criteria']['reused']:
                        entry = ds['models'][f'{ratio}_{criterion}']
                        payload = torch.load(Path(entry['file']), map_location='cpu',
                                             weights_only=True)
                        model = materialize_model(base, rows, int(plan['base_parameters']))
                        model.load_state_dict(payload['state_dict'], strict=True)
                        require(count_parameters(model) == int(plan['compressed_parameters']),
                                f'{name}: parameter budget mismatch')
                        if rank == 0:
                            torch.save({'variant': name, 'dataset': key, 'ratio': ratio,
                                        'criterion': criterion, 'compression_applied': True,
                                        'structure_rows': rows,
                                        'groups_sha256': ds['candidate_manifest']['sha256'],
                                        'parameters': count_parameters(model),
                                        'model_sha256': model_digest(model),
                                        'state_dict': model.state_dict()},
                                       run / 'merge' / 'models' / f'{name}.pt')
                        dist.barrier()
                        run_cell(name, model, reference[f'{ratio}_{criterion}']
                                 if key == 'tiny' else reference[criterion])
                    else:
                        groups = read_json(run / 'groups' / f'{key}_{ratio}_{criterion}.json')
                        model = fold_model(base, rows, groups['groups'],
                                           int(plan['base_parameters']))
                        require(count_parameters(model) == int(plan['compressed_parameters']),
                                f'{name}: parameter budget mismatch')
                        if rank == 0:
                            payload = {'variant': name, 'dataset': key, 'ratio': ratio,
                                       'criterion': criterion, 'compression_applied': True,
                                       'structure_rows': rows,
                                       'groups_sha256': digest(
                                           run / 'groups' / f'{key}_{ratio}_{criterion}.json'),
                                       'parameters': count_parameters(model),
                                       'model_sha256': model_digest(model),
                                       'state_dict': model.state_dict()}
                            torch.save(payload, run / 'merge' / 'models' / f'{name}.pt')
                        dist.barrier()
                        run_cell(name, model)
                    del model
                    torch.cuda.empty_cache()
            run_cell(f'{key}_BASE', base, ref_base)
            del base
            torch.cuda.empty_cache()
        if rank == 0:
            write_json(run / 'merge' / 'zero_metrics.json', zero_metrics)
            print('E010_ZERO_METRICS=' + str(len(zero_metrics)), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
