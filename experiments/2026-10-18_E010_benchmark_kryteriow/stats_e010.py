"""E010: statystyki APoZ i Taylor na zbiorze kalibracyjnym A (4 GPU).

Eksperyment: 2026-10-18_E010_benchmark_kryteriow; protokol: OPIS.md i PROTOKOL_v1.md.
Wejscia: config frozen, baseline E007/E001, dane A (5/klase), 4 GPU.
Wyjscia: stats/<dataset>.npz (b{i}_apoz, b{i}_taylor, b{i}_count),
stats/<dataset>.json (settings, pokrycie ID, hashe) oraz czastki rank.
Miejsce pomiaru: wyjscie expansion conv+BN+SiLU (wejscie depthwise).
Bez treningu i bez zmiany wag.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from datetime import timedelta

from support_e010 import (dataset_cfg, plan_rows, base_model, load_dataset, loader,
                          digest, require, read_json, write_json, seed_worker,
                          load_config, DATASET_KEYS)


class Capture:
    def __init__(self, model, rows):
        self.values = {}
        self.handles = []
        for row in rows:
            module = model.get_submodule(row['block']).block[0]
            self.handles.append(module.register_forward_hook(self._hook(row['block'])))

    def _hook(self, name):
        def hook(module, inputs, output):
            if not torch.is_grad_enabled():
                raise RuntimeError('Stats forward must run with grad enabled')
            output.retain_grad()
            self.values[name] = output
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run = args.run_dir.resolve()
    (run / 'stats').mkdir(parents=True, exist_ok=True)
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
    try:
        for key in DATASET_KEYS:
            ds = dataset_cfg(cfg, key)
            rows = plan_rows(ds, sorted(ds['criteria']['ratios'])[0])
            base = base_model(ds).to(device).eval()
            for parameter in base.parameters():
                parameter.requires_grad_(True)
            images, labels, records, _train = load_dataset(ds)
            batch = int(cfg['statistics']['batch_per_gpu'])
            data_loader = loader(images, labels, records['A'], ds, rank, batch)
            capture = Capture(base, rows)
            accum = {row['block']: {
                'zeros': torch.zeros(int(row['expanded_channels']), dtype=torch.float64, device=device),
                'spatial': torch.zeros(1, dtype=torch.float64, device=device),
                'taylor': torch.zeros(int(row['expanded_channels']), dtype=torch.float64, device=device),
                'samples': torch.zeros(1, dtype=torch.float64, device=device)}
                for row in rows}
            ids_seen, targets_seen = [], []
            start = time.time()
            try:
                for x, y, ids in data_loader:
                    capture.values.clear()
                    x, y = x.to(device), y.to(device)
                    logits = base(x)
                    loss = nn.functional.cross_entropy(logits, y)
                    loss.backward()
                    for row in rows:
                        name = row['block']
                        activation = capture.values[name]
                        gradient = activation.grad
                        require(activation is not None and gradient is not None,
                                f'Missing capture/gradient for {name}')
                        accum[name]['zeros'] += (activation.detach() == 0).sum(
                            dim=(0, 2, 3)).double()
                        accum[name]['spatial'] += float(
                            activation.shape[0] * activation.shape[2] * activation.shape[3])
                        accum[name]['taylor'] += (gradient.detach() * activation.detach()).sum(
                            dim=(2, 3)).abs().sum(dim=0).double()
                        accum[name]['samples'] += float(activation.shape[0])
                    for parameter in base.parameters():
                        parameter.grad = None
                    ids_seen.extend(ids.tolist())
                    targets_seen.extend(y.tolist())
            finally:
                capture.close()
            require(ids_seen == [r['id'] for r in records['A'][rank::4]]
                    and targets_seen == [r['target'] for r in records['A'][rank::4]],
                    'Calibration coverage mismatch')
            local = {}
            for index, row in enumerate(rows):
                name = row['block']
                local[f'b{index}_zeros'] = accum[name]['zeros'].cpu().numpy()
                local[f'b{index}_spatial'] = accum[name]['spatial'].cpu().numpy()
                local[f'b{index}_taylor'] = accum[name]['taylor'].cpu().numpy()
                local[f'b{index}_samples'] = accum[name]['samples'].cpu().numpy()
            np.savez(run / 'stats' / f'{key}_rank{rank}.npz', **local)
            write_json(run / 'stats' / f'{key}_rank{rank}.json',
                       {'rank': rank, 'ids': ids_seen, 'targets': targets_seen,
                        'count': len(ids_seen), 'dataset': key})
            for row in rows:
                name = row['block']
                for field in ('zeros', 'spatial', 'taylor', 'samples'):
                    dist.all_reduce(accum[name][field])
            if rank == 0:
                arrays, meta = {}, {'settings': cfg['statistics'] | {
                    'kmeans_seed': int(cfg['bootstrap']['seed']),
                    'dataset': key, 'calibration_count': int(len(records['A']))},
                    'dataset': key, 'count': int(accum[rows[0]['block']]['samples'].item()),
                    'base_sha256': ds['baseline']['sha256'],
                    'splits_sha256': ds['data']['splits']['sha256'],
                    'source': 'E010 stats_e010.py'}
                for index, row in enumerate(rows):
                    name = row['block']
                    zeros = accum[name]['zeros'].cpu().numpy()
                    spatial = float(accum[name]['spatial'].item())
                    taylor = accum[name]['taylor'].cpu().numpy()
                    samples = float(accum[name]['samples'].item())
                    require(spatial > 0 and samples > 0, f'Empty stats for {name}')
                    arrays[f'b{index}_apoz'] = zeros / spatial
                    arrays[f'b{index}_taylor'] = taylor / samples
                    arrays[f'b{index}_count'] = np.asarray([samples], dtype=np.float64)
                np.savez(run / 'stats' / f'{key}.npz', **arrays)
                write_json(run / 'stats' / f'{key}.json', meta)
                print(f'E010_STATS={key} samples={meta["count"]} '
                      f'seconds={time.time() - start:.1f}', flush=True)
            dist.barrier()
            del base, data_loader, accum
            torch.cuda.empty_cache()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
