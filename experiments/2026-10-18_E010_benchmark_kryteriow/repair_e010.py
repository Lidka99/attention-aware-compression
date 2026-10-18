"""E010: krotkie naprawy T epok dla wszystkich kryteriow (4 GPU, DDP).

Eksperyment: 2026-10-18_E010_benchmark_kryteriow; protokol: OPIS.md i PROTOKOL_v1.md.
Wejscia: modele scalone z merge_e010.py, dane train/dev, recipe E009
(AdamW, cosine per T, warmup min(3,T), AMP, batch 32/GPU).
Wyjscia: repair/<dataset>_<ratio>_<kryterium>_T<T>/ z metrics.jsonl,
dev_final*.npz i last.pt. Wynik = ostatnia epoka; bez early stoppingu.
"""
import argparse
import math
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from support_e010 import (dataset_cfg, plan_spec, plan_rows, base_model, load_dataset,
                          loader, ImagesWithIds, train_transform, materialize_model,
                          count_parameters, model_digest, read_json, write_json,
                          require, seed_all, seed_worker, load_config, DATASET_KEYS)


def lr_factor(epoch, epochs, warmup, min_lr, lr):
    if epoch < warmup:
        return (epoch + 1) / warmup
    min_factor = min_lr / lr
    progress = (epoch - warmup) / max(1, epochs - warmup)
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))


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
            images, labels, records, train_indices = load_dataset(ds)
            base = base_model(ds)
            train_dataset = ImagesWithIds(images, labels, train_indices, train_transform(ds))
            dev_batch = int(ds['repair']['eval_batch_per_gpu'])
            for ratio, spec in ds['criteria']['ratios'].items():
                rows = plan_rows(ds, ratio)
                plan = plan_spec(ds, ratio)
                for criterion in spec['criteria']:
                    name = f'{key}_{ratio}_{criterion}'
                    payload_file = run / 'merge' / 'models' / f'{name}.pt'
                    require(payload_file.exists(), f'Missing merged model: {payload_file}')
                    for epochs in spec['repair_epochs']:
                        cell = run / 'repair' / f'{name}_T{epochs}'
                        cell.mkdir(parents=True, exist_ok=True)
                        payload = torch.load(payload_file, map_location='cpu', weights_only=True)
                        model = materialize_model(base, rows, int(plan['base_parameters']))
                        model.load_state_dict(payload['state_dict'], strict=True)
                        require(count_parameters(model) == int(plan['compressed_parameters']),
                                f'{name}: parameter budget mismatch')
                        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
                        model = DDP(model, device_ids=[local_rank])
                        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['repair']['lr'],
                                                      weight_decay=cfg['repair']['weight_decay'])
                        scaler = torch.amp.GradScaler('cuda', enabled=bool(cfg['repair']['amp']))
                        seed_all(int(cfg['seed']) + rank)
                        sampler = DistributedSampler(train_dataset, num_replicas=4, rank=rank,
                                                     shuffle=True, seed=int(cfg['seed']),
                                                     drop_last=True)
                        row = None
                        for epoch in range(epochs):
                            tick = time.monotonic()
                            sampler.set_epoch(epoch)
                            factor = lr_factor(epoch, epochs, min(3, epochs),
                                               cfg['repair']['min_lr'], cfg['repair']['lr'])
                            for group in optimizer.param_groups:
                                group['lr'] = cfg['repair']['lr'] * factor
                            train_loader = DataLoader(
                                train_dataset, batch_size=int(ds['repair']['batch_per_gpu']),
                                sampler=sampler, num_workers=int(ds['workers']),
                                drop_last=True, worker_init_fn=seed_worker,
                                generator=torch.Generator().manual_seed(
                                    int(cfg['seed']) + rank + 10000 * epoch),
                                pin_memory=True)
                            model.train()
                            stats = torch.zeros(3, dtype=torch.float64, device=device)
                            for x, y, _ in train_loader:
                                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                                optimizer.zero_grad(set_to_none=True)
                                with torch.autocast('cuda', dtype=torch.float16,
                                                    enabled=bool(cfg['repair']['amp'])):
                                    logits = model(x)
                                    loss = nn.functional.cross_entropy(
                                        logits, y, label_smoothing=cfg['repair']['label_smoothing'])
                                    plain = nn.functional.cross_entropy(logits, y, reduction='sum')
                                require(bool(torch.isfinite(loss)), 'Non-finite training loss')
                                scaler.scale(loss).backward()
                                scaler.unscale_(optimizer)
                                nn.utils.clip_grad_norm_(model.parameters(), cfg['repair']['grad_clip'])
                                scaler.step(optimizer)
                                scaler.update()
                                stats += torch.stack([plain.detach().double(),
                                                      (logits.argmax(1) == y).sum().double(),
                                                      torch.tensor(y.numel(), device=device).double()])
                            dist.all_reduce(stats)
                            for buffer in model.module.buffers():
                                dist.broadcast(buffer, src=0)
                            val_stats, _ = evaluate(model, loader(images, labels, records['dev'],
                                                                  ds, rank, dev_batch),
                                                    records['dev'], rank, device)
                            val = torch.tensor(val_stats, dtype=torch.float64, device=device)
                            dist.all_reduce(val)
                            row = {'dataset': key, 'ratio': ratio, 'criterion': criterion,
                                   'T': epochs, 'epoch': epoch + 1,
                                   'lr': optimizer.param_groups[0]['lr'],
                                   'train_nll': float(stats[0] / stats[2]),
                                   'train_correct': int(stats[1]), 'train_count': int(stats[2]),
                                   'dev_nll': float(val[0] / val[2]), 'dev_correct': int(val[1]),
                                   'dev_count': int(val[2]),
                                   'dev_top1': float(val[1] / val[2]),
                                   'elapsed_seconds': time.monotonic() - tick}
                            if rank == 0:
                                with (cell / 'metrics.jsonl').open('a') as stream:
                                    stream.write(__import__('json').dumps(row) + '\n')
                                print(__import__('json').dumps(row), flush=True)
                            dist.barrier()
                        _, preds = evaluate(model, loader(images, labels, records['dev'],
                                                          ds, rank, dev_batch),
                                            records['dev'], rank, device)
                        np.savez_compressed(cell / f'dev_final_rank{rank}.npz', **preds)
                        dist.barrier()
                        if rank == 0:
                            merged = []
                            for r in range(4):
                                with np.load(cell / f'dev_final_rank{r}.npz',
                                             allow_pickle=False) as f:
                                    merged.append({k: f[k].copy() for k in f.files})
                            joined = {k: np.concatenate([v[k] for v in merged]) for k in merged[0]}
                            order = np.argsort(joined['ids'])
                            joined = {k: v[order] for k, v in joined.items()}
                            require(joined['ids'].tolist() == [r['id'] for r in records['dev']]
                                    and len(np.unique(joined['ids'])) == int(ds['data']['dev_count']),
                                    f'Global dev coverage mismatch for {name} T={epochs}')
                            np.savez_compressed(cell / 'dev_final.npz', **joined)
                            torch.save({'variant': name, 'dataset': key, 'ratio': ratio,
                                        'criterion': criterion, 'T': epochs,
                                        'structure_rows': rows,
                                        'parameters': int(plan['compressed_parameters']),
                                        'model_sha256': model_digest(model.module),
                                        'last_row': row, 'state_dict': model.module.state_dict()},
                                       cell / 'last.pt')
                            print(f'E010_REPAIR={name}_T{epochs} '
                                  f'top1={row["dev_top1"]:.4f}', flush=True)
                        dist.barrier()
                        del model, optimizer, scaler
                        torch.cuda.empty_cache()
            del base, train_dataset
            torch.cuda.empty_cache()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
