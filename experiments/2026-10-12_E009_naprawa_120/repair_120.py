"""E009 etap naprawy: rowne dostrajanie 120 epok (BASE/PRE/POST/RANDOM przy 30%).
Eksperyment: 2026-10-12_E009_naprawa_120; protokol: OPIS.md i PROTOKOL_v1.md.
Wejscia: frozen config, modele etapu 1 E008 (stage1/models/*.pt, pinowane hashem),
cache_train.npz i splits.json Tiny ImageNet, baseline E007.
Wyjscia: katalog biegu z metrykami, predykcjami dev i checkpointami per wariant.
Jeden ciagly harmonogram cosine 120 epok; identyczny budzet dla wszystkich wariantow.
"""
import argparse
import copy
import json
import math
import os
import time
import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms as T
from PIL import Image

from support_e009 import (HERE, ROOT, DATA, digest, read_json, write_json, require,
                          model_digest, load_data, loader, base_model, seed_all, seed_worker,
                          eval_transform, verify_inputs, e008_run)
from folding_e008 import materialize_model, count_parameters


def train_transform(cfg):
    size = cfg['image_size']
    return T.Compose([T.RandomResizedCrop(size, scale=tuple(cfg['rrc_scale']),
                                          ratio=tuple(cfg['rrc_ratio']),
                                          interpolation=T.InterpolationMode.BICUBIC,
                                          antialias=True),
                      T.ColorJitter(**cfg['color_jitter']),
                      T.RandomHorizontalFlip(),
                      T.ToTensor(),
                      T.Normalize(cfg['normalization_mean'], cfg['normalization_std'])])


class TrainImages(Dataset):
    def __init__(self, images, labels, indices, transform):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        return self.transform(Image.fromarray(self.images[index])), int(self.labels[index]), index


def lr_factor(epoch, repair):
    warm = repair['warmup_epochs']
    if epoch < warm:
        return (epoch + 1) / warm
    min_factor = repair['min_lr'] / repair['lr']
    progress = (epoch - warm) / max(1, repair['epochs'] - warm)
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_variant(stage1, name, cpu_base, device):
    if name == 'BASE':
        model = copy.deepcopy(cpu_base)
    else:
        state = torch.load(stage1 / 'models' / f'30_{name}.pt', map_location='cpu',
                           weights_only=True)
        model = materialize_model(cpu_base, state['structure_rows'])
        model.load_state_dict(state['state_dict'], strict=True)
    return nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)


def evaluate_logits(model, data_loader, records, rank, device):
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
    require(ids_seen == [r['id'] for r in records['dev'][rank::4]]
            and targets_seen == [r['target'] for r in records['dev'][rank::4]],
            'Dev rank coverage mismatch')
    return stats, {'ids': np.asarray(ids_seen), 'targets': np.asarray(targets_seen),
                   'logits': np.concatenate(logits)}


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
        repair = dict(cfg['repair'])
        repair.update({k: cfg[k] for k in ('image_size', 'normalization_mean',
                                           'normalization_std', 'rrc_scale', 'rrc_ratio',
                                           'color_jitter', 'seed', 'num_classes')})
        if rank == 0:
            write_json(run / 'input_verification.json', verify_inputs(cfg))
        dist.barrier()
        stage1 = e008_run(cfg) / 'stage1'
        stage1_metrics = read_json(stage1 / 'stage1_metrics.json')
        require(stage1_metrics['baseline_reproduced'], 'Stage 1 baseline not reproduced')
        model_files = stage1_metrics['model_files']
        for name in repair['variants']:
            key = 'BASE' if name == 'BASE' else f'30_{name}'
            require(key in model_files, f'Missing stage-1 model file entry: {key}')
        images, labels, records = load_data(cfg)
        train_indices = np.asarray(read_json(DATA / 'splits.json')['splits']['train'])
        require(len(train_indices) == 90000, 'Train split mismatch')
        transform = train_transform(cfg)
        cpu_base = base_model(cfg)
        base_hash = model_digest(cpu_base)
        require(base_hash == model_files['BASE']['model_sha256'], 'Stage 1 BASE mismatch')
        dev_loader = loader(images, labels, records['dev'], cfg, rank,
                            repair['eval_batch_per_gpu'])
        if rank == 0:
            write_json(run / 'repair_plan.json', {
                **{k: repair[k] for k in repair},
                'stage1_dir': str(stage1), 'stage1_model_files': model_files,
                'global_batch': repair['batch_per_gpu'] * 4,
                'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        final_metrics = {}
        for name in repair['variants']:
            seed_all(cfg['seed'] + rank)
            model = build_variant(stage1, name, cpu_base, device)
            parameters = count_parameters(model)
            key = 'BASE' if name == 'BASE' else f'30_{name}'
            require(parameters == stage1_metrics['modes'][key]['parameters'],
                    f'Parameter count mismatch for {name}')
            require(model_digest(model) == model_files[key]['model_sha256'],
                    f'Stage 1 weights mismatch for {name}')
            model = DDP(model, device_ids=[local_rank])
            optimizer = torch.optim.AdamW(model.parameters(), lr=repair['lr'],
                                          weight_decay=repair['weight_decay'])
            scaler = torch.amp.GradScaler('cuda', enabled=repair['amp'])
            sampler = DistributedSampler(
                TrainImages(images, labels, train_indices, transform),
                num_replicas=4, rank=rank, shuffle=True, seed=cfg['seed'], drop_last=True)
            row = None
            for epoch in range(repair['epochs']):
                tick = time.monotonic()
                sampler.set_epoch(epoch)
                factor = lr_factor(epoch, repair)
                for group in optimizer.param_groups:
                    group['lr'] = repair['lr'] * factor
                train_loader = DataLoader(sampler.dataset, batch_size=repair['batch_per_gpu'],
                                          sampler=sampler, num_workers=repair['workers'],
                                          drop_last=True, worker_init_fn=seed_worker,
                                          generator=torch.Generator().manual_seed(
                                              cfg['seed'] + rank + 10000 * epoch),
                                          pin_memory=True)
                model.train()
                stats = torch.zeros(3, dtype=torch.float64, device=device)
                for x, y, _ in train_loader:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.float16, enabled=repair['amp']):
                        logits = model(x)
                        loss = nn.functional.cross_entropy(
                            logits, y, label_smoothing=repair['label_smoothing'])
                        plain_nll = nn.functional.cross_entropy(logits, y, reduction='sum')
                    require(bool(torch.isfinite(loss)), 'Non-finite training loss')
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), repair['grad_clip'])
                    scaler.step(optimizer)
                    scaler.update()
                    stats += torch.stack([plain_nll.detach().double(),
                                          (logits.argmax(1) == y).sum().double(),
                                          torch.tensor(y.numel(), device=device).double()])
                dist.all_reduce(stats)
                for buffer in model.module.buffers():
                    dist.broadcast(buffer, src=0)
                val_stats, _ = evaluate_logits(model, dev_loader, records, rank, device)
                val = torch.tensor(val_stats, dtype=torch.float64, device=device)
                dist.all_reduce(val)
                row = {'variant': name, 'epoch': epoch + 1,
                       'lr': optimizer.param_groups[0]['lr'],
                       'train_nll': float(stats[0] / stats[2]),
                       'train_correct': int(stats[1]), 'train_count': int(stats[2]),
                       'dev_nll': float(val[0] / val[2]), 'dev_correct': int(val[1]),
                       'dev_count': int(val[2]), 'dev_top1': float(val[1] / val[2]),
                       'elapsed_seconds': time.monotonic() - tick}
                if rank == 0:
                    with (run / f'metrics_{name}.jsonl').open('a') as stream:
                        stream.write(json.dumps(row) + '\n')
                    print(json.dumps(row), flush=True)
                    if (epoch + 1) % 10 == 0 and epoch + 1 < repair['epochs']:
                        temporary = run / f'partial_{name}.pt.tmp'
                        torch.save({'variant': name, 'epoch': epoch + 1,
                                    'model': model.module.state_dict(), 'config': cfg}, temporary)
                        os.replace(temporary, run / f'partial_{name}.pt')
                dist.barrier()
            _, preds = evaluate_logits(model, dev_loader, records, rank, device)
            np.savez_compressed(run / f'dev_final_{name}_rank{rank}.npz', **preds)
            dist.barrier()
            if rank == 0:
                merged = []
                for r in range(4):
                    with np.load(run / f'dev_final_{name}_rank{r}.npz', allow_pickle=False) as f:
                        merged.append({k: f[k].copy() for k in f.files})
                joined = {k: np.concatenate([v[k] for v in merged]) for k in merged[0]}
                order = np.argsort(joined['ids'])
                joined = {k: v[order] for k, v in joined.items()}
                require(joined['ids'].tolist() == [r['id'] for r in records['dev']]
                        and len(np.unique(joined['ids'])) == cfg['data']['dev_count'],
                        f'Global dev coverage mismatch for {name}')
                np.savez_compressed(run / f'dev_final_{name}.npz', **joined)
                model_sha = model_digest(model.module)
                payload = {'variant': name, 'epochs': repair['epochs'],
                           'model': model.module.state_dict(), 'config': cfg,
                           'parameters': parameters, 'model_sha256': model_sha}
                torch.save(payload, run / f'last_{name}.pt')
                final_metrics[name] = {'epochs': repair['epochs'], 'parameters': parameters,
                                       'last_row': row, 'model_sha256': model_sha,
                                       'last_sha256': digest(run / f'last_{name}.pt')}
                write_json(run / f'variant_{name}.json', final_metrics[name])
            dist.barrier()
            del model, optimizer, scaler
            torch.cuda.empty_cache()
        require(model_digest(cpu_base) == base_hash, 'Base altered during repair')
        if rank == 0:
            write_json(run / 'repair_metrics.json', {
                'variants': final_metrics,
                'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
            print('E009_REPAIR_METRICS=' + json.dumps(
                {k: {'dev_top1': v['last_row']['dev_top1'], 'dev_nll': v['last_row']['dev_nll']}
                 for k, v in final_metrics.items()}), flush=True)
    except Exception:
        (run / f'error_rank{rank}.txt').write_text(traceback.format_exc())
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
