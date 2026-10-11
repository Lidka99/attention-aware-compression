"""E007: trening EfficientNet-B0 od zera na Tiny ImageNet, 224x224, 4 GPU (DDP).
Eksperyment: 2026-10-11_E007_baseline_224; protokół: OPIS.md (po zamrożeniu
także PROTOKOL_v1.md). Receptura oparta na Shahriar 2026 (arXiv:2607.01984),
zaadaptowana na 4 GPU (batch globalny 128, wd 0,01, warmup 5); bez mixup.
Wejścia: cache_train.npz, cache_val.npz i splits.json w data/tiny-imagenet,
config.json oraz tryb --smoke (podzbiór, wyłącznie kontrola techniczna).
Wyjścia: nowy katalog biegu z manifestem, metrykami, git logiem, last.pt,
partial_checkpoint.pt (rolling) i metrykami końcowymi.
"""
import argparse
import json
import math
import os
import random
import sys
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
from torchvision.models import efficientnet_b0
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'data/tiny-imagenet'
GLOBAL_BATCH_MULTIPLIER = 4


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class CachedImages(Dataset):
    def __init__(self, images, labels, indices, transform):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        image = Image.fromarray(self.images[index])
        return self.transform(image), int(self.labels[index]), index


def build_transforms(cfg):
    size = cfg['image_size']
    train = T.Compose([T.RandomResizedCrop(size, scale=tuple(cfg['rrc_scale']),
                                           ratio=tuple(cfg['rrc_ratio']),
                                           interpolation=T.InterpolationMode.BICUBIC,
                                           antialias=True),
                       T.ColorJitter(**cfg['color_jitter']),
                       T.RandomHorizontalFlip(),
                       T.ToTensor(),
                       T.Normalize(cfg['normalization_mean'], cfg['normalization_std'])])
    dev = T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC,
                              antialias=True),
                     T.ToTensor(),
                     T.Normalize(cfg['normalization_mean'], cfg['normalization_std'])])
    return train, dev


def lr_factor(epoch, cfg):
    warm = cfg['warmup_epochs']
    if epoch < warm:
        return (epoch + 1) / warm
    min_factor = cfg['min_lr'] / cfg['lr']
    progress = (epoch - warm) / max(1, cfg['epochs'] - warm)
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(path, model, epoch, cfg):
    payload = {'epoch': epoch, 'model': model.state_dict(), 'config': cfg}
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true',
                        help='Kontrola techniczna na podzbiorze; nie jest wynikiem naukowym.')
    parser.add_argument('--subset-per-class', type=int, default=50)
    parser.add_argument('--dev-per-class', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()
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
    run = args.run_dir.resolve()
    require(run.is_dir() and not (run / 'metrics.jsonl').exists()
            and not (run / 'metrics.json').exists(),
            'Use a fresh run directory without previous metrics')
    try:
        cfg = json.loads(args.config.read_text())
        if args.epochs:
            cfg['epochs'] = args.epochs
        splits = json.loads((DATA / 'splits.json').read_text())
        require(splits['dataset'] == 'TinyImageNet' and len(splits['classes']) == 200,
                'Unexpected splits manifest')
        train_cache = np.load(DATA / 'cache_train.npz', allow_pickle=False)
        val_cache = np.load(DATA / 'cache_val.npz', allow_pickle=False)
        train_images, train_labels = train_cache['images'], train_cache['labels']
        val_images, val_labels = val_cache['images'], val_cache['labels']
        require(train_images.shape == (100000, 64, 64, 3)
                and val_images.shape == (10000, 64, 64, 3), 'Cache shape mismatch')
        train_idx = np.asarray(splits['splits']['train'])
        dev_idx = np.asarray(splits['splits']['dev'])
        if args.smoke:
            subset = args.subset_per_class
            dev_subset = args.dev_per_class
            train_idx = np.concatenate([train_idx[c * 450:c * 450 + subset]
                                        for c in range(200)])
            dev_idx = np.concatenate([dev_idx[c * 50:c * 50 + dev_subset] for c in range(200)])
        train_transform, dev_transform = build_transforms(cfg)
        seed_all(cfg['seed'] + rank)
        model = efficientnet_b0(weights=None, num_classes=cfg['num_classes'])
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = DDP(model, device_ids=[local_rank])
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                                      weight_decay=cfg['weight_decay'])
        scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'])
        sampler = DistributedSampler(CachedImages(train_images, train_labels, train_idx,
                                                  train_transform),
                                     num_replicas=4, rank=rank, shuffle=True,
                                     seed=cfg['seed'], drop_last=True)
        dev_records = CachedImages(train_images, train_labels, dev_idx[rank::4], dev_transform)
        if rank == 0:
            manifest = {'experiment': cfg['experiment'], 'smoke': args.smoke,
                        'config': cfg, 'argv': sys.argv,
                        'splits_sha256': digest(DATA / 'splits.json'),
                        'cache_train_sha256': digest(DATA / 'cache_train.npz'),
                        'cache_val_sha256': digest(DATA / 'cache_val.npz'),
                        'train_count': int(len(train_idx)), 'dev_count': int(len(dev_idx)),
                        'world_size': 4, 'batch_per_gpu': cfg['batch_per_gpu'],
                        'global_batch': cfg['batch_per_gpu'] * GLOBAL_BATCH_MULTIPLIER,
                        'image_size': cfg['image_size'], 'seed': cfg['seed'],
                        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                        'git_sha': os.popen('git rev-parse HEAD').read().strip()}
            write_json(run / 'manifest.json', manifest)
        dev_loader = DataLoader(dev_records, batch_size=cfg['eval_batch_per_gpu'], shuffle=False,
                                drop_last=False, num_workers=cfg['workers'],
                                worker_init_fn=seed_worker, pin_memory=True,
                                generator=torch.Generator().manual_seed(cfg['seed'] + rank + 100000))
        row = None
        for epoch in range(cfg['epochs']):
            tick = time.monotonic()
            sampler.set_epoch(epoch)
            factor = lr_factor(epoch, cfg)
            for group in optimizer.param_groups:
                group['lr'] = cfg['lr'] * factor
            loader = DataLoader(sampler.dataset, batch_size=cfg['batch_per_gpu'], sampler=sampler,
                                num_workers=cfg['workers'], drop_last=True,
                                worker_init_fn=seed_worker,
                                generator=torch.Generator().manual_seed(cfg['seed'] + rank + 10000 * epoch),
                                pin_memory=True)
            model.train()
            stats = torch.zeros(3, dtype=torch.float64, device=device)
            for x, y, _ in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16, enabled=cfg['amp']):
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits, y,
                                                       label_smoothing=cfg['label_smoothing'])
                    plain_nll = nn.functional.cross_entropy(logits, y, reduction='sum')
                require(bool(torch.isfinite(loss)), 'Non-finite training loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
                stats += torch.stack([plain_nll.detach().double(),
                                      (logits.argmax(1) == y).sum().double(),
                                      torch.tensor(y.numel(), device=device).double()])
            dist.all_reduce(stats)
            for buffer in model.module.buffers():
                dist.broadcast(buffer, src=0)
            model.eval()
            val = torch.zeros(3, dtype=torch.float64, device=device)
            with torch.inference_mode():
                for x, y, _ in dev_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    require(bool(torch.isfinite(logits).all()), 'Non-finite dev logits')
                    val += torch.stack([nn.functional.cross_entropy(logits, y, reduction='sum').double(),
                                        (logits.argmax(1) == y).sum().double(),
                                        torch.tensor(y.numel(), device=device).double()])
            dist.all_reduce(val)
            row = {'epoch': epoch + 1, 'lr': optimizer.param_groups[0]['lr'],
                   'train_nll': float(stats[0] / stats[2]), 'train_correct': int(stats[1]),
                   'train_count': int(stats[2]), 'dev_nll': float(val[0] / val[2]),
                   'dev_correct': int(val[1]), 'dev_count': int(val[2]),
                   'dev_top1': float(val[1] / val[2]), 'elapsed_seconds': time.monotonic() - tick}
            if args.smoke:
                row['peak_cuda_mib'] = torch.cuda.max_memory_allocated(device) // (1024 * 1024)
                print(f'rank={rank} smoke_peak_cuda_mib={row["peak_cuda_mib"]}', flush=True)
            if rank == 0:
                with (run / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                print(json.dumps(row), flush=True)
                if not args.smoke and (epoch + 1) % cfg['checkpoint_every'] == 0:
                    save_checkpoint(run / 'partial_checkpoint.pt', model.module,
                                    epoch + 1, cfg)
            dist.barrier()
        if rank == 0:
            torch.save({'epoch': cfg['epochs'], 'model': model.module.state_dict(),
                        'config': cfg, 'smoke': args.smoke}, run / 'last.pt')
            metrics = {'epochs': cfg['epochs'], 'smoke': args.smoke, 'final': row,
                       'last_pt_sha256': digest(run / 'last.pt')}
            if (run / 'partial_checkpoint.pt').exists():
                metrics['partial_checkpoint_sha256'] = digest(run / 'partial_checkpoint.pt')
            write_json(run / 'metrics.json', metrics)
            print('E007_TRAINING=completed', flush=True)
    except Exception:
        (run / f'error_rank{rank}.txt').write_text(traceback.format_exc())
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
