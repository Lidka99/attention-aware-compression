"""E001: trenuje EfficientNet-B0 od losowych wag na dokładnie 4 GPU (DDP).
Wejścia: config.json, CIFAR-100 i splits.json; opcjonalnie własny checkpoint.
Wyjścia: manifest, metryki, predykcje dev, checkpointy i stany losowości.
--smoke używa wyłącznie danych syntetycznych do testu infrastruktury.
Protokół, ograniczenia i instrukcje: OPIS.md i URUCHOMIENIE.md obok tego pliku.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torchvision
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset, TensorDataset
from torchvision import transforms as T
from torchvision.datasets import CIFAR100
from torchvision.models import efficientnet_b0
from prepare_data import digest

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def model_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rng_state(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device)}


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state(state['cuda'], device)


def environment():
    return {
        'python': sys.version, 'platform': platform.platform(),
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'numpy': np.__version__, 'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'gpu': [torch.cuda.get_device_name(i) for i in range(4)],
        'gpu_memory': [torch.cuda.get_device_properties(i).total_memory for i in range(4)],
        'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
                                          '--format=csv,noheader'], text=True).strip(),
        'deterministic_algorithms': True, 'tf32': False,
        'env': {k: os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES',
                'CUBLAS_WORKSPACE_CONFIG', 'PYTHONHASHSEED', 'OMP_NUM_THREADS']},
    }


def datasets(cfg, data_root, smoke):
    if smoke:
        g = torch.Generator().manual_seed(9876)
        def synthetic(n):
            return TensorDataset(torch.randn(n, 3, cfg['image_size'], cfg['image_size'], generator=g),
                                 torch.randint(100, (n,), generator=g))
        return synthetic(256), synthetic(64), list(range(64)), {'synthetic': True}
    manifest_path = data_root / 'splits.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['split_seed'] == cfg['split_seed']
    for name, expected in manifest['files_sha256'].items():
        if digest(data_root / name) != expected:
            raise RuntimeError(f'Data hash mismatch: {name}')
    splits = manifest['splits']
    flat = sum(splits.values(), [])
    assert len(flat) == len(set(flat)) == 50000
    assert sorted(flat) == list(range(50000))
    assert [len(splits[k]) for k in ['train', 'dev', 'reserve']] == [40000, 5000, 5000]
    normalize = T.Normalize(cfg['normalization_mean'], cfg['normalization_std'])
    resize = T.Resize((cfg['image_size'], cfg['image_size']), interpolation=T.InterpolationMode.BILINEAR,
                      antialias=True)
    train_transform = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                                 resize, T.ToTensor(), normalize])
    dev_transform = T.Compose([resize, T.ToTensor(), normalize])
    train = CIFAR100(data_root, train=True, transform=train_transform, download=False)
    dev = CIFAR100(data_root, train=True, transform=dev_transform, download=False)
    return Subset(train, splits['train']), Subset(dev, splits['dev']), splits['dev'], {
        'splits_sha256': digest(manifest_path), **manifest}


def checkpoint(path, state):
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/cifar100')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', type=Path, help='Trusted checkpoint created by this script only.')
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', '-1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '-1'))
    if int(os.environ.get('WORLD_SIZE', '0')) != 4 or torch.cuda.device_count() != 4:
        raise RuntimeError('Exactly 4 visible GPUs and torchrun --nproc_per_node=4 required.')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(minutes=5), device_id=device)
    run = args.run_dir.resolve()
    cfg = json.loads(args.config.read_text())
    assert cfg['weights'] is None and cfg['model'] == 'efficientnet_b0'
    assert cfg['world_size'] == 4 and cfg['num_classes'] == 100 and cfg['dataset'] == 'CIFAR100'
    if args.smoke:
        cfg.update(epochs=2, workers=0, warmup_epochs=2)
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    if dirty and not args.smoke:
        raise RuntimeError('Commit changes before scientific training; working tree is dirty.')
    # launcher creates the unique directory and stdout/stderr logs, never reuses it.
    if not run.is_dir() or (run / 'manifest.json').exists():
        raise RuntimeError('Use launch.sh with a fresh output directory.')
    env = environment()
    manifest = {'experiment': cfg['experiment'], 'run_id': run.name, 'status': 'running',
                'started_utc': datetime.now(timezone.utc).isoformat(), 'git_sha': git_sha,
                'dirty': bool(dirty), 'smoke': args.smoke, 'config_sha256': config_hash,
                'argv': sys.argv, 'cwd': str(Path.cwd()),
                'source_sha256': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
                'resume': str(args.resume.resolve()) if args.resume else None}
    try:
        if rank == 0:
            write_json(run / 'manifest.json', manifest)
            write_json(run / 'config.resolved.json', cfg)
            write_json(run / 'environment.json', env)
            (run / 'pip-freeze.txt').write_text(subprocess.check_output(
                [sys.executable, '-m', 'pip', 'freeze'], text=True))
        train, dev, dev_ids, data_manifest = datasets(cfg, args.data_root.resolve(), args.smoke)
        data_hash = hashlib.sha256(json.dumps(data_manifest, sort_keys=True).encode()).hexdigest()
        if rank == 0:
            write_json(run / 'data_manifest.json', data_manifest)
        seed_all(cfg['seed'])
        base = efficientnet_b0(weights=None, num_classes=100)
        base = nn.SyncBatchNorm.convert_sync_batchnorm(base).to(device)
        model = DDP(base, device_ids=[local_rank])
        initial_hash = model_digest(base)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'])
        seed_all(cfg['seed'] + rank)
        start, best = 0, -1.0
        if args.resume:
            # Only local checkpoints generated here are accepted; pickle is not safe for untrusted files.
            state = torch.load(args.resume, map_location='cpu', weights_only=False)
            for key, expected in [('config_sha256', config_hash), ('data_sha256', data_hash), ('git_sha', git_sha)]:
                if state[key] != expected:
                    raise RuntimeError(f'Resume mismatch: {key}')
            if state['environment'] != env:
                raise RuntimeError('Resume environment mismatch.')
            base.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scaler.load_state_dict(state['scaler'])
            restore_rng(state['rng'][rank], device)
            start, best = state['epoch'] + 1, state['best_dev_top1']
            initial_hash = state['initial_sha256']
            if start >= cfg['epochs']:
                raise RuntimeError('Checkpoint already finished configured training.')
            if rank == 0:
                manifest['parent_checkpoint_sha256'] = digest(args.resume)
        sampler = DistributedSampler(train, num_replicas=4, rank=rank, shuffle=True,
                                     seed=cfg['seed'], drop_last=True)
        local_dev = list(range(rank, len(dev), 4))
        dev_loader = DataLoader(Subset(dev, local_dev), batch_size=cfg['batch_per_gpu'],
                                shuffle=False, num_workers=cfg['workers'], worker_init_fn=seed_worker,
                                generator=torch.Generator().manual_seed(cfg['seed'] + rank + 100000),
                                pin_memory=True)
        print(f'rank={rank} GPU={torch.cuda.get_device_name(local_rank)} initial_sha256={initial_hash}', flush=True)
        for epoch in range(start, cfg['epochs']):
            tick = time.monotonic()
            sampler.set_epoch(epoch)
            # Epoch-specific generators make an epoch-boundary resume reproducible.
            epoch_seed = cfg['seed'] + rank + 10000 * epoch
            loader = DataLoader(train, batch_size=cfg['batch_per_gpu'], sampler=sampler,
                                num_workers=cfg['workers'], drop_last=True, worker_init_fn=seed_worker,
                                generator=torch.Generator().manual_seed(epoch_seed), pin_memory=True)
            warm = cfg['warmup_epochs']
            factor = (epoch + 1) / warm if epoch < warm else 0.5 * (
                1 + math.cos(math.pi * (epoch - warm) / max(1, cfg['epochs'] - warm - 1)))
            for group in optimizer.param_groups:
                group['lr'] = cfg['lr'] * factor
            model.train()
            stats = torch.zeros(3, dtype=torch.float64, device=device)
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16, enabled=cfg['amp']):
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite loss; run stopped, configuration unchanged.')
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                stats += torch.stack([loss.detach().double() * y.numel(),
                                      (logits.argmax(1) == y).sum().double(),
                                      torch.tensor(y.numel(), device=device).double()])
            dist.all_reduce(stats)
            # Unwrapped eval avoids collectives with unequal numbers of validation batches.
            for buffer in base.buffers():
                dist.broadcast(buffer, src=0)
            base.eval()
            val = torch.zeros(3, dtype=torch.float64, device=device)
            predictions = []
            offset = 0
            with torch.inference_mode():
                for x, y in dev_loader:
                    x, y = x.to(device), y.to(device)
                    logits = base(x)  # FP32 evaluation, no random augmentations.
                    loss_sum = nn.functional.cross_entropy(logits, y, reduction='sum')
                    pred = logits.argmax(1)
                    val += torch.stack([loss_sum.double(), (pred == y).sum().double(),
                                        torch.tensor(y.numel(), device=device).double()])
                    for target, guess in zip(y.tolist(), pred.tolist()):
                        predictions.append([dev_ids[local_dev[offset]], target, guess])
                        offset += 1
            dist.all_reduce(val)
            if int(val[2]) != len(dev):
                raise RuntimeError('Validation counted duplicates or missed samples.')
            top1 = float(val[1] / val[2])
            improved = top1 > best
            best = max(best, top1)
            states = [None] * 4
            dist.all_gather_object(states, rng_state(device))
            all_preds = [None] * 4
            dist.all_gather_object(all_preds, predictions)
            metric = {'epoch': epoch + 1, 'lr': optimizer.param_groups[0]['lr'],
                      'train_nll': float(stats[0] / stats[2]), 'train_correct': int(stats[1]),
                      'train_count': int(stats[2]), 'dev_nll': float(val[0] / val[2]),
                      'dev_correct': int(val[1]), 'dev_count': int(val[2]), 'dev_top1': top1,
                      'elapsed_seconds': time.monotonic() - tick}
            if rank == 0:
                with (run / 'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(metric) + '\n')
                print(json.dumps(metric), flush=True)
                saved = {'epoch': epoch, 'model': base.state_dict(), 'optimizer': optimizer.state_dict(),
                         'scaler': scaler.state_dict(), 'rng': states, 'best_dev_top1': best,
                         'initial_sha256': initial_hash, 'config_sha256': config_hash,
                         'data_sha256': data_hash, 'git_sha': git_sha, 'environment': env}
                checkpoint(run / 'last.pt', saved)
                if improved:
                    checkpoint(run / 'best.pt', saved)
                    write_json(run / 'best_dev_predictions.json', sorted(sum(all_preds, [])))
            dist.barrier()
        final_hash = model_digest(base)
        gpu_results = [None] * 4
        dist.all_gather_object(gpu_results, {'rank': rank, 'model_sha256': final_hash,
                                          'peak_memory_bytes': torch.cuda.max_memory_allocated(device)})
        if len({entry['model_sha256'] for entry in gpu_results}) != 1:
            raise RuntimeError('Model replicas differ across ranks.')
        if rank == 0:
            write_json(run / 'metrics.json', {'best_dev_top1': best, 'final_model_sha256': final_hash,
                       'initial_model_sha256': initial_hash, 'gpu_results': gpu_results,
                       'official_test_used': False})
            manifest.update(status='completed', exit_code=0, finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(run / 'manifest.json', manifest)
        dist.barrier()
    except Exception:
        (run / f'error_rank{rank}.txt').write_text(traceback.format_exc())
        if rank == 0:
            manifest.update(status='failed', exit_code=1)
            write_json(run / 'manifest.json', manifest)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
