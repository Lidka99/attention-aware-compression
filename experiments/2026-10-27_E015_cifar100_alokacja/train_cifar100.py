"""E015: jeden niezależny trening ResNet18 od zera na CIFAR-100 (seed = GPU).

Eksperyment: 2026-10-27_E015_cifar100_alokacja; pełny opis: OPIS.md.
Wejście: frozen ``configs/train.json``, lokalny CIFAR-100 train i istniejący
podział ``data/cifar100/splits.json`` (40 000 train / 5 000 dev / 5 000
rezerwy). Wyjście w ``<run>/seed_<seed>/``: ``metrics.jsonl`` po każdej epoce,
finalny ``last.pt`` (tylko state_dict i metadane, bez optymizatora) oraz
``summary.json``. Oficjalny test i rezerwa nie są ładowane ani oceniane.

Uruchamiane przez ``launch_training.py``. Bezpośredni trening wymaga
``--run-dir``, ``--seed`` i ``--gpu`` oraz ``CUDA_VISIBLE_DEVICES`` równego
``--gpu``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
if str(HERE) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(HERE))

import data_cifar100 as data  # noqa: E402

EXPERIMENT = data.EXPERIMENT
OPIS_PATH = data.OPIS_PATH
REQUIRED_CONFIG = (
    'epochs', 'seeds', 'batch_size', 'split_sha256', 'split_seed',
    'normalization_mean', 'normalization_std', 'lr', 'momentum',
    'weight_decay', 'num_workers',
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_config(path):
    path = Path(path)
    require(path.is_file(), f'Brak pliku konfiguracji: {path}')
    cfg = data.read_json(path)
    missing = [key for key in REQUIRED_CONFIG if key not in cfg]
    require(not missing, f'Brak pól konfiguracji: {missing}')
    if 'status' in cfg:
        require(cfg['status'] == 'frozen', 'Konfiguracja nie jest zamrożona')
    require(int(cfg['epochs']) > 0, 'epochs musi być dodatnie')
    seeds = [int(value) for value in cfg['seeds']]
    require(len(seeds) == 4 and sorted(seeds) == [42, 43, 44, 45],
            'seeds musi być permutacją [42, 43, 44, 45]')
    require(int(cfg['batch_size']) > 0, 'batch_size musi być dodatni')
    require(int(cfg['num_workers']) >= 0, 'num_workers musi być nieujemne')
    require(tuple(float(value) for value in cfg['normalization_mean'])
            == data.NORMALIZATION_MEAN, 'Zła średnia normalizacji')
    require(tuple(float(value) for value in cfg['normalization_std'])
            == data.NORMALIZATION_STD, 'Złe odchylenie normalizacji')
    require(str(cfg['split_sha256']).lower() == data.SPLITS_SHA256,
            'Config nie wskazuje właściwego splits.json')
    require(int(cfg['split_seed']) == data.SPLIT_SEED, 'Zły seed podziału')
    return cfg


def make_loader(dataset, batch_size, workers, shuffle, generator=None):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=True, drop_last=False,
        worker_init_fn=data.seed_worker, generator=generator,
        persistent_workers=workers > 0)


def evaluate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            require(logits.dtype == torch.float32
                    and bool(torch.isfinite(logits).all()),
                    'Niefinitywne logity dev')
            loss_sum += float(F.cross_entropy(logits, targets, reduction='sum'))
            correct += int((logits.argmax(1) == targets).sum())
            count += int(targets.numel())
    return loss_sum / count, correct / count, count


def save_checkpoint(path, model, metadata):
    path = Path(path)
    payload = {
        'state_dict': {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        'metadata': metadata,
    }
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(run_dir, args, cfg, config_path):
    seed = int(args.seed)
    epochs = 2 if args.smoke else int(args.epochs)
    seed_dir = Path(run_dir) / f'seed_{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = seed_dir / 'metrics.jsonl'
    require(not metrics_path.exists(), f'Istnieje już {metrics_path}')
    last_path = seed_dir / 'last.pt'
    summary_path = seed_dir / 'summary.json'
    config_sha256 = data.digest(config_path)
    started = time.monotonic()
    last_metrics = None
    state = {
        'experiment': EXPERIMENT,
        'opis': OPIS_PATH,
        'seed': seed,
        'status': 'running',
        'smoke': bool(args.smoke),
        'epochs': epochs,
        'physical_gpu': int(args.gpu),
        'config_sha256': config_sha256,
        'official_test_used': False,
        'reserve_used': False,
        'started_utc': data.utc_now(),
    }
    try:
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        require(visible is not None and visible.strip() == str(args.gpu),
                'CUDA_VISIBLE_DEVICES musi wskazywać fizyczny indeks z --gpu')
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                'Worker wymaga dokładnie jednego widocznego GPU')
        torch.cuda.set_device(0)
        device = torch.device('cuda', 0)
        data.seed_everything(seed)

        manifest = data.read_json(Path(run_dir) / 'manifest.json')
        require(manifest.get('status') in ('running', 'completed'),
                'Manifest biegu nie jest aktywny')
        require(manifest.get('config_sha256') == config_sha256,
                'Config w katalogu biegu różni się od config.json')
        require(seed in [int(value) for value in manifest.get('seeds', [])],
                'Seed nie występuje w manifeście biegu')
        require(manifest.get('gpu_by_seed', {}).get(str(seed)) == int(args.gpu),
                'Przypisanie seeda do GPU nie zgadza się z manifestem')
        require(bool(manifest.get('smoke')) == bool(args.smoke),
                'Flaga smoke nie zgadza się z manifestem')
        require(int(manifest.get('effective_epochs', epochs)) == epochs,
                'Liczba epok nie zgadza się z manifestem')
        split_path = Path(run_dir) / 'splits.json'
        require(split_path.is_file(), 'Brak splits.json w katalogu biegu')
        require(manifest.get('split_sha256') == data.digest(split_path),
                'Hash splits.json nie zgadza się z manifestem')
        split = data.load_split(split_path)
        images, labels, _ = data.load_cifar100_train(data.DATA_ROOT)
        data.validate_split(split, labels, int(cfg['split_seed']))
        train_indices = np.asarray(split['splits']['train'], dtype=np.int64)
        dev_indices = np.asarray(split['splits']['dev'], dtype=np.int64)
        if args.smoke:
            train_indices = data.stratified_subset(
                train_indices, labels, data.SMOKE_TRAIN_PER_CLASS,
                int(cfg['split_seed']))
            dev_indices = data.stratified_subset(
                dev_indices, labels, data.SMOKE_DEV_PER_CLASS,
                int(cfg['split_seed']))
            require(len(train_indices) == 1000 and len(dev_indices) == 200,
                    'Zły podzbiór smoke')
            require(not np.intersect1d(train_indices, dev_indices).size,
                    'Podzbiór smoke nie jest rozłączny')
        data_manifest_path = Path(run_dir) / 'data_manifest.json'
        data_manifest_sha256 = (
            data.digest(data_manifest_path)
            if data_manifest_path.is_file() else None)

        train_dataset = data.CifarTrainImages(
            images, labels, train_indices, data.train_transform())
        dev_dataset = data.CifarTrainImages(
            images, labels, dev_indices, data.dev_transform())
        batch_size = int(cfg['batch_size'])
        workers = int(cfg['num_workers'])
        train_loader = make_loader(
            train_dataset, batch_size, workers, True,
            generator=torch.Generator().manual_seed(seed))
        dev_loader = make_loader(
            dev_dataset, batch_size, workers, False)

        model = data.build_model()
        parameters = data.count_parameters(model)
        require(parameters == data.EXPECTED_PARAMETERS,
                f'Nieoczekiwana liczba parametrów: {parameters}')
        model = model.to(device)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=float(cfg['lr']),
            momentum=float(cfg['momentum']),
            weight_decay=float(cfg['weight_decay']), nesterov=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=0.0)
        state.update({
            'parameters': parameters,
            'batch_size_per_gpu': batch_size,
            'gradient_accumulation': 1,
            'effective_global_batch_per_run': batch_size,
            'independent_runs': 4,
            'aggregation': 'none (independent seeds)',
            'train_count': int(len(train_indices)),
            'dev_count': int(len(dev_indices)),
            'split_sha256': data.digest(split_path),
            'data_manifest_sha256': data_manifest_sha256,
            'gpu_name': torch.cuda.get_device_name(0),
            'torch': torch.__version__,
            'torchvision': __import__('torchvision').__version__,
            'numpy': np.__version__,
            'cuda': torch.version.cuda,
            'normalization_mean': list(data.NORMALIZATION_MEAN),
            'normalization_std': list(data.NORMALIZATION_STD),
            'optimizer': 'SGD',
            'lr': float(cfg['lr']),
            'momentum': float(cfg['momentum']),
            'weight_decay': float(cfg['weight_decay']),
            'nesterov': False,
            'schedule': 'cosine',
            'eta_min': 0.0,
            'amp': False,
            'label_smoothing': 0.0,
        })
        metadata = {
            'experiment': EXPERIMENT,
            'opis': OPIS_PATH,
            'seed': seed,
            'epochs': epochs,
            'smoke': bool(args.smoke),
            'parameters': parameters,
            'architecture': (
                'resnet18_cifar100_conv1_3x3_maxpool_identity_fc100'),
            'dataset': 'CIFAR-100',
            'normalization_mean': list(data.NORMALIZATION_MEAN),
            'normalization_std': list(data.NORMALIZATION_STD),
            'config_sha256': config_sha256,
            'split_sha256': state['split_sha256'],
            'physical_gpu': int(args.gpu),
            'batch_size_per_gpu': batch_size,
            'gradient_accumulation': 1,
            'official_test_used': False,
            'reserve_used': False,
        }

        with metrics_path.open('w') as metrics:
            for epoch in range(1, epochs + 1):
                tick = time.monotonic()
                model.train()
                loss_sum = 0.0
                correct = 0
                count = 0
                for images_batch, targets in train_loader:
                    images_batch = images_batch.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(images_batch)
                    loss = F.cross_entropy(logits, targets)
                    require(bool(torch.isfinite(loss)),
                            'Niefinitywna strata treningowa')
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach()) * int(targets.numel())
                    correct += int((logits.argmax(1) == targets).sum())
                    count += int(targets.numel())
                dev_loss, dev_top1, dev_count = evaluate(
                    model, dev_loader, device)
                torch.cuda.synchronize()
                row = {
                    'experiment': EXPERIMENT,
                    'seed': seed,
                    'epoch': epoch,
                    'epochs': epochs,
                    'lr': float(optimizer.param_groups[0]['lr']),
                    'train_loss': loss_sum / count,
                    'train_top1': correct / count,
                    'train_count': count,
                    'dev_loss': dev_loss,
                    'dev_top1': dev_top1,
                    'dev_count': dev_count,
                    'elapsed_seconds': time.monotonic() - tick,
                    'smoke': bool(args.smoke),
                    'config_sha256': config_sha256,
                    'split_sha256': state['split_sha256'],
                    'batch_size_per_gpu': batch_size,
                    'gradient_accumulation': 1,
                    'timestamp_utc': data.utc_now(),
                }
                metrics.write(json.dumps(row, sort_keys=True) + '\n')
                metrics.flush()
                os.fsync(metrics.fileno())
                last_metrics = row
                scheduler.step()
                print(json.dumps(row, sort_keys=True), flush=True)
            metadata['epoch'] = epochs
            save_checkpoint(last_path, model, dict(metadata))

        state.update({
            'status': 'completed',
            'final_epoch': epochs,
            'selection': 'final_epoch_no_best',
            'checkpoint_retention': 'final last.pt only',
            'best_checkpoint': None,
            'last_checkpoint': last_path.name,
            'last_checkpoint_sha256': data.digest(last_path),
            'final_metrics': last_metrics,
            'runtime_seconds': time.monotonic() - started,
            'peak_memory_mb': torch.cuda.max_memory_allocated() / (1024 ** 2),
            'finished_utc': data.utc_now(),
        })
        data.write_json_atomic(summary_path, state)
        print(f'E015_SEED={seed}_STATUS=completed', flush=True)
        return state
    except BaseException as error:
        state.update({
            'status': 'failed',
            'error': repr(error),
            'traceback': traceback.format_exc(),
            'final_metrics': last_metrics,
            'runtime_seconds': time.monotonic() - started,
            'finished_utc': data.utc_now(),
        })
        data.write_json_atomic(summary_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--config', type=Path,
                        default=HERE / 'configs' / 'train.json')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--gpu', type=int,
                        help='fizyczny indeks GPU z CUDA_VISIBLE_DEVICES')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--smoke', action='store_true',
                        help='2 epoki, 1000 train / 200 dev')
    parser.add_argument('--data-only', action='store_true',
                        help='tylko walidacja CIFAR-100 i przygotowanie podziału')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.data_only:
        require(args.run_dir is not None, '--data-only wymaga --run-dir')
        manifest = data.prepare_run_data(args.run_dir, cfg)
        print(json.dumps({
            'run_dir': str(args.run_dir.resolve()),
            'split_sha256': manifest['split_sha256'],
            'train_count': manifest['train_count'],
            'dev_count': manifest['dev_count'],
            'reserve_count': manifest['reserve_count'],
            'official_test_used': False,
            'reserve_used': False,
        }, sort_keys=True))
        return
    require(args.run_dir is not None and args.seed is not None
            and args.gpu is not None,
            'Trening wymaga --run-dir, --seed i --gpu')
    require(int(args.seed) in [int(value) for value in cfg['seeds']],
            'Seed spoza listy seeds w config.json')
    train(args.run_dir.resolve(), args, cfg, args.config.resolve())


if __name__ == '__main__':
    main()
