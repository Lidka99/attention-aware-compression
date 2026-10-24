"""E011: niezależny trening ResNet18 od zera na CIFAR-10 (jeden seed = jeden GPU).

Eksperyment: 2026-10-22_E011_cifar10_resnet18_checkpoint; pełny opis: OPIS.md.
Wejścia: frozen config.json, wyłącznie oficjalne CIFAR-10 train (50 000 obrazów),
splits.json przygotowany w katalogu biegu (45 000 train / 5 000 dev).
Wyjścia w <run>/seed_<seed>/: metrics.jsonl po każdej epoce, atomowo nadpisywany
last.pt (tylko state_dict + metadane, bez optymizatora) oraz finalny summary.json.
Archiwum zawiera test, ale test nie jest wypakowywany, ładowany ani oceniany.
"""
import argparse
import hashlib
import json
import os
import pickle
import random
import shutil
import tarfile
import tempfile
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import resnet18

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_ROOT = ROOT / 'data' / 'cifar10'
ARCHIVE_NAME = 'cifar-10-python.tar.gz'
ARCHIVE_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
ARCHIVE_MD5 = 'c58f30108f718f92721af3b95e74349a'
TRAIN_FILES = (
    ('data_batch_1', 'c99cafc152244af753f735de768cd75f'),
    ('data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'),
    ('data_batch_3', '54ebc095f3ab1f0389bbae665268c751'),
    ('data_batch_4', '634d18415352ddfa80567beed471001a'),
    ('data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'),
)
META_FILE = ('batches.meta', '5ff9c542aee3614f3951f8cda6e48888')
NORMALIZATION_MEAN = (0.4914, 0.4822, 0.4465)
NORMALIZATION_STD = (0.2023, 0.1994, 0.2010)
TRAIN_PER_CLASS = 4500
DEV_PER_CLASS = 500
SMOKE_TRAIN_PER_CLASS = 100
SMOKE_DEV_PER_CLASS = 20
EXPECTED_PARAMETERS = 11173962
EXPERIMENT = '2026-10-22_E011_cifar10_resnet18_checkpoint'
REQUIRED_CONFIG = ('epochs', 'seeds', 'batch_size', 'split_seed', 'lr',
                   'momentum', 'weight_decay', 'num_workers')


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def md5_file(path):
    hasher = hashlib.md5(usedforsecurity=False)
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def ensure_cifar10_train(root):
    root = Path(root)
    target = root / 'cifar-10-batches-py'
    expected = list(TRAIN_FILES) + [META_FILE]
    if target.is_dir() and all((target / name).is_file()
                               and md5_file(target / name) == checksum
                               for name, checksum in expected):
        return target
    root.mkdir(parents=True, exist_ok=True)
    archive = root / ARCHIVE_NAME
    if not archive.is_file() or md5_file(archive) != ARCHIVE_MD5:
        temporary = root / f'{ARCHIVE_NAME}.{os.getpid()}.part'
        try:
            print(f'Pobieranie CIFAR-10 train: {ARCHIVE_URL}', flush=True)
            urllib.request.urlretrieve(ARCHIVE_URL, temporary)
            require(md5_file(temporary) == ARCHIVE_MD5, 'Zły MD5 archiwum CIFAR-10')
            os.replace(temporary, archive)
        finally:
            if temporary.exists():
                temporary.unlink()
    wanted = {name for name, _ in expected}
    staging = Path(tempfile.mkdtemp(prefix='.extract-', dir=root))
    try:
        with tarfile.open(archive, 'r:gz') as source:
            for member in source.getmembers():
                name = Path(member.name).name
                if not member.isfile() or name not in wanted:
                    continue
                extracted = source.extractfile(member)
                if extracted is None:
                    continue
                with (staging / name).open('wb') as destination:
                    shutil.copyfileobj(extracted, destination)
        target.mkdir(parents=True, exist_ok=True)
        for name, checksum in expected:
            staged = staging / name
            require(staged.is_file() and md5_file(staged) == checksum,
                    f'Brak lub zły plik treningowy CIFAR-10: {name}')
            os.replace(staged, target / name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    require(all((target / name).is_file() and md5_file(target / name) == checksum
                for name, checksum in expected),
            'Niekompletne dane treningowe CIFAR-10')
    return target


def load_cifar10_train(root):
    target = ensure_cifar10_train(root)
    images = []
    labels = []
    for name, _ in TRAIN_FILES:
        with (target / name).open('rb') as stream:
            entry = pickle.load(stream, encoding='latin1')
        images.append(entry['data'])
        labels.extend(entry.get('labels', entry.get('fine_labels')))
    with (target / META_FILE[0]).open('rb') as stream:
        meta = pickle.load(stream, encoding='latin1')
    data = np.vstack(images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    data = np.ascontiguousarray(data, dtype=np.uint8)
    targets = np.asarray(labels, dtype=np.int64)
    require(len(data) == 50000 and len(targets) == 50000,
            'CIFAR-10 train powinien mieć 50 000 obrazów')
    require(int(targets.min()) == 0 and int(targets.max()) == 9,
            'Nieoczekiwane etykiety CIFAR-10')
    classes = [str(value) for value in meta['label_names']]
    return data, targets, classes


def make_split(labels, seed):
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=10)
    require(len(counts) == 10 and np.all(counts == 5000),
            'Podział wymaga 5 000 obrazów na klasę w CIFAR-10 train')
    rng = np.random.default_rng(int(seed))
    train = []
    dev = []
    for label in range(10):
        indices = np.flatnonzero(labels == label)
        order = rng.permutation(indices)
        dev.extend(int(index) for index in order[:DEV_PER_CLASS])
        train.extend(int(index) for index in order[DEV_PER_CLASS:])
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(sorted(dev), dtype=np.int64)


def validate_split(split, labels, seed):
    labels = np.asarray(labels)
    train = np.asarray(split['train'], dtype=np.int64)
    dev = np.asarray(split['dev'], dtype=np.int64)
    require(int(split['seed']) == int(seed), 'Niezgodny seed podziału')
    require(len(train) == TRAIN_PER_CLASS * 10 and len(dev) == DEV_PER_CLASS * 10,
            'Zła liczba próbek train/dev w splits.json')
    require(np.all(train >= 0) and np.all(train < len(labels))
            and np.all(dev >= 0) and np.all(dev < len(labels)),
            'Indeks poza zakresem CIFAR-10 train')
    mask_train = np.zeros(len(labels), dtype=bool)
    mask_dev = np.zeros(len(labels), dtype=bool)
    mask_train[train] = True
    mask_dev[dev] = True
    require(int(mask_train.sum()) == len(train), 'Duplikaty w train')
    require(int(mask_dev.sum()) == len(dev), 'Duplikaty w dev')
    require(not np.any(mask_train & mask_dev), 'Train i dev nie są rozłączne')
    require(np.all(mask_train | mask_dev), 'Podział nie pokrywa całego CIFAR-10 train')
    train_counts = np.bincount(labels[train], minlength=10)
    dev_counts = np.bincount(labels[dev], minlength=10)
    require(np.all(train_counts == TRAIN_PER_CLASS) and np.all(dev_counts == DEV_PER_CLASS),
            'Podział nie jest stratyfikowany po klasach')


def stratified_subset(indices, labels, per_class, seed):
    indices = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(labels)
    rng = np.random.default_rng(int(seed))
    chosen = []
    for label in range(10):
        pool = indices[labels[indices] == label]
        require(len(pool) >= per_class, f'Za mało próbek klasy {label} do podzbioru')
        chosen.extend(int(index) for index in rng.permutation(pool)[:per_class])
    return np.asarray(sorted(chosen), dtype=np.int64)


def prepare_split(run_dir, cfg, labels):
    path = Path(run_dir) / 'splits.json'
    seed = int(cfg['split_seed'])
    if path.exists():
        split = read_json(path)
    else:
        train, dev = make_split(labels, seed)
        split = {
            'dataset': 'CIFAR-10 official train',
            'official_test_used': False,
            'seed': seed,
            'strategy': 'stratified per class',
            'train_per_class': TRAIN_PER_CLASS,
            'dev_per_class': DEV_PER_CLASS,
            'train': train.tolist(),
            'dev': dev.tolist(),
        }
        write_json_atomic(path, split)
    validate_split(split, labels, seed)
    return split, digest(path)


def prepare_data(run_dir, cfg):
    images, labels, classes = load_cifar10_train(DATA_ROOT)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    split, split_sha256 = prepare_split(run_dir, cfg, labels)
    target = DATA_ROOT / 'cifar-10-batches-py'
    archive = DATA_ROOT / ARCHIVE_NAME
    manifest = {
        'dataset': 'CIFAR-10 official train only',
        'official_test_used': False,
        'official_test_extracted': False,
        'official_test_loaded': False,
        'data_root': str(DATA_ROOT),
        'classes': classes,
        'image_shape': [int(value) for value in images.shape[1:]],
        'train_count': int(len(split['train'])),
        'dev_count': int(len(split['dev'])),
        'split_seed': int(cfg['split_seed']),
        'split_sha256': split_sha256,
        'train_file_sha256': {name: digest(target / name) for name, _ in TRAIN_FILES},
        'meta_sha256': digest(target / META_FILE[0]),
        'archive_sha256': digest(archive) if archive.is_file() else None,
        'prepared_utc': utc_now(),
    }
    write_json_atomic(run_dir / 'data_manifest.json', manifest)
    return manifest


def build_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def train_transform():
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
    ])


def dev_transform():
    return T.Compose([
        T.CenterCrop(32),
        T.ToTensor(),
        T.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
    ])


class CifarTrainImages(Dataset):
    def __init__(self, images, labels, indices, transform):
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        image = Image.fromarray(self.images[index])
        return self.transform(image), int(self.labels[index])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(1)


def make_loader(dataset, batch_size, workers, shuffle, generator=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=True, drop_last=False,
                      worker_init_fn=seed_worker, generator=generator,
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
            require(logits.dtype == torch.float32 and bool(torch.isfinite(logits).all()),
                    'Niefinitywne logity dev')
            loss_sum += float(F.cross_entropy(logits, targets, reduction='sum'))
            correct += int((logits.argmax(1) == targets).sum())
            count += int(targets.numel())
    return loss_sum / count, correct / count, count


def save_checkpoint(path, model, metadata):
    path = Path(path)
    payload = {
        'state_dict': {name: tensor.detach().cpu()
                       for name, tensor in model.state_dict().items()},
        'metadata': metadata,
    }
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_config(path):
    path = Path(path)
    require(path.is_file(), f'Brak pliku konfiguracji: {path}')
    cfg = read_json(path)
    missing = [key for key in REQUIRED_CONFIG if key not in cfg]
    require(not missing, f'Brak pól konfiguracji: {missing}')
    if 'status' in cfg:
        require(cfg['status'] == 'frozen', 'Konfiguracja nie jest zamrożona')
    require(int(cfg['epochs']) > 0, 'epochs musi być dodatnie')
    require(isinstance(cfg['seeds'], list) and len(cfg['seeds']) == 4,
            'seeds musi zawierać cztery wartości')
    require(int(cfg['batch_size']) > 0, 'batch_size musi być dodatni')
    require(int(cfg['num_workers']) >= 0, 'num_workers musi być nieujemne')
    return cfg


def train(run_dir, args, cfg, config_path):
    seed = int(args.seed)
    epochs = 2 if args.smoke else int(args.epochs)
    seed_dir = Path(run_dir) / f'seed_{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = seed_dir / 'metrics.jsonl'
    require(not metrics_path.exists(), f'Istnieje już {metrics_path}')
    last_path = seed_dir / 'last.pt'
    summary_path = seed_dir / 'summary.json'
    config_sha256 = digest(config_path)
    started = time.monotonic()
    last_metrics = None
    state = {
        'experiment': EXPERIMENT,
        'seed': seed,
        'status': 'running',
        'smoke': bool(args.smoke),
        'epochs': epochs,
        'physical_gpu': int(args.gpu),
        'config_sha256': config_sha256,
        'official_test_used': False,
        'started_utc': utc_now(),
    }
    try:
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        require(visible is not None and visible.strip() == str(args.gpu),
                'CUDA_VISIBLE_DEVICES musi wskazywać fizyczny indeks z --gpu')
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                'Worker wymaga dokładnie jednego widocznego GPU')
        torch.cuda.set_device(0)
        device = torch.device('cuda', 0)
        seed_everything(seed)

        manifest = read_json(Path(run_dir) / 'manifest.json')
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
        require(manifest.get('split_sha256') == digest(split_path),
                'Hash splits.json nie zgadza się z manifestem')
        split = read_json(split_path)
        images, labels, _ = load_cifar10_train(DATA_ROOT)
        validate_split(split, labels, int(cfg['split_seed']))
        train_indices = np.asarray(split['train'], dtype=np.int64)
        dev_indices = np.asarray(split['dev'], dtype=np.int64)
        if args.smoke:
            train_indices = stratified_subset(train_indices, labels,
                                              SMOKE_TRAIN_PER_CLASS, int(cfg['split_seed']))
            dev_indices = stratified_subset(dev_indices, labels,
                                            SMOKE_DEV_PER_CLASS, int(cfg['split_seed']))
            require(len(train_indices) == 1000 and len(dev_indices) == 200,
                    'Zły podzbiór smoke')
            require(not np.intersect1d(train_indices, dev_indices).size,
                    'Podzbiór smoke nie jest rozłączny')
        data_manifest_path = Path(run_dir) / 'data_manifest.json'
        data_manifest_sha256 = digest(data_manifest_path) if data_manifest_path.is_file() else None

        train_dataset = CifarTrainImages(images, labels, train_indices, train_transform())
        dev_dataset = CifarTrainImages(images, labels, dev_indices, dev_transform())
        batch_size = int(cfg['batch_size'])
        workers = int(cfg['num_workers'])
        train_loader = make_loader(
            train_dataset, batch_size, workers, True,
            generator=torch.Generator().manual_seed(seed))
        dev_loader = make_loader(dev_dataset, batch_size, workers, False)

        model = build_model()
        parameters = count_parameters(model)
        require(parameters == EXPECTED_PARAMETERS,
                f'Nieoczekiwana liczba parametrów: {parameters}')
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=float(cfg['lr']),
                                    momentum=float(cfg['momentum']),
                                    weight_decay=float(cfg['weight_decay']),
                                    nesterov=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=0.0)
        state.update({
            'parameters': parameters,
            'batch_size_per_gpu': batch_size,
            'gradient_accumulation': 1,
            'train_count': int(len(train_indices)),
            'dev_count': int(len(dev_indices)),
            'split_sha256': digest(split_path),
            'data_manifest_sha256': data_manifest_sha256,
            'gpu_name': torch.cuda.get_device_name(0),
            'torch': torch.__version__,
            'torchvision': __import__('torchvision').__version__,
            'numpy': np.__version__,
            'cuda': torch.version.cuda,
            'normalization_mean': list(NORMALIZATION_MEAN),
            'normalization_std': list(NORMALIZATION_STD),
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
            'seed': seed,
            'epochs': epochs,
            'smoke': bool(args.smoke),
            'parameters': parameters,
            'architecture': 'resnet18_cifar10_conv1_3x3_maxpool_identity_fc10',
            'normalization_mean': list(NORMALIZATION_MEAN),
            'normalization_std': list(NORMALIZATION_STD),
            'config_sha256': config_sha256,
            'split_sha256': state['split_sha256'],
            'physical_gpu': int(args.gpu),
            'official_test_used': False,
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
                    require(bool(torch.isfinite(loss)), 'Niefinitywna strata treningowa')
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach()) * int(targets.numel())
                    correct += int((logits.argmax(1) == targets).sum())
                    count += int(targets.numel())
                dev_loss, dev_top1, dev_count = evaluate(model, dev_loader, device)
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
                    'timestamp_utc': utc_now(),
                }
                metrics.write(json.dumps(row, sort_keys=True) + '\n')
                metrics.flush()
                os.fsync(metrics.fileno())
                last_metrics = row
                metadata['epoch'] = epoch
                save_checkpoint(last_path, model, dict(metadata))
                scheduler.step()
                print(json.dumps(row, sort_keys=True), flush=True)

        state.update({
            'status': 'completed',
            'final_epoch': epochs,
            'selection': 'final_epoch_no_best',
            'best_checkpoint': None,
            'last_checkpoint': last_path.name,
            'last_checkpoint_sha256': digest(last_path),
            'final_metrics': last_metrics,
            'runtime_seconds': time.monotonic() - started,
            'peak_memory_mb': torch.cuda.max_memory_allocated() / (1024 ** 2),
            'finished_utc': utc_now(),
        })
        write_json_atomic(summary_path, state)
        print(f'E011_SEED={seed}_STATUS=completed', flush=True)
        return state
    except BaseException as error:
        state.update({
            'status': 'failed',
            'error': repr(error),
            'traceback': traceback.format_exc(),
            'final_metrics': last_metrics,
            'runtime_seconds': time.monotonic() - started,
            'finished_utc': utc_now(),
        })
        write_json_atomic(summary_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--gpu', type=int, help='fizyczny indeks GPU z CUDA_VISIBLE_DEVICES')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--smoke', action='store_true',
                        help='2 epoki, 1000 train / 200 dev (stratyfikowany podzbiór)')
    parser.add_argument('--data-only', action='store_true',
                        help='tylko pobranie CIFAR-10 train i zapis podziału')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.data_only:
        require(args.run_dir is not None, '--data-only wymaga --run-dir')
        manifest = prepare_data(args.run_dir, cfg)
        print(json.dumps({'run_dir': str(args.run_dir.resolve()),
                          'split_sha256': manifest['split_sha256'],
                          'train_count': manifest['train_count'],
                          'dev_count': manifest['dev_count'],
                          'official_test_used': False}, sort_keys=True))
        return
    require(args.run_dir is not None and args.seed is not None and args.gpu is not None,
            'Trening wymaga --run-dir, --seed i --gpu')
    require(int(args.seed) in [int(value) for value in cfg['seeds']],
            'Seed spoza listy seeds w config.json')
    train(args.run_dir.resolve(), args, cfg, args.config.resolve())


if __name__ == '__main__':
    main()
