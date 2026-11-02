"""E019: dane Tiny ImageNet, podział 90k/10k/10k i ResNet18 (200 klas).

Eksperyment: 2026-10-29_E019_tinyimagenet_alokacja; pełny opis: OPIS.md.
Wejście: lokalne ``data/tiny-imagenet/cache_train.npz`` i ``splits.json``
(pochodzenie E005); 100 000 obrazów 64x64, 200 klas. Indeksy train/dev
wskazują cache_train (90 000/10 000). Rezerwa 10 000 to osobny
``cache_val.npz`` (oficjalny val); nie jest ładowana ani oceniana.

Model: torchvision ResNet18 (standardowy stem 7x7/stride2 + maxpool),
200 klas, trenowany od zera. API zgodne z helperami E012/E014.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.models import resnet18

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = '2026-10-29_E019_tinyimagenet_alokacja'
OPIS_PATH = 'experiments/2026-10-29_E019_tinyimagenet_alokacja/OPIS.md'
DATA_ROOT = ROOT / 'data' / 'tiny-imagenet'
CACHE_TRAIN = DATA_ROOT / 'cache_train.npz'
SPLITS_FILE = DATA_ROOT / 'splits.json'

CACHE_SHA256 = '90642db970cc6f83745e569e19f0659a726acfe62612908f1ee63331df23fa8a'
SPLITS_SHA256 = '9df906f7ba0b217f779387c9f23c063a30c7a99a0e4376d0fd73119c100efcbd'

NUM_CLASSES = 200
EXPECTED_TRAIN = 90000
EXPECTED_DEV = 10000
EXPECTED_RESERVE = 10000
TRAIN_PER_CLASS = 450
DEV_PER_CLASS = 50
RESERVE_PER_CLASS = 50
SMOKE_TRAIN_PER_CLASS = 10
SMOKE_DEV_PER_CLASS = 2
EXPECTED_PARAMETERS = 11279112
SPLIT_SEED = 20261010
NORMALIZATION_MEAN = (0.48030321444761837, 0.4482903024765458,
                      0.3977083882277699)
NORMALIZATION_STD = (0.2763871312320068, 0.2688358266538272,
                     0.28161664632282396)
IMAGE_SIZE = 64


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
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


def verify_source_files():
    require(CACHE_TRAIN.is_file(), f'Brak cache Tiny ImageNet: {CACHE_TRAIN}')
    require(SPLITS_FILE.is_file(), f'Brak splits.json: {SPLITS_FILE}')
    require(digest(CACHE_TRAIN) == CACHE_SHA256,
            f'Hash cache Tiny ImageNet nie zgadza się: {CACHE_TRAIN}')
    require(digest(SPLITS_FILE) == SPLITS_SHA256,
            f'Hash splits.json Tiny ImageNet nie zgadza się: {SPLITS_FILE}')


def load_tiny_train(root=DATA_ROOT):
    root = Path(root)
    require(root.resolve() == DATA_ROOT.resolve(),
            f'Dane muszą pochodzić z {DATA_ROOT}, otrzymano {root}')
    verify_source_files()
    with np.load(CACHE_TRAIN, allow_pickle=False) as cache:
        images = np.asarray(cache['images'], dtype=np.uint8)
        labels = np.asarray(cache['labels'], dtype=np.int64)
    require(images.shape == (100000, IMAGE_SIZE, IMAGE_SIZE, 3),
            f'Nieoczekiwany kształt obrazów: {images.shape}')
    require(labels.shape == (100000,), 'Nieoczekiwana liczba etykiet')
    require(int(labels.min()) == 0 and int(labels.max()) == NUM_CLASSES - 1,
            'Etykiety poza zakresem 0..199')
    split = read_json(SPLITS_FILE)
    classes = [str(value) for value in split['classes']]
    require(len(classes) == NUM_CLASSES and len(set(classes)) == NUM_CLASSES,
            'Nieoczekiwana lista klas Tiny ImageNet')
    return images, labels, classes


def load_split(path=SPLITS_FILE):
    path = Path(path)
    require(path.is_file(), f'Brak pliku podziału: {path}')
    split = read_json(path)
    require(split.get('dataset') == 'TinyImageNet',
            'Podział nie dotyczy Tiny ImageNet')
    require(split.get('official_test_used') is False,
            'Podział oznacza użycie oficjalnego testu')
    require(int(split.get('split_seed')) == SPLIT_SEED,
            f'Seed podziału musi wynosić {SPLIT_SEED}')
    require(isinstance(split.get('splits'), dict)
            and set(split['splits']) == {'train', 'dev', 'reserve'},
            'Podział musi zawierać train, dev i reserve')
    return split


def validate_split(split, labels, split_seed=SPLIT_SEED):
    require(int(split.get('split_seed', split.get('seed'))) == int(split_seed),
            'Niezgodny seed podziału')
    labels = np.asarray(labels)
    train = np.asarray(split['splits']['train'], dtype=np.int64)
    dev = np.asarray(split['splits']['dev'], dtype=np.int64)
    reserve = np.asarray(split['splits']['reserve'], dtype=np.int64)
    require(len(train) == EXPECTED_TRAIN
            and len(dev) == EXPECTED_DEV
            and len(reserve) == EXPECTED_RESERVE,
            'Zła liczba próbek train/dev/reserve')
    for name, indices in (('train', train), ('dev', dev)):
        require(np.all(indices >= 0) and np.all(indices < len(labels)),
                f'Indeks poza zakresem w {name}')
        require(len(set(indices.tolist())) == len(indices),
                f'Duplikaty w {name}')
    require(len(set(reserve.tolist())) == EXPECTED_RESERVE,
            'Rezerwa zawiera duplikaty')
    require(not np.intersect1d(train, dev).size,
            'Podziały train/dev nie są rozłączne')
    union = np.union1d(train, dev)
    require(len(union) == 100000,
            'Podziały nie pokrywają 100 000 obrazów cache_train')
    expected = {'train': TRAIN_PER_CLASS, 'dev': DEV_PER_CLASS}
    for name, indices in (('train', train), ('dev', dev)):
        counts = np.bincount(labels[indices], minlength=NUM_CLASSES)
        require(np.all(counts == expected[name]),
                f'{name} nie jest stratyfikowany po 200 klasach')


def stratified_subset(indices, labels, per_class, seed):
    indices = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(labels)
    rng = np.random.default_rng(int(seed))
    chosen = []
    for label in range(NUM_CLASSES):
        pool = indices[labels[indices] == label]
        require(len(pool) >= per_class,
                f'Za mało próbek klasy {label} do podzbioru')
        chosen.extend(int(index) for index in rng.permutation(pool)[:per_class])
    return np.asarray(sorted(chosen), dtype=np.int64)


def build_model(num_classes=NUM_CLASSES):
    require(int(num_classes) == NUM_CLASSES,
            f'E019 wymaga {NUM_CLASSES} klas, otrzymano {num_classes}')
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(512, NUM_CLASSES)
    return model


def count_parameters(model):
    return int(sum(parameter.numel() for parameter in model.parameters()))


def train_transform():
    return T.Compose([
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.4, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
    ])


def dev_transform():
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE),
                 interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.ToTensor(),
        T.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
    ])


class CifarTrainImages(torch.utils.data.Dataset):
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


def prepare_run_data(run_dir, cfg):
    images, labels, classes = load_tiny_train(DATA_ROOT)
    split = load_split(SPLITS_FILE)
    validate_split(split, labels, int(cfg['split_seed']))
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    split_target = run_dir / 'splits.json'
    shutil.copy2(SPLITS_FILE, split_target)
    require(digest(split_target) == SPLITS_SHA256,
            'Kopia splits.json w katalogu biegu ma zły hash')
    require(digest(split_target) == str(cfg['split_sha256']).lower(),
            'Config nie wskazuje właściwego splits.json')
    manifest = {
        'experiment': EXPERIMENT,
        'opis': OPIS_PATH,
        'dataset': 'Tiny ImageNet (oficjalny train)',
        'num_classes': NUM_CLASSES,
        'official_test_used': False,
        'official_test_loaded': False,
        'reserve_used': False,
        'reserve_role': 'oficjalny val w cache_val.npz; nietknięty',
        'data_root': str(DATA_ROOT),
        'classes': classes,
        'image_shape': [int(value) for value in images.shape[1:]],
        'train_count': int(EXPECTED_TRAIN),
        'dev_count': int(EXPECTED_DEV),
        'reserve_count': int(EXPECTED_RESERVE),
        'split_seed': int(cfg['split_seed']),
        'split_sha256': SPLITS_SHA256,
        'cache_train_sha256': CACHE_SHA256,
        'normalization_mean': list(NORMALIZATION_MEAN),
        'normalization_std': list(NORMALIZATION_STD),
        'prepared_utc': utc_now(),
    }
    write_json_atomic(run_dir / 'data_manifest.json', manifest)
    return manifest
