"""E015: dane CIFAR-100, podział train/dev/rezerwa i model ResNet18 (100 klas).

Eksperyment: 2026-10-27_E015_cifar100_alokacja; pełny opis: OPIS.md.
Wejście: wyłącznie lokalne pliki ``data/cifar100/cifar-100-python/train``,
``meta`` oraz ``data/cifar100/splits.json``. Wyjście: funkcje ładowania,
walidacji podziału, transformacje, dataset, model i przygotowanie manifestu
biegu. Oficjalny test CIFAR-100 nie jest wypakowywany, ładowany ani oceniany;
rezerwa 5 000 obrazów z train także pozostaje nietknięta.

Moduł jest używany przez trening E015, generowanie configów foldingu i worker
foldingu. Nie modyfikuje istniejących eksperymentów E011–E014.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms as T
from torchvision.models import resnet18

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = '2026-10-27_E015_cifar100_alokacja'
OPIS_PATH = 'experiments/2026-10-27_E015_cifar100_alokacja/OPIS.md'
DATA_ROOT = ROOT / 'data' / 'cifar100'
TRAIN_FILE = DATA_ROOT / 'cifar-100-python' / 'train'
META_FILE = DATA_ROOT / 'cifar-100-python' / 'meta'
SPLITS_FILE = DATA_ROOT / 'splits.json'
ARCHIVE_FILE = DATA_ROOT / 'cifar-100-python.tar.gz'

TRAIN_SHA256 = '735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b'
META_SHA256 = 'a5d4786345c961390f865e93b434dbd5c6904ce880667e0cb888c97d449f28b9'
SPLITS_SHA256 = 'ecb8804e41fbc4bee7c61fd0585bf1c7dcfc94ac8472333fc283d51dc6460e86'
ARCHIVE_SHA256 = '85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7'

NUM_CLASSES = 100
EXPECTED_TRAIN = 40000
EXPECTED_DEV = 5000
EXPECTED_RESERVE = 5000
TRAIN_PER_CLASS = 400
DEV_PER_CLASS = 50
RESERVE_PER_CLASS = 50
SMOKE_TRAIN_PER_CLASS = 10
SMOKE_DEV_PER_CLASS = 2
EXPECTED_PARAMETERS = 11220132
SPLIT_SEED = 20260926
NORMALIZATION_MEAN = (0.507421, 0.486865, 0.441380)
NORMALIZATION_STD = (0.267468, 0.256482, 0.275979)


def digest(path):
    hasher = hashlib.sha256()
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


def verify_source_files():
    require(TRAIN_FILE.is_file(), f'Brak pliku train CIFAR-100: {TRAIN_FILE}')
    require(META_FILE.is_file(), f'Brak pliku meta CIFAR-100: {META_FILE}')
    require(SPLITS_FILE.is_file(), f'Brak splits.json CIFAR-100: {SPLITS_FILE}')
    require(digest(TRAIN_FILE) == TRAIN_SHA256,
            f'Hash train CIFAR-100 nie zgadza się: {TRAIN_FILE}')
    require(digest(META_FILE) == META_SHA256,
            f'Hash meta CIFAR-100 nie zgadza się: {META_FILE}')
    require(digest(SPLITS_FILE) == SPLITS_SHA256,
            f'Hash splits.json nie zgadza się: {SPLITS_FILE}')
    if ARCHIVE_FILE.is_file():
        require(digest(ARCHIVE_FILE) == ARCHIVE_SHA256,
                f'Hash archiwum CIFAR-100 nie zgadza się: {ARCHIVE_FILE}')


def load_cifar100_train(root=DATA_ROOT):
    root = Path(root)
    require(root.resolve() == DATA_ROOT.resolve(),
            f'Dane muszą pochodzić z {DATA_ROOT}, otrzymano {root}')
    verify_source_files()
    with TRAIN_FILE.open('rb') as stream:
        entry = pickle.load(stream, encoding='latin1')
    require(isinstance(entry, dict) and 'data' in entry and 'fine_labels' in entry,
            'Plik train CIFAR-100 ma nieoczekiwaną strukturę')
    data = np.asarray(entry['data'], dtype=np.uint8)
    require(data.shape == (50000, 3072),
            f'Nieoczekiwany kształt train CIFAR-100: {data.shape}')
    images = np.ascontiguousarray(
        data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
    labels = np.asarray(entry['fine_labels'], dtype=np.int64)
    require(labels.shape == (50000,), 'Nieoczekiwana liczba etykiet CIFAR-100')
    require(int(labels.min()) == 0 and int(labels.max()) == NUM_CLASSES - 1,
            'Etykiety CIFAR-100 poza zakresem 0..99')
    with META_FILE.open('rb') as stream:
        meta = pickle.load(stream, encoding='latin1')
    classes = [str(value) for value in meta['fine_label_names']]
    require(len(classes) == NUM_CLASSES and len(set(classes)) == NUM_CLASSES,
            'Nieoczekiwana lista klas CIFAR-100')
    return images, labels, classes


def load_split(path=SPLITS_FILE):
    path = Path(path)
    require(path.is_file(), f'Brak pliku podziału: {path}')
    split = read_json(path)
    require(split.get('dataset') == 'CIFAR100', 'Podział nie dotyczy CIFAR100')
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
    for name, indices in (('train', train), ('dev', dev), ('reserve', reserve)):
        require(np.all(indices >= 0) and np.all(indices < len(labels)),
                f'Indeks poza zakresem w {name}')
        require(len(set(indices.tolist())) == len(indices),
                f'Duplikaty w {name}')
    require(not np.intersect1d(train, dev).size
            and not np.intersect1d(train, reserve).size
            and not np.intersect1d(dev, reserve).size,
            'Podziały train/dev/reserve nie są rozłączne')
    union = np.union1d(np.union1d(train, dev), reserve)
    require(len(union) == 50000, 'Podziały nie pokrywają 50 000 obrazów train')
    expected = {
        'train': TRAIN_PER_CLASS,
        'dev': DEV_PER_CLASS,
        'reserve': RESERVE_PER_CLASS,
    }
    for name, indices in (('train', train), ('dev', dev), ('reserve', reserve)):
        counts = np.bincount(labels[indices], minlength=NUM_CLASSES)
        require(np.all(counts == expected[name]),
                f'{name} nie jest stratyfikowany po 100 klasach')


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
            f'E015 wymaga {NUM_CLASSES} klas, otrzymano {num_classes}')
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, NUM_CLASSES)
    return model


def count_parameters(model):
    return int(sum(parameter.numel() for parameter in model.parameters()))


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
    torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(1)


def prepare_run_data(run_dir, cfg):
    images, labels, classes = load_cifar100_train(DATA_ROOT)
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
        'dataset': 'CIFAR-100 (oficjalny train)',
        'num_classes': NUM_CLASSES,
        'official_test_used': False,
        'official_test_loaded': False,
        'reserve_used': False,
        'reserve_role': 'held out from official train; untouched',
        'data_root': str(DATA_ROOT),
        'classes': classes,
        'image_shape': [int(value) for value in images.shape[1:]],
        'train_count': int(EXPECTED_TRAIN),
        'dev_count': int(EXPECTED_DEV),
        'reserve_count': int(EXPECTED_RESERVE),
        'split_seed': int(cfg['split_seed']),
        'split_sha256': SPLITS_SHA256,
        'train_file_sha256': TRAIN_SHA256,
        'meta_sha256': META_SHA256,
        'archive_sha256': ARCHIVE_SHA256 if ARCHIVE_FILE.is_file() else None,
        'normalization_mean': list(NORMALIZATION_MEAN),
        'normalization_std': list(NORMALIZATION_STD),
        'prepared_utc': utc_now(),
    }
    write_json_atomic(run_dir / 'data_manifest.json', manifest)
    return manifest
