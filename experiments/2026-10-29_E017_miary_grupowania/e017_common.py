"""E017: wspólne helpery ładowania danych, modeli i konfiguracji.

Eksperyment: 2026-10-29_E017_miary_grupowania; pełny protokół: OPIS.md.
Wejście: zamrożony ``configs/stage1.json`` oraz checkpointy E011/E015.
Wyjście: funkcje pomocnicze dla workera, launchera i analizy; moduł nie
trenuje i nie używa dev do wyboru grupowania.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = '2026-10-29_E017_miary_grupowania'
OPIS_PATH = 'experiments/2026-10-29_E017_miary_grupowania/OPIS.md'
CONFIG_PATH = HERE / 'configs' / 'stage1.json'
E012_FOLDING = ROOT / (
    'experiments/2026-10-24_E012_folding_referencja/folding.py')
E012_EVALUATE = ROOT / (
    'experiments/2026-10-24_E012_folding_referencja/evaluate.py')
E011_TRAIN = ROOT / (
    'experiments/2026-10-22_E011_cifar10_resnet18_checkpoint/train.py')
E015_DATA = ROOT / (
    'experiments/2026-10-27_E015_cifar100_alokacja/data_cifar100.py')
REUSED_SOURCES = {
    'e012_folding.py': E012_FOLDING,
    'e012_evaluate.py': E012_EVALUATE,
    'e011_train.py': E011_TRAIN,
    'e015_data_cifar100.py': E015_DATA,
}


def require(condition, message):
    """Podnosi RuntimeError, jeśli warunek nie jest spełniony."""
    if not condition:
        raise RuntimeError(message)


def utc_now():
    """Zwraca bieżący znacznik czasu UTC."""
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    """Liczy SHA-256 pliku."""
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path):
    """Wczytuje JSON."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json_atomic(path, value):
    """Zapisuje JSON atomowo."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_module(path, name):
    """Ładuje moduł z pliku bez pakietu."""
    path = Path(path)
    require(path.is_file(), f'Brak modułu: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None,
            f'Nie można wczytać modułu {name}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def folding_module():
    """Zwraca moduł E012 folding (bez zmian)."""
    return load_module(E012_FOLDING, '_e017_e012_folding')


def evaluate_module():
    """Zwraca moduł E012 evaluate (bez zmian)."""
    return load_module(E012_EVALUATE, '_e017_e012_evaluate')


def load_config(path=CONFIG_PATH):
    """Wczytuje i waliduje zamrożoną konfigurację etapu 1."""
    cfg = read_json(path)
    require(cfg.get('status') == 'frozen', 'Konfiguracja nie jest zamrożona')
    require(cfg.get('experiment') == EXPERIMENT, 'Zły eksperyment w configu')
    require([float(v) for v in cfg['budgets']] == [0.1, 0.3, 0.5],
            'Budżety muszą wynosić 10/30/50% kanałów')
    names = [item['name'] for item in cfg['measures']]
    require(names == ['reference', 'cosine', 'standardized', 'bn_effective',
                      'random1', 'random2', 'random3'],
            f'Nieoczekiwane warianty: {names}')
    require(len(cfg['datasets']) == 2, 'Oczekiwano dwóch datasetów')
    for dataset in cfg['datasets']:
        for key in ('checkpoint', 'split'):
            path_value = Path(str(dataset[key])).expanduser()
            require(path_value.is_absolute(), f'{key} musi być absolutny')
            require(path_value.is_file(), f'Brak pliku {key}: {path_value}')
        for key in ('checkpoint_sha256', 'split_sha256'):
            value = str(dataset[key]).lower()
            require(len(value) == 64, f'{key} musi być SHA-256')
        require(digest(dataset['checkpoint']) == dataset['checkpoint_sha256'],
                f"Zły hash checkpointu {dataset['name']}")
        require(digest(dataset['split']) == dataset['split_sha256'],
                f"Zły hash splitu {dataset['name']}")
    run = cfg['run']
    require([int(g) for g in run['gpus']] == [0, 1, 2, 3],
            'Wymagane fizyczne GPU 0–3')
    require(int(run['dev_count']) == 5000, 'Dev musi mieć 5000 próbek')
    return cfg


def build_tasks(cfg):
    """Zwraca 42 zadania: dataset × miara × budżet."""
    tasks = []
    for dataset in cfg['datasets']:
        for measure in cfg['measures']:
            for budget in cfg['budgets']:
                tasks.append({
                    'dataset': dataset['name'],
                    'measure': measure['name'],
                    'kind': measure['kind'],
                    'role': measure['role'],
                    'replicate': int(measure.get('replicate', 0)),
                    'budget': float(budget),
                    'name': f"{dataset['name']}__{measure['name']}__b{int(round(budget * 100)):02d}",
                })
    require(len(tasks) == 42, f'Oczekiwano 42 zadań, jest {len(tasks)}')
    return tasks


def load_dataset(entry):
    """Ładuje dane, split, checkpoint i model dla jednego datasetu."""
    name = entry['name']
    if name == 'cifar10':
        data = load_module(E011_TRAIN, '_e017_e011_train')
        images, labels, classes = data.load_cifar10_train(data.DATA_ROOT)
        split = read_json(entry['split'])
        data.validate_split(split, labels, int(split.get('seed')))
        train_indices = np.asarray(split['train'], dtype=np.int64)
        dev_indices = np.asarray(split['dev'], dtype=np.int64)
    elif name == 'cifar100':
        data = load_module(E015_DATA, '_e017_e015_data')
        images, labels, classes = data.load_cifar100_train(data.DATA_ROOT)
        split = data.load_split(entry['split'])
        data.validate_split(split, labels, data.SPLIT_SEED)
        train_indices = np.asarray(split['splits']['train'], dtype=np.int64)
        dev_indices = np.asarray(split['splits']['dev'], dtype=np.int64)
    else:
        raise ValueError(f'Nieznany dataset: {name}')
    require(len(set(train_indices.tolist()) & set(dev_indices.tolist())) == 0,
            'Train i dev mają wspólne indeksy')
    require(len(dev_indices) == 5000, 'Dev musi mieć 5000 próbek')
    require(len(classes) == int(entry['num_classes']), 'Zła liczba klas')
    payload = torch.load(entry['checkpoint'], map_location='cpu',
                         weights_only=True)
    require(isinstance(payload, dict)
            and isinstance(payload.get('state_dict'), dict),
            'Checkpoint bez state_dict')
    model = data.build_model()
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    params = int(sum(p.numel() for p in model.parameters()))
    require(params == int(entry['expected_params']),
            f'Zła liczba parametrów: {params}')
    return {
        'name': name,
        'data': data,
        'images': images,
        'labels': labels,
        'train_indices': train_indices,
        'dev_indices': dev_indices,
        'model': model,
        'params': params,
    }


def labels_digest(labels):
    """Stały skrót etykiet klastrów (int64 little-endian)."""
    array = torch.as_tensor(labels, dtype=torch.int64).cpu().numpy()
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()
