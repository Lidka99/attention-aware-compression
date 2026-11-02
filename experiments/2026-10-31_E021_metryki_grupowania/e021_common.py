"""E021: wspólne helpery benchmarku metryk grupowania.

Eksperyment: 2026-10-31_E021_metryki_grupowania; pełny protokół: OPIS.md.
Wejście: checkpointy E011/E015/E019 i lokalne dane CIFAR/Tiny ImageNet.
Wyjście: helpery dla workera, launchera i analizy; bez treningu wag,
bez użycia dev do wyboru grupowania, bez oficjalnego testu i rezerwy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E017 = ROOT / 'experiments/2026-10-29_E017_miary_grupowania'
E019 = ROOT / 'experiments/2026-10-29_E019_tinyimagenet_alokacja'
E014 = ROOT / 'experiments/2026-10-27_E014_replikacja_alokacji'
for path in (str(E014), str(E017), str(E019), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import e017_common as base  # noqa: E402
import folding_budget  # noqa: E402

EXPERIMENT = '2026-10-31_E021_metryki_grupowania'
OPIS_PATH = 'experiments/2026-10-31_E021_metryki_grupowania/OPIS.md'
CONFIG_PATH = HERE / 'configs' / 'stage1.json'
REUSED_SOURCES = {
    'e012_folding.py': base.E012_FOLDING,
    'e012_evaluate.py': base.E012_EVALUATE,
    'e011_train.py': base.E011_TRAIN,
    'e015_data_cifar100.py': base.E015_DATA,
    'e017_common.py': E017 / 'e017_common.py',
    'e017_grouping.py': E017 / 'grouping.py',
    'e019_data_tiny.py': E019 / 'data_tiny.py',
    'e014_folding_budget.py': E014 / 'folding_budget.py',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return base.digest(path)


def read_json(path):
    return base.read_json(path)


def write_json_atomic(path, value):
    base.write_json_atomic(path, value)


def utc_now():
    return base.utc_now()


def load_config(path=CONFIG_PATH):
    cfg = read_json(path)
    require(cfg.get('status') == 'frozen', 'Konfiguracja nie jest zamrożona')
    require(cfg.get('experiment') == EXPERIMENT, 'Zły eksperyment')
    require([float(v) for v in cfg['budgets']] == [0.3, 0.5, 0.7],
            'Budżety muszą wynosić 30/50/70% parametrów')
    require([item['name'] for item in cfg['variants']] == [
        'reference_hkmeans', 'agg_l2', 'agg_l1', 'agg_cosine',
        'agg_correlation', 'ward_l2', 'standardized_hkmeans',
        'random_balanced'], 'Nieoczekiwane warianty')
    require(len(cfg['datasets']) == 3, 'Oczekiwano trzech datasetów')
    for entry in cfg['datasets']:
        require(Path(entry['checkpoint']).is_file(),
                f"Brak checkpointu {entry['tag']}")
        require(Path(entry['split']).is_file(),
                f"Brak splitu {entry['tag']}")
        require(digest(entry['checkpoint']) == entry['checkpoint_sha256'],
                f"Zły hash checkpointu {entry['tag']}")
        require(digest(entry['split']) == entry['split_sha256'],
                f"Zły hash splitu {entry['tag']}")
    require([int(g) for g in cfg['run']['gpus']] == [0, 1, 2, 3],
            'Wymagane GPU 0–3')
    return cfg


def build_tasks(cfg):
    tasks = []
    for entry in cfg['datasets']:
        for variant in cfg['variants']:
            for budget in cfg['budgets']:
                tasks.append({
                    'dataset': entry['tag'],
                    'family': entry['family'],
                    'seed': int(entry['seed']),
                    'variant': variant['name'],
                    'kind': variant['kind'],
                    'role': variant['role'],
                    'budget': float(budget),
                    'name': (f"{entry['tag']}__{variant['name']}"
                             f"__b{int(round(budget * 100)):02d}"),
                })
    require(len(tasks) == 72, f'Oczekiwano 72 zadań, jest {len(tasks)}')
    return tasks


def dataset_entry(cfg, tag):
    entry = next((item for item in cfg['datasets'] if item['tag'] == tag), None)
    require(entry is not None, f'Nieznany dataset: {tag}')
    return entry


def load_bundle(entry):
    family = entry['family']
    if family in ('cifar10', 'cifar100'):
        bundle = base.load_dataset({**entry, 'name': family})
        data = bundle['data']
        images = bundle['images']
        labels = bundle['labels']
        train_indices = bundle['train_indices']
        dev_indices = bundle['dev_indices']
        model = bundle['model']
        params = bundle['params']
    elif family == 'tiny':
        data = base.load_module(E019 / 'data_tiny.py', '_e021_data_tiny')
        images, labels, classes = data.load_tiny_train(data.DATA_ROOT)
        split = data.load_split(entry['split'])
        data.validate_split(split, labels, data.SPLIT_SEED)
        train_indices = __import__('numpy').asarray(
            split['splits']['train'], dtype=__import__('numpy').int64)
        dev_indices = __import__('numpy').asarray(
            split['splits']['dev'], dtype=__import__('numpy').int64)
        import torch
        payload = torch.load(entry['checkpoint'], map_location='cpu',
                             weights_only=True)
        model = data.build_model()
        model.load_state_dict(payload['state_dict'], strict=True)
        model.eval()
        params = int(sum(p.numel() for p in model.parameters()))
        require(len(classes) == int(entry['num_classes']), 'Zła liczba klas')
    else:
        raise ValueError(f'Nieznana rodzina: {family}')
    require(params == int(entry['expected_params']),
            f'Zła liczba parametrów: {params}')
    require(len(dev_indices) == int(entry['dev_count']),
            'Zła liczba próbek dev')
    return {
        'data': data, 'images': images, 'labels': labels,
        'train_indices': train_indices, 'dev_indices': dev_indices,
        'model': model, 'params': params,
    }


def e017_grouping():
    return base.load_module(E017 / 'grouping.py', '_e021_e017_grouping')


def folding_module():
    return base.folding_module()


def evaluate_module():
    return base.evaluate_module()
