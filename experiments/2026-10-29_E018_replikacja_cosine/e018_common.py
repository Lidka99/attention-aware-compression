"""E018: konfiguracja i zadania replikacji miary cosine.

Eksperyment: 2026-10-29_E018_replikacja_cosine; pełny protokół: OPIS.md.
Wejście: zamrożony ``configs/stage1.json`` i checkpointy E011/E015 seed43–45.
Wyjście: helpery dla workera, launchera i analizy; bez treningu i bez dev
w wyborze grupowania. Kod grupowania pochodzi z E017 (bez zmian).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E017 = ROOT / 'experiments/2026-10-29_E017_miary_grupowania'
if str(E017) not in sys.path:
    sys.path.insert(0, str(E017))

import e017_common as base  # noqa: E402

EXPERIMENT = '2026-10-29_E018_replikacja_cosine'
OPIS_PATH = 'experiments/2026-10-29_E018_replikacja_cosine/OPIS.md'
CONFIG_PATH = HERE / 'configs' / 'stage1.json'
REUSED_SOURCES = {
    'e012_folding.py': base.E012_FOLDING,
    'e012_evaluate.py': base.E012_EVALUATE,
    'e011_train.py': base.E011_TRAIN,
    'e015_data_cifar100.py': base.E015_DATA,
    'e017_common.py': E017 / 'e017_common.py',
    'e017_grouping.py': E017 / 'grouping.py',
    'e017_worker.py': E017 / 'worker.py',
}
REQUIRED_DATASET = (
    'tag', 'family', 'seed', 'num_classes', 'checkpoint', 'checkpoint_sha256',
    'split', 'split_sha256', 'baseline_top1', 'baseline_nll',
    'expected_params',
)


def require(condition, message):
    """Podnosi RuntimeError, jeśli warunek nie jest spełniony."""
    if not condition:
        raise RuntimeError(message)


def digest(path):
    """SHA-256 pliku."""
    return base.digest(path)


def read_json(path):
    """Wczytuje JSON."""
    return base.read_json(path)


def write_json_atomic(path, value):
    """Zapisuje JSON atomowo."""
    base.write_json_atomic(path, value)


def utc_now():
    """Bieżący znacznik UTC."""
    return base.utc_now()


def load_config(path=CONFIG_PATH):
    """Wczytuje i waliduje zamrożoną konfigurację E018."""
    cfg = read_json(path)
    require(cfg.get('status') == 'frozen', 'Konfiguracja nie jest zamrożona')
    require(cfg.get('experiment') == EXPERIMENT, 'Zły eksperyment w configu')
    require([float(v) for v in cfg['budgets']] == [0.3, 0.5],
            'Budżety muszą wynosić 30/50% kanałów')
    require([item['name'] for item in cfg['measures']]
            == ['reference', 'cosine'], 'Nieoczekiwane miary')
    require(len(cfg['datasets']) == 6, 'Oczekiwano sześciu checkpointów')
    seen = set()
    for entry in cfg['datasets']:
        missing = [key for key in REQUIRED_DATASET if key not in entry]
        require(not missing, f'Brak pól datasetu: {missing}')
        require(entry['tag'] not in seen, 'Powtórzony tag datasetu')
        seen.add(entry['tag'])
        for key in ('checkpoint', 'split'):
            value = Path(str(entry[key]))
            require(value.is_absolute() and value.is_file(),
                    f'Brak pliku {key} dla {entry["tag"]}')
        require(digest(entry['checkpoint']) == entry['checkpoint_sha256'],
                f"Zły hash checkpointu {entry['tag']}")
        require(digest(entry['split']) == entry['split_sha256'],
                f"Zły hash splitu {entry['tag']}")
        require(int(entry['seed']) in (43, 44, 45),
                'Replikacja używa wyłącznie seedów 43–45')
    run = cfg['run']
    require([int(g) for g in run['gpus']] == [0, 1, 2, 3],
            'Wymagane fizyczne GPU 0–3')
    require(int(run['dev_count']) == 5000, 'Dev musi mieć 5000 próbek')
    return cfg


def build_tasks(cfg):
    """Zwraca 24 zadania: checkpoint × miara × budżet."""
    tasks = []
    for entry in cfg['datasets']:
        for measure in cfg['measures']:
            for budget in cfg['budgets']:
                tasks.append({
                    'dataset': entry['tag'],
                    'family': entry['family'],
                    'seed': int(entry['seed']),
                    'measure': measure['name'],
                    'kind': measure['kind'],
                    'role': measure['role'],
                    'budget': float(budget),
                    'name': (f"{entry['tag']}__{measure['name']}"
                             f"__b{int(round(budget * 100)):02d}"),
                })
    require(len(tasks) == 24, f'Oczekiwano 24 zadań, jest {len(tasks)}')
    return tasks


def dataset_entry(cfg, tag):
    """Zwraca wpis datasetu po tagu."""
    entry = next((item for item in cfg['datasets']
                  if item['tag'] == tag), None)
    require(entry is not None, f'Nieznany dataset: {tag}')
    return entry


def load_bundle(entry):
    """Ładuje model i dane przez helper E017 (nazwa rodziny datasetu)."""
    return base.load_dataset({**entry, 'name': entry['family']})


def folding_module():
    """Moduł E012 folding."""
    return base.folding_module()


def evaluate_module():
    """Moduł E012 evaluate."""
    return base.evaluate_module()
