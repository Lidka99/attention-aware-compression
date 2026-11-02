"""E016: wspólne helpery diagnostyki wrażliwości foldingu na pojedyncze grupy.

Eksperyment: 2026-10-27_E016_wrazliwosc_foldingu; pełny opis: OPIS.md
(``experiments/2026-10-27_E016_wrazliwosc_foldingu/OPIS.md``).

Wejście: zamrożony ``configs/stage1.json``, checkpointy E011/E015, splity
train/dev oraz operator E012/E014 importowany bez zmian. Wyjście: helpery do
budowy itemów, ładowania danych, stałej kalibracji bez etykiet, pomiaru
szkody i predyktorów. Moduł nie modyfikuje E011–E015.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = '2026-10-27_E016_wrazliwosc_foldingu'
OPIS_PATH = 'experiments/2026-10-27_E016_wrazliwosc_foldingu/OPIS.md'
CONFIG_PATH = HERE / 'configs' / 'stage1.json'

E011_TRAIN = ROOT / (
    'experiments/2026-10-22_E011_cifar10_resnet18_checkpoint/train.py')
E012_EVALUATE = ROOT / (
    'experiments/2026-10-24_E012_folding_referencja/evaluate.py')
E012_FOLDING = ROOT / (
    'experiments/2026-10-24_E012_folding_referencja/folding.py')
E014_FOLDING_BUDGET = ROOT / (
    'experiments/2026-10-27_E014_replikacja_alokacji/folding_budget.py')
E015_DATA = ROOT / (
    'experiments/2026-10-27_E015_cifar100_alokacja/data_cifar100.py')
SOURCES = {
    'E011_train.py': E011_TRAIN,
    'E012_folding.py': E012_FOLDING,
    'E012_evaluate.py': E012_EVALUATE,
    'E014_folding_budget.py': E014_FOLDING_BUDGET,
    'E015_data_cifar100.py': E015_DATA,
}

_MODULES = {}


def require(condition, message):
    """Podnosi RuntimeError, jeśli warunek nie jest spełniony."""
    if not condition:
        raise RuntimeError(message)


def digest(path):
    """Zwraca SHA-256 pliku."""
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path):
    """Wczytuje JSON z pliku."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json_atomic(path, value):
    """Zapisuje JSON atomowo, z sortowaniem kluczy."""
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
    """Zwraca bieżący znacznik czasu UTC w ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def git_head():
    """Zwraca SHA git HEAD albo 'unknown', jeśli repo nie jest dostępne."""
    import subprocess

    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def git_dirty():
    """Zwraca True, jeśli repo ma niecommitowane zmiany."""
    import subprocess

    try:
        value = subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=ROOT, text=True)
        return bool(value.strip())
    except (OSError, subprocess.CalledProcessError):
        return True


def _load_module(path, name):
    """Wczytuje moduł Python z pełnej ścieżki bez zmiany sys.path na stałe."""
    path = Path(path)
    require(path.is_file(), f'Brak modułu: {path}')
    if name in _MODULES:
        return _MODULES[name]
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None,
            f'Nie można wczytać modułu {name} z {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    _MODULES[name] = module
    return module


def e012_evaluate():
    """Zwraca moduł evaluate.py E012 (bez edycji E012)."""
    return _load_module(E012_EVALUATE, '_e016_e012_evaluate')


def folding_budget():
    """Zwraca moduł folding_budget.py E014 (bez edycji E014)."""
    return _load_module(E014_FOLDING_BUDGET, '_e016_e014_folding_budget')


def load_data_module(dataset):
    """Wczytuje moduł danych datasetu: E011 train.py albo E015 data_cifar100."""
    path = ROOT / str(dataset['data_module'])
    name = f"_e016_data_{dataset['name']}"
    return _load_module(path, name)


def load_config(path=CONFIG_PATH):
    """Wczytuje i waliduje zamrożony config etapu S1."""
    cfg = read_json(path)
    require(cfg.get('status') == 'frozen', 'Config S1 musi być frozen')
    require(cfg.get('protocol_version') == '1.0',
            'Nieoczekiwana wersja protokołu')
    require(cfg.get('official_test_used') is False,
            'Config nie może używać oficjalnego testu')
    require(cfg.get('reserve_used') is False,
            'Config nie może używać rezerwy')
    require([float(value) for value in cfg['levels']]
            == [0.25, 0.5, 0.75, 0.9],
            'Poziomy muszą być dokładnie 25/50/75/90%')
    require(len(cfg['datasets']) == 2, 'Oczekiwano dwóch datasetów')
    names = [item['name'] for item in cfg['datasets']]
    require(names == ['cifar10', 'cifar100'],
            'Kolejność datasetów musi być cifar10, cifar100')
    for dataset in cfg['datasets']:
        require(int(dataset['seed']) == 42,
                'Etap S1 używa wyłącznie seeda 42')
        require(int(dataset['dev_count']) == 5000,
                'Dev musi mieć 5 000 próbek')
        require(int(dataset['bn_samples']) == int(dataset['train_count']),
                'BN musi obejmować pełny train')
        require(float(dataset['baseline_top1']) > 0.0,
                'Baseline top1 musi być dodatni')
    return cfg


def dataset_by_name(cfg, name):
    """Zwraca konfigurację datasetu po nazwie."""
    for dataset in cfg['datasets']:
        if dataset['name'] == name:
            return dataset
    raise KeyError(f'Nieznany dataset: {name}')


def verify_frozen_sources(cfg):
    """Sprawdza, że importowane źródła mają zamrożone hashe z configu."""
    expected = cfg['source_sha256']
    actual = {name: digest(path) for name, path in SOURCES.items()}
    require(actual == expected,
            f'Hash źródeł nie zgadza się z configiem: {actual} != {expected}')
    return actual


def verify_dataset_inputs(dataset):
    """Sprawdza hashe checkpointu i splitu oraz istnienie plików."""
    checkpoint = Path(dataset['checkpoint'])
    split = Path(dataset['split'])
    require(checkpoint.is_file(), f'Brak checkpointu: {checkpoint}')
    require(split.is_file(), f'Brak splitu: {split}')
    require(digest(checkpoint) == dataset['checkpoint_sha256'],
            f'Zły hash checkpointu: {checkpoint}')
    require(digest(split) == dataset['split_sha256'],
            f'Zły hash splitu: {split}')


def load_dataset_bundle(dataset, verify_hashes=True):
    """Ładuje obrazy, split, checkpoint i model CPU jednego datasetu.

    Zwraca słownik z kluczami: ``module``, ``images``, ``labels``,
    ``classes``, ``split``, ``train_indices``, ``dev_indices``,
    ``source_cpu``.
    """
    if verify_hashes:
        verify_dataset_inputs(dataset)
    module = load_data_module(dataset)
    name = dataset['name']
    if name == 'cifar10':
        images, labels, classes = module.load_cifar10_train(module.DATA_ROOT)
        split = read_json(dataset['split'])
        require(split.get('official_test_used') is False,
                'Split CIFAR-10 używa oficjalnego testu')
        module.validate_split(split, labels, int(split['seed']))
        train_indices = np.asarray(split['train'], dtype=np.int64)
        dev_indices = np.asarray(split['dev'], dtype=np.int64)
        model = module.build_model()
    elif name == 'cifar100':
        images, labels, classes = module.load_cifar100_train(module.DATA_ROOT)
        split = module.load_split(dataset['split'])
        require(split.get('official_test_used') is False,
                'Split CIFAR-100 używa oficjalnego testu')
        module.validate_split(split, labels, module.SPLIT_SEED)
        require('reserve' in split['splits'],
                'Split CIFAR-100 musi mieć rezerwę')
        require(len(split['splits']['reserve']) == 5000,
                'Rezerwa CIFAR-100 musi mieć 5 000 próbek')
        train_indices = np.asarray(split['splits']['train'], dtype=np.int64)
        dev_indices = np.asarray(split['splits']['dev'], dtype=np.int64)
        model = module.build_model(num_classes=int(dataset['num_classes']))
    else:
        raise KeyError(f'Nieznany dataset: {name}')

    require(len(classes) == int(dataset['num_classes']),
            f'Zła liczba klas dla {name}')
    require(len(train_indices) == int(dataset['train_count']),
            f'Zły train count dla {name}')
    require(len(dev_indices) == int(dataset['dev_count']),
            f'Zły dev count dla {name}')
    require(len(train_indices) == int(dataset['bn_samples']),
            f'BN samples nie pokrywa train dla {name}')
    require(len(set(train_indices.tolist())) == len(train_indices),
            f'Duplikaty w train dla {name}')
    require(len(set(dev_indices.tolist())) == len(dev_indices),
            f'Duplikaty w dev dla {name}')
    require(not set(train_indices.tolist()) & set(dev_indices.tolist()),
            f'Train i dev nachodzą na siebie dla {name}')

    payload = torch.load(dataset['checkpoint'], map_location='cpu',
                         weights_only=True)
    require(isinstance(payload, dict)
            and isinstance(payload.get('state_dict'), dict),
            f'Checkpoint {name} bez state_dict')
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    parameters = int(sum(value.numel() for value in model.parameters()))
    require(parameters == int(dataset['expected_parameters']),
            f'Zła liczba parametrów {name}: {parameters}')
    return {
        'module': module,
        'images': images,
        'labels': labels,
        'classes': classes,
        'split': split,
        'train_indices': train_indices,
        'dev_indices': dev_indices,
        'source_cpu': model,
    }


def make_eval_loader(module, images, labels, indices, cfg, seed):
    """Tworzy deterministyczny loader eval (dev_transform, shuffle=False)."""
    return e012_evaluate().make_eval_loader(
        module, images, labels, indices, cfg, int(seed))


def make_bn_loader(module, images, labels, train_indices, cfg):
    """Tworzy loader korekty BN na pełnym train (train_transform, shuffle)."""
    return e012_evaluate().make_bn_loader(
        module, images, labels, train_indices, cfg)


def calibration_indices(train_indices, seed, samples):
    """Zwraca stały, posortowany podzbiór train bez używania etykiet."""
    values = np.asarray(train_indices, dtype=np.int64)
    samples = int(samples)
    require(0 < samples <= len(values),
            f'calibration_samples={samples} poza zakresem')
    generator = np.random.default_rng(int(seed))
    chosen = generator.choice(values, size=samples, replace=False)
    chosen.sort()
    return chosen.astype(np.int64)


def indices_digest(indices):
    """Zwraca SHA-256 indeksów jako little-endian int64."""
    array = np.ascontiguousarray(indices, dtype='<i8')
    return hashlib.sha256(array.tobytes()).hexdigest()


def collect_logits(model, loader, device):
    """Zbiera logity w kolejności loadera; etykiety nie są używane."""
    model.eval()
    parts = []
    with torch.inference_mode():
        for images, _targets in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            require(logits.dtype == torch.float32,
                    f'Logity nie są float32: {logits.dtype}')
            require(bool(torch.isfinite(logits).all()),
                    'Niefinitywne logity')
            parts.append(logits.detach().to('cpu', torch.float32))
    require(parts, 'Pusty loader logitów')
    return torch.cat(parts, dim=0).numpy()


def logit_change(candidate, baseline):
    """Liczy zmianę logitów względem baseline bez używania etykiet."""
    import torch.nn.functional as functional

    candidate = torch.as_tensor(candidate, dtype=torch.float32)
    baseline = torch.as_tensor(baseline, dtype=torch.float32)
    require(candidate.shape == baseline.shape, 'Niezgodne kształty logitów')
    diff = candidate - baseline
    logp_candidate = functional.log_softmax(candidate, dim=1)
    logp_baseline = functional.log_softmax(baseline, dim=1)
    return {
        'prediction_agreement': float(
            (candidate.argmax(1) == baseline.argmax(1)).float().mean()),
        'logit_mae': float(diff.abs().mean()),
        'logit_rmse': float(diff.pow(2).mean().sqrt()),
        'logit_max_abs': float(diff.abs().max()),
        'kl_baseline_candidate': float(functional.kl_div(
            logp_candidate, logp_baseline.exp(), reduction='batchmean')),
        'kl_candidate_baseline': float(functional.kl_div(
            logp_baseline, logp_candidate.exp(), reduction='batchmean')),
    }


def group_names():
    """Zwraca 12 nazw grup E012 w kolejności depth."""
    names = folding_budget().group_names()
    require(len(names) == 12, 'Oczekiwano 12 grup E012')
    return list(names)


def group_slug(name):
    """Zamienia nazwę grupy na bezpieczny fragment katalogu."""
    return str(name).replace('.', '_').replace('/', '_')


def item_name(dataset, group_index, level):
    """Buduje unikalną nazwę itemu S1."""
    if group_index is None:
        return f'{dataset["name"]}__control__r000'
    name = group_names()[int(group_index)]
    return (f'{dataset["name"]}__g{int(group_index):02d}_{group_slug(name)}'
            f'__r{int(round(float(level) * 100)):03d}')


def build_items(cfg, smoke=False):
    """Buduje listę unikalnych itemów S1 (98 pełnych, 8 smoke)."""
    items = []
    names = group_names()
    if smoke:
        smoke_levels = [0.25, 0.5, 0.75, 0.9]
        smoke_groups = [0, 3, 6, 9]
        for dataset in cfg['datasets']:
            for group_index, level in zip(smoke_groups, smoke_levels):
                items.append({
                    'name': item_name(dataset, group_index, level),
                    'dataset': dataset['name'],
                    'group_index': int(group_index),
                    'group_name': names[int(group_index)],
                    'level': float(level),
                    'control': False,
                })
        require(len(items) == 8, 'Smoke musi mieć 8 itemów')
    else:
        for dataset in cfg['datasets']:
            items.append({
                'name': item_name(dataset, None, 0.0),
                'dataset': dataset['name'],
                'group_index': None,
                'group_name': None,
                'level': 0.0,
                'control': True,
            })
            for group_index, name in enumerate(names):
                for level in cfg['levels']:
                    items.append({
                        'name': item_name(dataset, group_index, level),
                        'dataset': dataset['name'],
                        'group_index': int(group_index),
                        'group_name': name,
                        'level': float(level),
                        'control': False,
                    })
        require(len(items) == 98, f'Pełny S1 musi mieć 98 itemów, jest {len(items)}')
    require(len({item['name'] for item in items}) == len(items),
            'Nazwy itemów nie są unikalne')
    return items


def shard_items(items, gpu_count=4):
    """Dzieli itemy datasetu round-robin na ``gpu_count`` rozłącznych shardów."""
    shards = {index: [] for index in range(int(gpu_count))}
    for offset, item in enumerate(items):
        shards[offset % int(gpu_count)].append(item)
    all_names = [item['name'] for shard in shards.values() for item in shard]
    require(len(all_names) == len(items) and len(set(all_names)) == len(items),
            'Shardy nie są rozłączne i kompletne')
    return shards


def ratios_for_item(item):
    """Zwraca 12 ratios: zero wszędzie poza wskazaną grupą."""
    ratios = [0.0] * 12
    if not item['control']:
        ratios[int(item['group_index'])] = float(item['level'])
    return ratios


def fold_item(source_cpu, item, fold_seed=42):
    """Wykonuje folding jednej grupy (albo kontrolę ratio 0)."""
    ratios = ratios_for_item(item)
    folded, diagnostics = folding_budget().fold_with_ratios(
        source_cpu, ratios, seed=int(fold_seed), deterministic=False,
        preserve_fc_bias=True)
    return folded, diagnostics


def weight_reconstruction_error(source_cpu, group_index, diagnostics):
    """Względny błąd Frobeniusa projekcji parametrów grupy po scaleniu.

    Dla ratio 0 lub kontroli zwraca 0.0. Predyktor nie używa obrazów ani
    etykiet; korzysta z etykiet klastrów faktycznie zastosowanych w foldingu.
    """
    if group_index is None:
        return 0.0
    reference = folding_budget().load_reference_folding()
    group = reference.resnet18_groups()[int(group_index)]
    entry = next(
        (value for value in diagnostics['groups']
         if value['name'] == group.name), None)
    require(entry is not None, f'Brak diagnostyki grupy {group.name}')
    n_channels = int(entry['n_channels'])
    n_clusters = int(entry['n_clusters'])
    if n_clusters >= n_channels:
        return 0.0
    labels = torch.as_tensor(entry['cluster_labels'], dtype=torch.int64)
    params = reference.collect_group_params(source_cpu, group)
    features = reference.concat_group_features(params, group).detach().float()
    merge = reference.build_merge_matrix(labels, n_clusters, n_channels)
    reconstructed = merge.transpose(0, 1) @ (merge @ features)
    numerator = torch.linalg.vector_norm(features - reconstructed)
    denominator = torch.linalg.vector_norm(features)
    if float(denominator) <= 0.0:
        return 0.0
    return float(numerator / denominator)


def stage_of_group(group_name):
    """Zwraca przybliżony etap sieci dla nazwy grupy."""
    if group_name is None:
        return 'control'
    if group_name == 'relu_conv':
        return 'stem'
    return str(group_name).split('.')[0]


def finite_float(value, name):
    """Sprawdza, że wartość jest skończonym floatem."""
    result = float(value)
    require(math.isfinite(result), f'{name} musi być skończone')
    return result
