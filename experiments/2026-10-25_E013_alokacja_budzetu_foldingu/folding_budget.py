"""E013: alokacja budżetu foldingu na operatorze referencyjnym E012.

Eksperyment: 2026-10-25_E013_alokacja_budzetu_foldingu; pełny opis: OPIS.md
(``experiments/2026-10-25_E013_alokacja_budzetu_foldingu/OPIS.md``).

Wejście: model ResNet18/CIFAR-10 zbudowany jak w E012 (``conv1`` 3x3/stride 1,
``maxpool`` Identity, ``fc`` 512->10) oraz wektor keep ratios, po jednym na
grupę foldingu. Wyjście: nowy model na CPU po foldingu, diagnostyka
JSON-serializowalna oraz helpery do alokacji i doboru ``base`` pod globalny
budżet parametrów.

Moduł nie zmienia E012. Operator E012 wczytujemy przez ``importlib`` z pełnej
ścieżki (``load_reference_folding``) i używamy jego funkcji grup, scalania,
walidacji etykiet i odbudowy modułów. Dla ``deterministic=True`` etykiety są
wyznaczane bez klastrowania (``arange(n) % k``), więc testy i solver działają
bez ``hkmeans``. Grupa bez redukcji (``k == n``) dostaje etykiety tożsamościowe
również w trybie niedeterministycznym; dzięki temu ``ratios == 0`` jest
kontrolą tożsamościową i nie wymaga ``hkmeans``.

Alokacje:

- ``allocation_from_tilt``: ``r_g = clip(base + tilt * (2g/(n-1) - 1))``,
- ``allocation_from_random``: ``clip(base + N(0, sigma))`` z
  ``numpy.random.default_rng(seed)``,
- ``solve_base_for_budget``: monotoniczne przeszukanie ``base`` w [0, 1] pod
  limit ``target_params`` (cap ważniejszy niż dokładność); raportuje
  ``delta_params`` i ``ok``.
"""

from __future__ import annotations

import copy
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

EXPERIMENT = '2026-10-25_E013_alokacja_budzetu_foldingu'
OPIS_PATH = 'experiments/2026-10-25_E013_alokacja_budzetu_foldingu/OPIS.md'
E012_EXPERIMENT = '2026-10-24_E012_folding_referencja'
E012_FOLDING_PATH = (
    Path(__file__).resolve().parents[1] / E012_EXPERIMENT / 'folding.py'
)

GROUP_NAMES = (
    'relu_conv',
    'layer1.0.relu1',
    'layer1.1.relu1',
    'layer2.0.relu1',
    'layer2.0.relu2',
    'layer2.1.relu1',
    'layer3.0.relu1',
    'layer3.0.relu2',
    'layer3.1.relu1',
    'layer4.0.relu1',
    'layer4.0.relu2',
    'layer4.1.relu1',
)

_FAMILIES = ('uniform', 'tilt', 'random')
_REFERENCE = None


def _load_module(path, module_name):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Nie można wczytać modułu {module_name} z {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_reference_folding():
    """Wczytuje ``folding.py`` E012 z pełnej ścieżki (bez edycji E012)."""
    global _REFERENCE
    if _REFERENCE is None:
        _REFERENCE = _load_module(
            E012_FOLDING_PATH, '_e013_folding_budget_e012_folding')
    return _REFERENCE


def group_names():
    """Zwraca 12 nazw grup w kolejności depth, jak w E012."""
    return list(GROUP_NAMES)


def _validate_ratio(value, name='ratio'):
    ratio = float(value)
    if not math.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
        raise ValueError(f'{name} musi być w [0, 1], otrzymano {value!r}')
    return ratio


def _validate_bounds(min_ratio, max_ratio):
    lower = _validate_ratio(min_ratio, 'min_ratio')
    upper = _validate_ratio(max_ratio, 'max_ratio')
    if lower > upper:
        raise ValueError(
            f'min_ratio ({lower}) nie może być większe niż max_ratio ({upper})')
    return lower, upper


def _clip(value, lower, upper):
    return float(min(max(float(value), lower), upper))


def allocation_from_tilt(groups, base, tilt, min_ratio=0.1, max_ratio=0.9):
    """Zwraca ``clip(base + tilt * p_g)`` dla ``p_g = 2g/(n-1) - 1``."""
    names = list(groups)
    lower, upper = _validate_bounds(min_ratio, max_ratio)
    base = float(base)
    tilt = float(tilt)
    if not math.isfinite(base) or not math.isfinite(tilt):
        raise ValueError('base i tilt muszą być skończone')
    n_groups = len(names)
    if n_groups == 0:
        return []
    if n_groups == 1:
        return [_clip(base, lower, upper)]
    ratios = []
    for index in range(n_groups):
        position = 2.0 * index / (n_groups - 1) - 1.0
        ratios.append(_clip(base + tilt * position, lower, upper))
    return ratios


def allocation_from_random(groups, base, sigma, seed,
                           min_ratio=0.1, max_ratio=0.9):
    """Zwraca ``clip(base + N(0, sigma))`` z generatora ``default_rng(seed)``."""
    names = list(groups)
    lower, upper = _validate_bounds(min_ratio, max_ratio)
    base = float(base)
    sigma = float(sigma)
    if not math.isfinite(base) or not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError('base musi być skończone, sigma skończone i >= 0')
    n_groups = len(names)
    if n_groups == 0:
        return []
    generator = np.random.default_rng(seed)
    offsets = generator.normal(0.0, sigma, size=n_groups)
    return [_clip(base + float(offset), lower, upper) for offset in offsets]


def _ratios_in_order(ratios, names):
    if isinstance(ratios, dict):
        missing = [name for name in names if name not in ratios]
        unknown = sorted(set(ratios) - set(names))
        if missing or unknown:
            raise KeyError(
                f'Brak ratios dla grup {missing}; nieznane grupy {unknown}')
        values = [ratios[name] for name in names]
    else:
        values = list(ratios)
        if len(values) != len(names):
            raise ValueError(
                f'Oczekiwano {len(names)} ratios, otrzymano {len(values)}')
    return [
        _validate_ratio(value, f'ratios[{index}]')
        for index, value in enumerate(values)
    ]


def _widths_from_ratios(model, ratios):
    reference = load_reference_folding()
    names = group_names()
    values = _ratios_in_order(ratios, names)
    widths = []
    for group in reference.resnet18_groups():
        params = reference.collect_group_params(model, group)
        n_channels = reference.group_n_channels(params, group)
        ratio = values[names.index(group.name)]
        widths.append(max(int(n_channels * (1.0 - ratio)), 1))
    return widths


def _parameter_count_from_widths(model, widths):
    """Liczy parametry po foldingu z szerokości grup (szybki odpowiednik)."""
    reference = load_reference_folding()
    groups = reference.resnet18_groups()
    if len(widths) != len(groups):
        raise ValueError(
            f'Oczekiwano {len(groups)} szerokości, otrzymano {len(widths)}')
    axis_width = {}
    for group, width in zip(groups, widths):
        width = int(width)
        if width < 1:
            raise ValueError(f'Szerokość grupy {group.name} musi być >= 1')
        for param_axis in group.axes:
            key = (param_axis.full_name, int(param_axis.axis))
            if key in axis_width and axis_width[key] != width:
                raise ValueError(f'Sprzeczne szerokości dla osi {key}')
            axis_width[key] = width

    total = 0
    for name, parameter in model.named_parameters():
        shape = list(parameter.shape)
        for axis in range(len(shape)):
            width = axis_width.get((name, axis))
            if width is not None:
                shape[axis] = width
        total += int(math.prod(shape))
    return int(total)


def _base_diagnostics(model, ratios):
    reference = load_reference_folding()
    return {
        'experiment': EXPERIMENT,
        'opis': OPIS_PATH,
        'reference_experiment': E012_EXPERIMENT,
        'group_order': group_names(),
        'ratios': [float(value) for value in ratios],
        'widths': [],
        'nparams_before': reference.count_parameters(model),
        'nparams_after': None,
        'params': None,
        'bypassed': False,
        'all_params_finite': None,
        'finite': None,
        'fc_bias': None,
        'groups': [],
    }


def fold_with_ratios(model, ratios, seed=42, deterministic=False,
                     preserve_fc_bias=False):
    """Folding ResNet18 z osobnym keep ratio dla każdej grupy E012.

    Zwraca ``(new_model_na_CPU, diagnostics)``. Diagnostyka zawiera m.in.
    ``ratios``, ``widths``, ``params``, ``finite`` i ``fc_bias``.
    """
    reference = load_reference_folding()
    names = group_names()
    values = _ratios_in_order(ratios, names)
    ratio_by_name = dict(zip(names, values))

    reference_fc = model.fc
    new_model = copy.deepcopy(model).to(torch.device('cpu'))
    diagnostics = _base_diagnostics(new_model, values)

    for group in reference.resnet18_groups():
        ratio = ratio_by_name[group.name]
        params = reference.collect_group_params(new_model, group)
        n_channels = reference.group_n_channels(params, group)
        features_shape = reference.group_features_shape(params, group)
        n_clusters = max(int(n_channels * (1.0 - ratio)), 1)
        features = reference.concat_group_features(params, group)

        if n_clusters >= n_channels:
            labels = torch.arange(n_channels, dtype=torch.int64)
        elif deterministic:
            labels = torch.arange(n_channels, dtype=torch.int64) % n_clusters
        else:
            labels = reference.cluster_group_features(features, n_clusters)

        k = reference.validate_cluster_labels(labels, n_channels, n_clusters)
        merge = reference.build_merge_matrix(labels, k, n_channels)
        merged_params = reference.merge_group_params(params, group, merge)
        reference._rebuild_group(
            new_model, group, merged_params, labels, k,
            preserve_fc_bias, seed)

        counts = torch.bincount(labels, minlength=k)
        diagnostics['widths'].append(int(k))
        diagnostics['groups'].append({
            'name': group.name,
            'ratio': float(ratio),
            'n_channels': int(n_channels),
            'n_clusters': int(k),
            'width': int(k),
            'features_shape': [int(features_shape[0]), int(features_shape[1])],
            'cluster_labels': [int(value) for value in labels.tolist()],
            'cluster_counts': [int(value) for value in counts.tolist()],
        })

    params_after = reference.count_parameters(new_model)
    finite = reference.state_dict_all_finite(new_model)
    diagnostics['nparams_after'] = params_after
    diagnostics['params'] = params_after
    diagnostics['all_params_finite'] = finite
    diagnostics['finite'] = finite
    policy = 'preserved' if preserve_fc_bias else 'regenerated_official_style'
    diagnostics['fc_bias'] = reference._fc_bias_diagnostics(
        reference_fc, new_model.fc, policy)
    return new_model, diagnostics


def parameter_count_for_ratios(model, ratios):
    """Zwraca liczbę parametrów po deterministycznym foldingu dla ratios."""
    _, diagnostics = fold_with_ratios(
        model, ratios, deterministic=True, preserve_fc_bias=True)
    return int(diagnostics['params'])


def percent_to_target(actual_params, target_params):
    """Zwraca ze znakiem odchylenie od celu w procentach celu.

    ``100 * (actual_params - target_params) / target_params``.
    """
    target = float(target_params)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError(f'target_params musi być > 0, otrzymano {target_params!r}')
    return 100.0 * (float(actual_params) - target) / target


def _normalize_family(family):
    name = str(family).lower()
    if name not in _FAMILIES:
        raise ValueError(
            f'Nieznana rodzina {family!r}; dopuszczalne: {_FAMILIES}')
    return name


def _family_ratios(groups, family, base, tilt, sigma, random_seed):
    if family == 'uniform':
        return allocation_from_tilt(groups, base, 0.0)
    if family == 'tilt':
        return allocation_from_tilt(groups, base, tilt)
    return allocation_from_random(groups, base, sigma, random_seed)


def solve_base_for_budget(model, family, target_params, tilt=0, sigma=0,
                          random_seed=0):
    """Dobiera ``base`` alokacji pod globalny limit ``target_params``.

    ``base`` jest współczynnikiem redukcji (``k = int(n*(1-r))``), więc liczba
    parametrów maleje monotonicznie z ``base``. Najpierw szukamy przedziału
    zmiany (siatka + 30 iteracji bisekcji), potem wybieramy kandydata
    z największą liczbą parametrów nie większą niż cel (cap). Gdy cap jest
    nieosiągalny, zwracamy najbliższą liczbę parametrów.
    """
    family = _normalize_family(family)
    target = float(target_params)
    if not math.isfinite(target) or target < 0.0:
        raise ValueError(
            f'target_params musi być skończone i >= 0, otrzymano {target_params!r}')
    groups = group_names()
    cache = {}

    def evaluate(base):
        base = float(min(max(base, 0.0), 1.0))
        key = round(base, 12)
        if key in cache:
            return cache[key]
        ratios = _family_ratios(
            groups, family, base, tilt, sigma, random_seed)
        widths = _widths_from_ratios(model, ratios)
        params = _parameter_count_from_widths(model, widths)
        result = {
            'base': base,
            'ratios': [float(value) for value in ratios],
            'widths': widths,
            'params': int(params),
        }
        cache[key] = result
        return result

    first = evaluate(0.0)
    last = evaluate(1.0)
    candidates = [first, last]

    if first['params'] <= target:
        chosen = first
    elif last['params'] > target:
        chosen = last
    else:
        bracket = None
        previous = first
        for index in range(1, 65):
            base = index / 64.0
            current = evaluate(base)
            candidates.append(current)
            if current['params'] <= target and previous['params'] > target:
                bracket = (previous['base'], current['base'])
                break
            previous = current
        if bracket is None:
            bracket = (0.0, 1.0)

        low, high = bracket
        for _ in range(30):
            middle = 0.5 * (low + high)
            result = evaluate(middle)
            candidates.append(result)
            if result['params'] <= target:
                high = middle
            else:
                low = middle

        step = 1.0 / (2.0 ** 30)
        for base in (low, low - step, low + step,
                     high, high - step, high + step, 0.0, 1.0):
            candidates.append(evaluate(base))

        feasible = [item for item in candidates if item['params'] <= target]
        if feasible:
            chosen = max(
                feasible,
                key=lambda item: (item['params'], -item['base']))
        else:
            chosen = min(
                candidates,
                key=lambda item: (abs(item['params'] - target), item['params']))

    folded, diagnostics = fold_with_ratios(
        model, chosen['ratios'], deterministic=True, preserve_fc_bias=True)
    del folded
    params = int(diagnostics['params'])
    delta = params - target
    return {
        'base': float(chosen['base']),
        'ratios': [float(value) for value in chosen['ratios']],
        'widths': [int(value) for value in diagnostics['widths']],
        'params': params,
        'delta_params': int(delta) if float(delta).is_integer() else float(delta),
        'delta_percent': percent_to_target(params, target),
        'target_params': int(target) if target.is_integer() else float(target),
        'family': family,
        'ok': bool(params <= target),
    }
