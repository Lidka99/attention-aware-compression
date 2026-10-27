"""E013: własna implementacja pruningu magnitudowego ResNet18 (MAG1/MAG2).

Eksperyment: 2026-10-25_E013_alokacja_budzetu_foldingu; pełny opis: OPIS.md.
Wejście: model torchvision ResNet18 przystosowany do CIFAR-10 tak jak w E012
(``conv1`` 3x3/stride 1, ``maxpool`` Identity, ``fc`` 512->10) oraz budżet
``ratio``. Wyjście: niezależna kopia modelu na CPU po pruning'u i diagnostyka
JSON-serializowalna.

Implementacja odtwarza reguły ``ResNet18_MagnitudePruning`` z
``compression/mag_prune.py`` (commit
``694691556e8314a835178a741938dd3d4d80c171``, repo
``osaukh/folding_as_projection``), ale jest napisana od nowa w tym
repozytorium. ``MAG1`` oznacza ``p=1`` (norma L1), ``MAG2`` oznacza ``p=2``
(norma L2); domyślnie ``p=2``, jak w źródle.

Zasady wyboru kanałów:

- wynik conv jest oceniany przez normę ``p`` wiersza oryginalnej macierzy
  wag, liczoną przed jakimkolwiek obcięciem wejścia,
- ``stem`` zachowuje ``max(floor(out * (1 - ratio)), 1)`` kanałów,
- w każdym bloku ``conv1`` zachowuje tyle samo kanałów wyjściowych,
- ``conv2`` wybiera własne kanały wyjściowe tylko wtedy, gdy blok ma
  niepusty ``downsample`` (pierwsze bloki stage2/3/4); w przeciwnym razie
  ``keep2 = in_keep``, aby zachować zgodność gałęzi residualnej,
- ``downsample`` ma wejście ``in_keep`` i wyjście ``keep2``,
- ``BatchNorm2d`` jest odbudowywany od zera: wycinane są gamma, beta,
  ``running_mean`` i ``running_var``, a ``eps``, ``momentum`` i
  ``track_running_stats`` pozostają domyślne; ``num_batches_tracked`` wynosi 0,
- ``fc`` wycina wejście według ostatniego ``keep``, a pełny bias 10 klas jest
  zachowany.

Świadome odchylenie od źródła: zamiast niestabilnego ``topk``/``argsort``
używamy ``torch.argsort(descending=True, stable=True)``, więc przy równych
normach wygrywa niższy oryginalny indeks. ``ratio=0`` jest bypassem i zwraca
dokładnie niezmieniony model. Nie oceniamy konkatenacji wag ani parametrów BN.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

EXPERIMENT = '2026-10-25_E013_alokacja_budzetu_foldingu'
OPIS_PATH = 'experiments/2026-10-25_E013_alokacja_budzetu_foldingu/OPIS.md'
REFERENCE_COMMIT = '694691556e8314a835178a741938dd3d4d80c171'
MIN_CHANNELS = 1
STAGE_CHANNELS = ((1, 64), (2, 128), (3, 256), (4, 512))


def _method_name(p):
    """Zwraca nazwę wariantu magnitude dla danego wykładnika normy."""
    if float(p) == 1.0:
        return 'MAG1'
    if float(p) == 2.0:
        return 'MAG2'
    return f'MAG_p{float(p):g}'


def _validate_ratio(ratio):
    """Sprawdza, że budżet jest skończony i należy do przedziału [0, 1]."""
    value = float(ratio)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(
            f'ratio musi być skończone i należeć do [0, 1], otrzymano {ratio!r}')
    return value


def _validate_p(p):
    """Sprawdza, że wykładnik normy magnitudy jest dodatni i skończony."""
    value = float(p)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f'p musi być skończone i dodatnie, otrzymano {p!r}')
    return value


def get_module_by_name(model, full_name):
    """Zwraca moduł modelu po nazwie z kropkami, także dla indeksów list."""
    module = model
    for part in full_name.split('.'):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _replace_module(model, full_name, new_module):
    """Podmienia moduł w drzewie modelu, zachowując nazwę i indeksy."""
    parts = full_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def count_parameters(model):
    """Zwraca liczbę wszystkich parametrów trenowalnych i nietrenowalnych."""
    return int(sum(parameter.numel() for parameter in model.parameters()))


def state_dict_all_finite(model):
    """Sprawdza, czy wszystkie parametry i bufory stanu są skończone."""
    return all(
        bool(torch.isfinite(tensor).all())
        for tensor in model.state_dict().values()
    )


def _row_norms(weight, p):
    """Liczy normę ``p`` każdego wiersza oryginalnej macierzy wag conv."""
    if weight.dim() < 2:
        raise ValueError(
            f'Waga conv musi mieć co najmniej 2 wymiary, ma {tuple(weight.shape)}')
    rows = weight.detach().to(device='cpu').reshape(int(weight.shape[0]), -1)
    return torch.norm(rows, p=float(p), dim=1)


def _keep_count(n_channels, ratio):
    """Zwraca liczbę kanałów do zachowania: floor(n * (1-ratio)), min. 1."""
    n_channels = int(n_channels)
    if n_channels < 1:
        raise ValueError(f'Liczba kanałów musi być dodatnia, otrzymano {n_channels}')
    keep = int(math.floor(float(n_channels) * (1.0 - float(ratio))))
    return max(keep, MIN_CHANNELS)


def _keep_indices(scores, k):
    """Wybiera ``k`` największych norm; remisy rozstrzyga niższy indeks."""
    values = torch.as_tensor(scores).detach().to(device='cpu').reshape(-1)
    n_scores = int(values.numel())
    k = int(k)
    if k < 1 or k > n_scores:
        raise ValueError(f'k={k} poza zakresem 1..{n_scores}')
    order = torch.argsort(values, descending=True, stable=True)
    return order[:k].to(dtype=torch.int64, device='cpu')


def _as_keep(keep, n_channels, name):
    """Waliduje indeksy kanałów i zwraca je jako ``torch.int64`` na CPU."""
    indices = torch.as_tensor(
        keep, dtype=torch.int64, device='cpu').reshape(-1)
    if int(indices.numel()) < 1:
        raise ValueError(f'{name}: puste indeksy kanałów')
    if int(indices.min()) < 0 or int(indices.max()) >= int(n_channels):
        raise ValueError(
            f'{name}: indeks poza zakresem 0..{int(n_channels) - 1}')
    if int(torch.unique(indices).numel()) != int(indices.numel()):
        raise ValueError(f'{name}: powtórzone indeksy kanałów')
    return indices


def _has_downsample(block):
    """Sprawdza, czy blok ma niepusty downsample Conv2d + BatchNorm2d."""
    downsample = getattr(block, 'downsample', None)
    return (
        isinstance(downsample, nn.Sequential)
        and len(downsample) >= 2
        and isinstance(downsample[0], nn.Conv2d)
        and isinstance(downsample[1], nn.BatchNorm2d)
    )


def _rebuild_conv(model, name, in_keep, out_keep, diagnostics):
    """Odbudowuje Conv2d, wycinając osie 1 (wejście) i 0 (wyjście)."""
    old_module = get_module_by_name(model, name)
    if not isinstance(old_module, nn.Conv2d):
        raise TypeError(f'{name} nie jest Conv2d')

    weight = old_module.weight.detach()
    out_idx = (
        torch.arange(old_module.out_channels, dtype=torch.int64)
        if out_keep is None
        else _as_keep(out_keep, old_module.out_channels, f'{name}.out')
    )
    in_idx = (
        torch.arange(old_module.in_channels, dtype=torch.int64)
        if in_keep is None
        else _as_keep(in_keep, old_module.in_channels, f'{name}.in')
    )

    new_weight = weight.index_select(0, out_idx).index_select(1, in_idx).clone()
    new_module = nn.Conv2d(
        in_channels=int(in_idx.numel()),
        out_channels=int(out_idx.numel()),
        kernel_size=old_module.kernel_size,
        stride=old_module.stride,
        padding=old_module.padding,
        dilation=old_module.dilation,
        groups=old_module.groups,
        bias=old_module.bias is not None,
        padding_mode=old_module.padding_mode,
    )
    new_module.weight = nn.Parameter(new_weight)
    if old_module.bias is not None:
        new_module.bias = nn.Parameter(
            old_module.bias.detach().index_select(0, out_idx).clone())

    _replace_module(model, name, new_module)
    diagnostics['modules'].append({
        'name': name,
        'kind': 'conv',
        'in_channels_before': int(old_module.in_channels),
        'in_channels_after': int(new_module.in_channels),
        'out_channels_before': int(old_module.out_channels),
        'out_channels_after': int(new_module.out_channels),
        'in_keep': None if in_keep is None else [int(v) for v in in_idx.tolist()],
        'out_keep': [int(v) for v in out_idx.tolist()],
    })
    return new_module


def _rebuild_bn(model, name, keep, diagnostics):
    """Odbudowuje BatchNorm2d z domyślnymi eps/momentum i wycina bufory."""
    old_module = get_module_by_name(model, name)
    if not isinstance(old_module, nn.BatchNorm2d):
        raise TypeError(f'{name} nie jest BatchNorm2d')

    keep_idx = _as_keep(keep, old_module.num_features, name)
    new_module = nn.BatchNorm2d(int(keep_idx.numel()))
    if old_module.weight is not None:
        new_module.weight = nn.Parameter(
            old_module.weight.detach().index_select(0, keep_idx).clone())
    if old_module.bias is not None:
        new_module.bias = nn.Parameter(
            old_module.bias.detach().index_select(0, keep_idx).clone())
    if old_module.running_mean is not None:
        new_module.running_mean = old_module.running_mean.detach().index_select(
            0, keep_idx).clone()
    if old_module.running_var is not None:
        new_module.running_var = old_module.running_var.detach().index_select(
            0, keep_idx).clone()

    _replace_module(model, name, new_module)
    diagnostics['modules'].append({
        'name': name,
        'kind': 'batch_norm',
        'num_features_before': int(old_module.num_features),
        'num_features_after': int(new_module.num_features),
        'keep': [int(v) for v in keep_idx.tolist()],
        'num_batches_tracked': int(new_module.num_batches_tracked.item()),
        'eps': float(new_module.eps),
        'momentum': (
            None if new_module.momentum is None
            else float(new_module.momentum)
        ),
    })
    return new_module


def _rebuild_linear(model, name, keep, diagnostics):
    """Odbudowuje Linear, wycinając wejście i zachowując pełny bias."""
    old_module = get_module_by_name(model, name)
    if not isinstance(old_module, nn.Linear):
        raise TypeError(f'{name} nie jest Linear')

    keep_idx = _as_keep(keep, old_module.in_features, name)
    new_module = nn.Linear(
        in_features=int(keep_idx.numel()),
        out_features=int(old_module.out_features),
        bias=old_module.bias is not None,
    )
    new_module.weight = nn.Parameter(
        old_module.weight.detach().index_select(1, keep_idx).clone())
    if old_module.bias is not None:
        new_module.bias = nn.Parameter(old_module.bias.detach().clone())

    _replace_module(model, name, new_module)
    diagnostics['modules'].append({
        'name': name,
        'kind': 'linear',
        'in_features_before': int(old_module.in_features),
        'in_features_after': int(new_module.in_features),
        'out_features': int(new_module.out_features),
        'in_keep': [int(v) for v in keep_idx.tolist()],
        'bias_preserved': bool(
            old_module.bias is not None
            and new_module.bias is not None
            and torch.equal(old_module.bias, new_module.bias)
        ),
    })
    return new_module


def _fc_diagnostics(old_fc, new_fc, keep, bypassed):
    """Buduje diagnostykę fc bez kopiowania pełnych macierzy wag."""
    return {
        'in_features_before': int(old_fc.in_features),
        'in_features_after': int(new_fc.in_features),
        'out_features': int(new_fc.out_features),
        'in_keep': (
            None if keep is None
            else [int(v) for v in torch.as_tensor(keep).reshape(-1).tolist()]
        ),
        'bias_present': new_fc.bias is not None,
        'bias_preserved': bool(
            old_fc.bias is not None
            and new_fc.bias is not None
            and torch.equal(old_fc.bias, new_fc.bias)
        ),
        'bypassed': bool(bypassed),
    }


def _base_diagnostics(model, ratio, p):
    """Tworzy wspólny szkielet diagnostyki dla wszystkich budżetów."""
    return {
        'experiment': EXPERIMENT,
        'opis': OPIS_PATH,
        'reference_commit': REFERENCE_COMMIT,
        'method': _method_name(p),
        'ratio': float(ratio),
        'keep_ratio': float(1.0 - float(ratio)),
        'p': float(p),
        'min_channels': int(MIN_CHANNELS),
        'bypassed': False,
        'score_rule': 'p_norm_of_original_conv_rows_before_input_slice',
        'keep_rule': 'argsort_descending_stable_true_lower_index_wins',
        'nparams_before': count_parameters(model),
        'nparams_after': None,
        'all_params_finite': None,
        'modules': [],
        'groups': [],
        'fc': None,
    }


def _score_names():
    """Zwraca nazwy conv, których wiersze są oceniane przed obcinaniem."""
    names = ['conv1']
    for stage, _channels in STAGE_CHANNELS:
        for index in (0, 1):
            prefix = f'layer{stage}.{index}'
            names.append(f'{prefix}.conv1')
            names.append(f'{prefix}.conv2')
            if stage > 1 and index == 0:
                names.append(f'{prefix}.downsample.0')
    return names


def prune(model, ratio, p=2):
    """Przycina ResNet18 MAG1/MAG2; zwraca (model_na_CPU, diagnostyka).

    Parametry
    ---------
    model : torch.nn.Module
        ResNet18 dla CIFAR-10, taki jak w E012. Wejściowy model nie jest
        modyfikowany.
    ratio : float
        Udział kanałów do usunięcia, ``0 <= ratio <= 1``. ``0`` oznacza
        bypass bez żadnej zmiany.
    p : float
        Wykładnik normy wiersza wagi: ``1`` to MAG1, ``2`` to MAG2
        (domyślnie, jak w źródle).

    Zwraca
    ------
    tuple
        ``(new_model, diagnostics)``. ``new_model`` jest niezależną kopią na
        CPU. ``diagnostics`` zawiera indeksy ``keep`` per moduł, grupy
        stage'y, liczbę parametrów przed/po oraz informację o skończoności.
    """
    ratio = _validate_ratio(ratio)
    p = _validate_p(p)

    new_model = copy.deepcopy(model).to(device=torch.device('cpu'))
    diagnostics = _base_diagnostics(new_model, ratio, p)
    reference_fc = copy.deepcopy(new_model.fc)

    if ratio == 0.0:
        diagnostics['bypassed'] = True
        diagnostics['nparams_after'] = diagnostics['nparams_before']
        diagnostics['all_params_finite'] = state_dict_all_finite(new_model)
        diagnostics['fc'] = _fc_diagnostics(
            reference_fc, new_model.fc, None, bypassed=True)
        return new_model, diagnostics

    scores = {
        name: _row_norms(get_module_by_name(new_model, name).weight, p)
        for name in _score_names()
    }

    stem = get_module_by_name(new_model, 'conv1')
    stem_keep = _keep_indices(
        scores['conv1'], _keep_count(stem.out_channels, ratio))
    _rebuild_conv(
        new_model, 'conv1', in_keep=None, out_keep=stem_keep,
        diagnostics=diagnostics)
    _rebuild_bn(new_model, 'bn1', stem_keep, diagnostics=diagnostics)
    previous_keep = stem_keep
    diagnostics['groups'].append({
        'name': 'stem',
        'modules': ['conv1', 'bn1'],
        'width_before': int(stem.out_channels),
        'width_after': int(stem_keep.numel()),
        'keep': [int(v) for v in stem_keep.tolist()],
    })

    for stage, _channels in STAGE_CHANNELS:
        blocks = getattr(new_model, f'layer{stage}')
        stage_input = previous_keep
        block_entries = []

        for index, block in enumerate(blocks):
            prefix = f'layer{stage}.{index}'
            in_keep = previous_keep if index == 0 else keep2

            conv1 = get_module_by_name(new_model, f'{prefix}.conv1')
            keep1 = _keep_indices(
                scores[f'{prefix}.conv1'],
                _keep_count(conv1.out_channels, ratio),
            )
            _rebuild_conv(
                new_model, f'{prefix}.conv1', in_keep=in_keep,
                out_keep=keep1, diagnostics=diagnostics)
            _rebuild_bn(
                new_model, f'{prefix}.bn1', keep1, diagnostics=diagnostics)

            conv2 = get_module_by_name(new_model, f'{prefix}.conv2')
            conv2_count = _keep_count(conv2.out_channels, ratio)
            downsample_keep = None
            if _has_downsample(block):
                keep2 = _keep_indices(
                    scores[f'{prefix}.conv2'], conv2_count)
                _rebuild_conv(
                    new_model, f'{prefix}.downsample.0', in_keep=in_keep,
                    out_keep=keep2, diagnostics=diagnostics)
                _rebuild_bn(
                    new_model, f'{prefix}.downsample.1', keep2,
                    diagnostics=diagnostics)
                downsample_keep = keep2
            else:
                keep2 = in_keep

            _rebuild_conv(
                new_model, f'{prefix}.conv2', in_keep=keep1,
                out_keep=keep2, diagnostics=diagnostics)
            _rebuild_bn(
                new_model, f'{prefix}.bn2', keep2, diagnostics=diagnostics)

            block_entries.append({
                'name': prefix,
                'in_keep': [int(v) for v in torch.as_tensor(in_keep).tolist()],
                'conv1_out_keep': [int(v) for v in keep1.tolist()],
                'conv2_out_keep': [int(v) for v in keep2.tolist()],
                'downsample_out_keep': (
                    None if downsample_keep is None
                    else [int(v) for v in downsample_keep.tolist()]
                ),
                'widths': {
                    'in': int(torch.as_tensor(in_keep).numel()),
                    'conv1_out': int(keep1.numel()),
                    'out': int(keep2.numel()),
                },
            })
            previous_keep = keep2

        diagnostics['groups'].append({
            'name': f'layer{stage}',
            'width_in': int(torch.as_tensor(stage_input).numel()),
            'width_out': int(previous_keep.numel()),
            'blocks': block_entries,
        })

    fc_keep = previous_keep
    new_fc = _rebuild_linear(
        new_model, 'fc', fc_keep, diagnostics=diagnostics)
    diagnostics['fc'] = _fc_diagnostics(
        reference_fc, new_fc, fc_keep, bypassed=False)
    diagnostics['nparams_after'] = count_parameters(new_model)
    diagnostics['all_params_finite'] = state_dict_all_finite(new_model)
    return new_model, diagnostics
