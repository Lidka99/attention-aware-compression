"""E012: własny operator foldingu ResNet18/CIFAR-10 jako adapter torchvision.

Eksperyment: 2026-10-24_E012_folding_referencja; pełny opis: OPIS.md.
Wejście: model torchvision ResNet18 przystosowany do CIFAR-10 (E011
build_model: conv1 3x3/stride1, maxpool Identity, fc 512->10) na CPU.
Wyjście: nowy model (deepcopy na CPU) oraz diagnostyka słownikowa.

Mapowanie 12 grup i kolejność cech odtwarzają commit
694691556e8314a835178a741938dd3d4d80c171 repo osaukh/folding_as_projection:
models/resnet.py (get_axis_to_perm_ResNet18, override=False),
compression/base_resnet.py, compression/fold.py oraz
utils/weight_clustering.py.

Kod nie importuje kodu referencyjnego ani E011. Import hkmeans/sklearn jest
leniwy, dzięki czemu helpery grup i odbudowy modułów działają bez klastrowania.

Odchylenia od źródła (świadome, opisane też w test_folding.py):
- ratio == 0.0 to kontrola bypass: model wraca bez klastrowania i odbudowy.
- HKMeans ma n_jobs=1 zamiast n_jobs=-1 (wymaganie E012).
- Niewykorzystany klaster przerywa folding zamiast remapować etykiety.
- Ukryty bias fc jest losowany z lokalnego generatora seeda (fork_rng), a nie
  z globalnego stanu RNG; rozkład inicjalizacji jest taki jak nn.Linear.
- preserve_fc_bias=True zachowuje stary bias fc; False emuluje oficjalny
  losowy bias (param_dict nie zawiera biasu, ale nn.Linear ma bias=True).
"""

import copy
import math
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn

EXPERIMENT = '2026-10-24_E012_folding_referencja'
OPIS_PATH = 'experiments/2026-10-24_E012_folding_referencja/OPIS.md'
REFERENCE_COMMIT = '694691556e8314a835178a741938dd3d4d80c171'


@dataclass(frozen=True)
class ParamAxis:
    full_name: str
    axis: int


@dataclass(frozen=True)
class FoldGroup:
    name: str
    axes: tuple


def _conv(name, p_in, p_out):
    return {f'{name}.weight': (p_out, p_in, None, None)}


def _norm(name, p):
    return {
        f'{name}.weight': (p,),
        f'{name}.bias': (p,),
        f'{name}.running_mean': (p,),
        f'{name}.running_var': (p,),
    }


def _basicblock(name, p_in, p_out):
    return {
        **_conv(f'{name}.conv1', p_in, f'{name}.relu1'),
        **_norm(f'{name}.bn1', f'{name}.relu1'),
        **_conv(f'{name}.conv2', f'{name}.relu1', p_out),
        **_norm(f'{name}.bn2', p_out),
    }


def _dense(name, p_in, p_out):
    return {
        f'{name}.weight': (p_out, p_in),
        f'{name}.bias': (p_out,),
    }


def _axis_to_perm():
    sh = 'downsample'
    fc = 'fc'
    axis_to_perm = {}
    axis_to_perm.update(_conv('conv1', None, 'relu_conv'))
    axis_to_perm.update(_norm('bn1', 'relu_conv'))

    axis_to_perm.update(_basicblock('layer1.0', 'relu_conv', 'relu_conv'))
    axis_to_perm.update(_basicblock('layer1.1', 'relu_conv', 'relu_conv'))

    axis_to_perm.update(_basicblock('layer2.0', 'relu_conv', 'layer2.0.relu2'))
    axis_to_perm.update(_conv(f'layer2.0.{sh}.0', 'relu_conv', 'layer2.0.relu2'))
    axis_to_perm.update(_norm(f'layer2.0.{sh}.1', 'layer2.0.relu2'))
    axis_to_perm.update(_basicblock('layer2.1', 'layer2.0.relu2', 'layer2.0.relu2'))

    axis_to_perm.update(_basicblock('layer3.0', 'layer2.0.relu2', 'layer3.0.relu2'))
    axis_to_perm.update(_conv(f'layer3.0.{sh}.0', 'layer2.0.relu2', 'layer3.0.relu2'))
    axis_to_perm.update(_norm(f'layer3.0.{sh}.1', 'layer3.0.relu2'))
    axis_to_perm.update(_basicblock('layer3.1', 'layer3.0.relu2', 'layer3.0.relu2'))

    axis_to_perm.update(_basicblock('layer4.0', 'layer3.0.relu2', 'layer4.0.relu2'))
    axis_to_perm.update(_conv(f'layer4.0.{sh}.0', 'layer3.0.relu2', 'layer4.0.relu2'))
    axis_to_perm.update(_norm(f'layer4.0.{sh}.1', 'layer4.0.relu2'))
    axis_to_perm.update(_basicblock('layer4.1', 'layer4.0.relu2', 'layer4.0.relu2'))

    axis_to_perm.update(_dense(fc, 'layer4.0.relu2', None))
    return axis_to_perm


def resnet18_axis_to_perm():
    """Zwraca pełne mapowanie moduł->permutacje dla torchvision ResNet18."""
    return _axis_to_perm()


def resnet18_groups():
    """Zwraca 12 grup w kolejności pierwszego wystąpienia permutacji."""
    perm_to_axes = OrderedDict()
    for full_name, perms in _axis_to_perm().items():
        for axis, perm in enumerate(perms):
            if perm is not None:
                perm_to_axes.setdefault(perm, []).append(ParamAxis(full_name, axis))
    return [
        FoldGroup(name, tuple(axes))
        for name, axes in perm_to_axes.items()
    ]


def group_order():
    return [group.name for group in resnet18_groups()]


def get_module_by_name(model, full_name):
    module = model
    for part in full_name.split('.'):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _resolve_tensor(model, full_name):
    value = get_module_by_name(model, full_name)
    data = value.data if hasattr(value, 'data') else value
    return data.detach()


def collect_group_params(model, group):
    """Zbiera bieżące tensory wag/stats dla osi grupy (tak jak źródło)."""
    return {pa.full_name: _resolve_tensor(model, pa.full_name) for pa in group.axes}


def group_n_channels(params, group):
    first = group.axes[0]
    return int(params[first.full_name].shape[0])


def group_features_shape(params, group):
    n_channels = group_n_channels(params, group)
    feature_dim = 0
    for pa in group.axes:
        if 'running' in pa.full_name:
            continue
        tensor = params[pa.full_name]
        feature_dim += tensor.numel() // tensor.shape[pa.axis]
    return (n_channels, int(feature_dim))


def concat_group_features(params, group):
    """Konkatenacja cech w kolejności osi grupy, bez running_mean/var."""
    n_channels = group_n_channels(params, group)
    concatenated = None
    for pa in group.axes:
        if 'running' in pa.full_name:
            continue
        tensor = params[pa.full_name]
        if pa.axis == 0:
            moved = tensor
        else:
            moved = torch.movedim(tensor, pa.axis, 0)
        flat = moved.reshape(n_channels, -1)
        concatenated = flat if concatenated is None else torch.hstack((concatenated, flat))
    if concatenated is None:
        raise RuntimeError(f'Grupa {group.name} nie ma cech do klastrowania')
    return concatenated.cpu()


def hkmeans_available():
    try:
        import hkmeans  # noqa: F401
        import sklearn  # noqa: F401
    except Exception:
        return False
    return True


def cluster_group_features(features, n_clusters):
    """PCA(min(n,features)) + HKMeans(random_state=42, n_init=1, n_jobs=1)."""
    import numpy as np
    from hkmeans import HKMeans
    from sklearn.decomposition import PCA

    x = features.view(features.shape[0], -1).cpu().numpy()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    n_samples, n_features = x.shape
    x = PCA(n_components=min(n_samples, n_features)).fit_transform(x)
    clusterer = HKMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=1,
        n_jobs=1,
        max_iter=10,
        verbose=False,
    )
    clusterer.fit(x)
    labels = torch.tensor(clusterer.labels_, dtype=torch.int64)
    return labels


def validate_cluster_labels(labels, n_channels, n_clusters=None):
    """Sprawdza długość, zakres i brak pustych klastrów; zwraca liczbę klastrów."""
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels, dtype=torch.int64)
    labels = labels.detach().to(dtype=torch.int64, device='cpu').reshape(-1)
    if labels.numel() != n_channels:
        raise ValueError(
            f'Etykiety mają {labels.numel()} elementów, oczekiwano {n_channels}')
    if labels.numel() == 0:
        raise ValueError('Puste etykiety klastrów')
    if int(labels.min()) < 0:
        raise ValueError('Ujemna etykieta klastra')
    unique = torch.unique(labels, sorted=True)
    k = int(unique.numel())
    if n_clusters is not None and k != int(n_clusters):
        raise RuntimeError(
            f'Niewykorzystane klastry: {k}/{int(n_clusters)}; budżet bez zmian')
    if int(unique.max()) != k - 1 or not torch.equal(
            unique, torch.arange(k, dtype=torch.int64)):
        raise RuntimeError('Etykiety klastrów nie są ciągłe od 0')
    counts = torch.bincount(labels, minlength=k)
    if int(counts.min()) < 1:
        raise RuntimeError('Pusty klaster')
    return k


def build_merge_matrix(labels, n_clusters, n_channels, device=None):
    k = validate_cluster_labels(labels, n_channels, n_clusters)
    if device is None:
        device = labels.device
    merge = torch.zeros((k, n_channels), device=device, dtype=torch.float32)
    merge.scatter_(0, labels.to(device).unsqueeze(0), 1.0)
    counts = merge.sum(dim=1, keepdim=True)
    if bool((counts < 1).any()):
        raise RuntimeError('Pusty klaster w macierzy scalania')
    merge /= counts.clamp(min=1)
    return merge


def merge_group_params(params, group, merge):
    """Scala parametry: axis0 sumuje (D@M@p), axis1 uśrednia (M@p)."""
    scaled = torch.diag(1.0 / torch.diag(merge @ merge.T))
    merged_params = {}
    for pa in group.axes:
        tensor = params[pa.full_name].detach().clone()
        if pa.axis == 0:
            if tensor.dim() == 1:
                result = (scaled @ merge @ tensor)
            else:
                shape = tensor.shape
                result = scaled @ merge @ tensor.reshape(shape[0], -1)
                result = result.reshape(merge.shape[0], *shape[1:])
        else:
            if tensor.dim() == 2:
                tensor = tensor.permute(1, 0)
            else:
                tensor = tensor.permute(1, 0, 2, 3)
            shape = tensor.shape
            result = merge @ tensor.reshape(shape[0], -1)
            result = result.reshape(merge.shape[0], *shape[1:])
            if tensor.dim() == 2:
                result = result.permute(1, 0)
            else:
                result = result.permute(1, 0, 2, 3)
        merged_params[pa.full_name] = result
    return merged_params


def _average_bn_by_labels(old_bn, labels, n_clusters):
    new_bn = nn.BatchNorm2d(n_clusters)
    for pname in ('weight', 'bias'):
        original = getattr(old_bn, pname).data
        fused = torch.stack([
            original[labels == k].mean() if bool((labels == k).any())
            else torch.tensor(0.0)
            for k in range(n_clusters)
        ])
        setattr(new_bn, pname, nn.Parameter(fused))
    for pname in ('running_mean', 'running_var'):
        original = getattr(old_bn, pname).data
        fused = torch.stack([
            original[labels == k].mean() if bool((labels == k).any())
            else torch.tensor(0.0)
            for k in range(n_clusters)
        ])
        getattr(new_bn, pname).data.copy_(fused)
    return new_bn


def _copy_bn(old_bn, param_dict):
    new_bn = nn.BatchNorm2d(param_dict['weight'].shape[0])
    for pname in ('weight', 'bias', 'running_mean', 'running_var'):
        if pname in param_dict:
            getattr(new_bn, pname).data.copy_(param_dict[pname])
    return new_bn


def rebuild_module(old_module, param_dict, cluster_labels=None,
                   n_clusters=None, preserve_fc_bias=False, seed=42):
    """Odbudowa Conv2d/BatchNorm2d/Linear z zachowaniem konfiguracji źródła."""
    if isinstance(old_module, nn.Conv2d):
        weight = param_dict['weight']
        new_conv = nn.Conv2d(
            in_channels=weight.shape[1],
            out_channels=weight.shape[0],
            kernel_size=old_module.kernel_size,
            stride=old_module.stride,
            padding=old_module.padding,
            dilation=old_module.dilation,
            groups=old_module.groups,
            bias='bias' in param_dict,
            padding_mode=old_module.padding_mode,
        )
        new_conv.weight.data.copy_(weight)
        if 'bias' in param_dict:
            new_conv.bias.data.copy_(param_dict['bias'])
        return new_conv

    if isinstance(old_module, nn.BatchNorm2d):
        if cluster_labels is not None:
            if n_clusters is None:
                raise ValueError('Brak liczby klastrów dla BatchNorm2d')
            return _average_bn_by_labels(old_module, cluster_labels, n_clusters)
        return _copy_bn(old_module, param_dict)

    if isinstance(old_module, nn.Linear):
        weight = param_dict['weight']
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            new_fc = nn.Linear(weight.shape[1], weight.shape[0])
        new_fc.weight.data.copy_(weight)
        if 'bias' in param_dict:
            new_fc.bias.data.copy_(param_dict['bias'])
        elif preserve_fc_bias:
            if old_module.bias is None:
                new_fc.bias = None
            else:
                new_fc.bias.data.copy_(old_module.bias.data)
        return new_fc

    return copy.deepcopy(old_module)


def _replace_module(model, full_name, new_module):
    parts = full_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def _rebuild_group(model, group, merged_params, labels, n_clusters,
                   preserve_fc_bias, seed):
    param_groups = OrderedDict()
    for full_name, tensor in merged_params.items():
        module_name, pname = full_name.rsplit('.', 1)
        param_groups.setdefault(module_name, {})[pname] = tensor
    for module_name, param_dict in param_groups.items():
        old_module = get_module_by_name(model, module_name)
        new_module = rebuild_module(
            old_module,
            param_dict,
            cluster_labels=labels,
            n_clusters=n_clusters,
            preserve_fc_bias=preserve_fc_bias,
            seed=seed,
        )
        _replace_module(model, module_name, new_module)


def count_parameters(model):
    return int(sum(parameter.numel() for parameter in model.parameters()))


def state_dict_all_finite(model):
    return all(bool(torch.isfinite(tensor).all()) for tensor in model.state_dict().values())


def _fc_bias_diagnostics(old_fc, new_fc, policy):
    old_bias = old_fc.bias
    new_bias = new_fc.bias
    if old_bias is None or new_bias is None:
        return {
            'policy': policy,
            'old_present': old_bias is not None,
            'new_present': new_bias is not None,
            'equal': old_bias is None and new_bias is None,
            'old_norm': None if old_bias is None else float(old_bias.norm()),
            'new_norm': None if new_bias is None else float(new_bias.norm()),
            'max_abs_diff': None,
        }
    diff = (new_bias - old_bias).detach()
    return {
        'policy': policy,
        'old_present': True,
        'new_present': True,
        'equal': bool(torch.equal(new_bias, old_bias)),
        'old_norm': float(old_bias.detach().norm()),
        'new_norm': float(new_bias.detach().norm()),
        'max_abs_diff': float(diff.abs().max()),
    }


def _group_diagnostics(group, features_shape, labels, counts):
    return {
        'name': group.name,
        'axes': [[pa.full_name, int(pa.axis)] for pa in group.axes],
        'n_channels': int(len(labels)),
        'n_clusters': int(len(counts)),
        'features_shape': [int(features_shape[0]), int(features_shape[1])],
        'cluster_labels': [int(v) for v in labels.tolist()],
        'cluster_counts': [int(v) for v in counts.tolist()],
    }


def _base_diagnostics(model):
    return {
        'experiment': EXPERIMENT,
        'opis': OPIS_PATH,
        'reference_commit': REFERENCE_COMMIT,
        'group_order': group_order(),
        'nparams_before': count_parameters(model),
        'nparams_after': None,
        'groups': [],
        'bypassed': False,
        'all_params_finite': None,
        'fc_bias': None,
    }


def fold(model, ratio, seed=42, preserve_fc_bias=False):
    """Folding ResNet18: zwraca (new_model_na_CPU, diagnostics)."""
    ratio = float(ratio)
    if not math.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
        raise ValueError(f'ratio musi być w [0, 1], otrzymano {ratio}')

    reference_fc = model.fc
    new_model = copy.deepcopy(model).to(torch.device('cpu'))
    diagnostics = _base_diagnostics(new_model)

    if ratio == 0.0:
        diagnostics['bypassed'] = True
        diagnostics['nparams_after'] = count_parameters(new_model)
        diagnostics['all_params_finite'] = state_dict_all_finite(new_model)
        diagnostics['fc_bias'] = _fc_bias_diagnostics(
            reference_fc, new_model.fc, 'bypass_unchanged')
        return new_model, diagnostics

    keep_ratio = 1.0 - ratio
    for group in resnet18_groups():
        params = collect_group_params(new_model, group)
        n_channels = group_n_channels(params, group)
        features_shape = group_features_shape(params, group)
        n_clusters = max(int(n_channels * keep_ratio), 1)
        features = concat_group_features(params, group)
        labels = cluster_group_features(features, n_clusters)
        k = validate_cluster_labels(labels, n_channels, n_clusters)
        merge = build_merge_matrix(labels, k, n_channels)
        merged_params = merge_group_params(params, group, merge)
        _rebuild_group(
            new_model, group, merged_params, labels, k,
            preserve_fc_bias, seed)
        counts = torch.bincount(labels, minlength=k)
        diagnostics['groups'].append(
            _group_diagnostics(group, features_shape, labels, counts))

    diagnostics['nparams_after'] = count_parameters(new_model)
    diagnostics['all_params_finite'] = state_dict_all_finite(new_model)
    policy = 'preserved' if preserve_fc_bias else 'regenerated_official_style'
    diagnostics['fc_bias'] = _fc_bias_diagnostics(
        reference_fc, new_model.fc, policy)
    return new_model, diagnostics


def apply_fold_labels(model, labels_by_group, preserve_fc_bias=False, seed=42):
    """Nakłada gotowe etykiety grup (testy bez hkmeans); zwraca model, diag."""
    reference_fc = model.fc
    new_model = copy.deepcopy(model).to(torch.device('cpu'))
    diagnostics = _base_diagnostics(new_model)
    supplied = {str(name): value for name, value in dict(labels_by_group).items()}
    unknown = set(supplied) - set(group_order())
    if unknown:
        raise KeyError(f'Nieznane grupy: {sorted(unknown)}')

    for group in resnet18_groups():
        params = collect_group_params(new_model, group)
        n_channels = group_n_channels(params, group)
        features_shape = group_features_shape(params, group)
        if group.name not in supplied:
            raise KeyError(f'Brak etykiet dla grupy {group.name}')
        labels = torch.as_tensor(supplied[group.name], dtype=torch.int64).reshape(-1)
        k = validate_cluster_labels(labels, n_channels)
        merge = build_merge_matrix(labels, k, n_channels)
        merged_params = merge_group_params(params, group, merge)
        _rebuild_group(
            new_model, group, merged_params, labels, k,
            preserve_fc_bias, seed)
        counts = torch.bincount(labels, minlength=k)
        diagnostics['groups'].append(
            _group_diagnostics(group, features_shape, labels, counts))

    diagnostics['nparams_after'] = count_parameters(new_model)
    diagnostics['all_params_finite'] = state_dict_all_finite(new_model)
    policy = 'preserved' if preserve_fc_bias else 'regenerated_official_style'
    diagnostics['fc_bias'] = _fc_bias_diagnostics(
        reference_fc, new_model.fc, policy)
    return new_model, diagnostics


def build_cifar_resnet18(num_classes=10):
    """Adapter E011: torchvision ResNet18 dla 3x32x32, bez importu E011."""
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model
