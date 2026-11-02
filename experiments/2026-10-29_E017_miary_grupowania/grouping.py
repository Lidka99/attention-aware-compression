"""E017: miary podobieństwa kanałów i deterministyczne przypisanie do grup.

Eksperyment: 2026-10-29_E017_miary_grupowania; pełny protokół: OPIS.md.
Wejście: cechy kanałów grupy E012 (konkatenacja wag) i liczba klastrów k.
Wyjście: etykiety klastrów 0..k-1; ten sam operator scalania E012.

Zmieniamy wyłącznie reprezentację lub metrykę używaną do grupowania.
Liczba klastrów, scalanie i korekta BN pozostają bez zmian.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

import e017_common as common


def _require_k(labels, k, n_channels):
    labels = torch.as_tensor(labels, dtype=torch.int64).reshape(-1)
    if labels.numel() != n_channels:
        raise ValueError(f'Etykiety {labels.numel()} != {n_channels}')
    unique = torch.unique(labels, sorted=True)
    if int(unique.numel()) != int(k):
        raise RuntimeError(f'Niewykorzystane klastry: {unique.numel()}/{k}')
    if not torch.equal(unique, torch.arange(k, dtype=torch.int64)):
        raise RuntimeError('Etykiety nie są ciągłe od 0')
    if int(torch.bincount(labels, minlength=k).min()) < 1:
        raise RuntimeError('Pusty klaster')
    return labels


def _cluster_euclidean(features, k):
    """PCA z pełną liczbą składowych + HKMeans, jak w E012."""
    import numpy as np
    from hkmeans import HKMeans
    from sklearn.decomposition import PCA

    x = features.view(features.shape[0], -1).cpu().numpy()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    n_samples, n_features = x.shape
    x = PCA(n_components=min(n_samples, n_features)).fit_transform(x)
    clusterer = HKMeans(
        n_clusters=int(k), random_state=42, n_init=1, n_jobs=1,
        max_iter=10, verbose=False)
    clusterer.fit(x)
    return torch.tensor(clusterer.labels_, dtype=torch.int64)


def _split_largest_cluster(labels, k, features):
    """Deterministyczny fallback: dzieli największy klaster najdalszą połową."""
    labels = labels.clone()
    x = features.detach().to(torch.float64)
    while int(torch.unique(labels).numel()) < int(k):
        counts = torch.bincount(labels)
        largest = int(torch.argmax(counts).item())
        members = torch.nonzero(labels == largest, as_tuple=False).reshape(-1)
        if members.numel() < 2:
            raise RuntimeError('Nie można rozdzielić klastra')
        points = x[members]
        center = points.mean(dim=0)
        distance = ((points - center) ** 2).sum(dim=1)
        order = torch.argsort(distance, descending=True)
        take = members[order[: max(1, members.numel() // 2)]]
        new_label = int(labels.max().item()) + 1
        labels[take] = new_label
    return labels


def _label_from_clustering(features, k, n_channels):
    labels = _cluster_euclidean(features, k)
    if int(torch.unique(labels).numel()) < int(k):
        labels = _split_largest_cluster(labels, k, features)
    return _require_k(labels, k, n_channels)


def _bn_prefixes(params):
    prefixes = set()
    for name in params:
        if name.endswith('.running_var'):
            prefixes.add(name[: -len('.running_var')])
    return prefixes


def bn_effective_features(params, group):
    """Skaluje wektory kanałów średnią geometryczną skal BN z grupy.

    Grupa może zawierać kilka modułów BN o tej samej liczbie kanałów
    (np. ``bn1`` i ``bn2`` bloków wejściowych). Dla kanału c używamy
    ``exp(mean_p log(|gamma|/sqrt(var+eps)))``; brak BN oznacza cechy surowe.
    """
    features = common.folding_module().concat_group_features(params, group)
    prefixes = sorted(_bn_prefixes(params))
    if not prefixes:
        return features, False
    scales = []
    for prefix in prefixes:
        gamma = params[f'{prefix}.weight'].detach().to(torch.float32)
        variance = params[f'{prefix}.running_var'].detach().to(torch.float32)
        value = gamma.abs() / torch.sqrt(variance + 1e-5)
        value = torch.nan_to_num(value, nan=1.0, posinf=1.0, neginf=1.0)
        scales.append(value.clamp_min(1e-6))
    scale = torch.stack(scales, dim=0).log().mean(dim=0).exp()
    return features * scale.reshape(-1, 1), True


def features_for(params, group, kind):
    """Buduje cechy kanałów dla wybranego wariantu grupowania."""
    folding = common.folding_module()
    if kind in ('official_weights', 'random_balanced'):
        return folding.concat_group_features(params, group), False
    if kind == 'cosine_weights':
        features = folding.concat_group_features(params, group).to(torch.float32)
        norm = features.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return features / norm, False
    if kind == 'standardized_features':
        features = folding.concat_group_features(params, group).to(torch.float32)
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-12)
        return (features - mean) / std, False
    if kind == 'bn_effective_weights':
        return bn_effective_features(params, group)
    raise ValueError(f'Nieznany wariant: {kind}')


def _stable_seed(*parts):
    text = '|'.join(str(part) for part in parts).encode('utf-8')
    return int.from_bytes(hashlib.sha256(text).digest()[:8], 'big')


def labels_for(params, group, kind, k, context):
    """Zwraca etykiety kanałów dla jednej grupy i jednego wariantu."""
    n_channels = int(common.folding_module().group_n_channels(params, group))
    features, used_bn = features_for(params, group, kind)
    if features.shape[0] != n_channels:
        raise RuntimeError('Zła liczba kanałów w cechach')
    if kind == 'random_balanced':
        reference = _label_from_clustering(features, k, n_channels)
        generator = np.random.default_rng(
            _stable_seed(context['dataset'], context['budget'],
                         context['replicate'], group.name))
        permutation = generator.permutation(n_channels)
        labels = torch.empty(n_channels, dtype=torch.int64)
        labels[permutation] = reference
        labels = _require_k(labels, k, n_channels)
    else:
        labels = _label_from_clustering(features, k, n_channels)
    return labels, {
        'group': group.name,
        'kind': kind,
        'n_channels': n_channels,
        'n_clusters': int(k),
        'features_dim': int(features.shape[1]),
        'used_bn_scale': bool(used_bn),
        'label_digest': common.labels_digest(labels),
        'cluster_counts': [int(v) for v in torch.bincount(labels).tolist()],
    }
