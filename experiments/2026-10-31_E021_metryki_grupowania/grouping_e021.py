"""E021: grupowanie kanałów różnymi metrykami przy stałej liczbie klastrów.

Eksperyment: 2026-10-31_E021_metryki_grupowania; pełny protokół: OPIS.md.
Wejście: cechy kanałów grupy E012 i liczba klastrów k.
Wyjście: etykiety 0..k-1; scalanie i odbudowa pozostają w E012.
Zmieniamy wyłącznie metrykę/algorytm grupowania, nie budżet ani operator.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

import e017_common as base
import e021_common as common

FOLDING = None


def _folding():
    global FOLDING
    if FOLDING is None:
        FOLDING = common.folding_module()
    return FOLDING


def _stable_seed(*parts):
    text = '|'.join(str(part) for part in parts).encode('utf-8')
    return int.from_bytes(hashlib.sha256(text).digest()[:8], 'big')


def _validate(labels, k, n):
    labels = torch.as_tensor(labels, dtype=torch.int64).reshape(-1)
    if labels.numel() != n:
        raise ValueError('Zła długość etykiet')
    unique = torch.unique(labels, sorted=True)
    if int(unique.numel()) != int(k):
        raise RuntimeError(f'Niewykorzystane klastry: {unique.numel()}/{k}')
    if not torch.equal(unique, torch.arange(k, dtype=torch.int64)):
        raise RuntimeError('Etykiety nie są ciągłe od 0')
    return labels


def features(params, group, kind):
    folding = _folding()
    raw = folding.concat_group_features(params, group).to(torch.float32)
    if kind in ('official_weights', 'random_balanced', 'agg_euclidean',
                'agg_manhattan', 'agg_cosine', 'agg_correlation',
                'ward_euclidean'):
        return raw
    if kind == 'standardized':
        mean = raw.mean(dim=0, keepdim=True)
        std = raw.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-12)
        return (raw - mean) / std
    raise ValueError(f'Nieznany wariant: {kind}')


def _hkmeans(features_tensor, k):
    return common.e017_grouping()._label_from_clustering(
        features_tensor, k, features_tensor.shape[0])


def _agglomerative(features_tensor, k, metric, linkage):
    x = features_tensor.cpu().numpy()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    clusterer = AgglomerativeClustering(
        n_clusters=int(k), metric=metric, linkage=linkage)
    labels = clusterer.fit_predict(x)
    return torch.as_tensor(labels, dtype=torch.int64)


def labels_for(params, group, kind, k, context):
    n_channels = int(_folding().group_n_channels(params, group))
    x = features(params, group, kind)
    if int(k) >= n_channels:
        labels = torch.arange(n_channels, dtype=torch.int64)
    elif kind in ('official_weights', 'random_balanced'):
        labels = _hkmeans(x, k)
    elif kind == 'standardized':
        labels = _hkmeans(x, k)
    elif kind == 'agg_euclidean':
        labels = _agglomerative(x, k, 'euclidean', 'average')
    elif kind == 'agg_manhattan':
        labels = _agglomerative(x, k, 'manhattan', 'average')
    elif kind == 'agg_cosine':
        labels = _agglomerative(x, k, 'cosine', 'average')
    elif kind == 'agg_correlation':
        labels = _agglomerative(x, k, 'correlation', 'average')
    elif kind == 'ward_euclidean':
        labels = _agglomerative(x, k, 'euclidean', 'ward')
    else:
        raise ValueError(f'Nieznany wariant: {kind}')
    labels = _validate(labels, k, n_channels)
    if kind == 'random_balanced':
        reference = labels.clone()
        generator = np.random.default_rng(_stable_seed(
            context['dataset'], context['budget'], group.name))
        permutation = generator.permutation(n_channels)
        labels = torch.empty(n_channels, dtype=torch.int64)
        labels[permutation] = reference
        labels = _validate(labels, k, n_channels)
    counts = torch.bincount(labels, minlength=int(k)).tolist()
    return labels, {
        'group': group.name,
        'kind': kind,
        'n_channels': n_channels,
        'n_clusters': int(k),
        'cluster_counts': [int(v) for v in counts],
        'label_digest': base.labels_digest(labels),
    }
