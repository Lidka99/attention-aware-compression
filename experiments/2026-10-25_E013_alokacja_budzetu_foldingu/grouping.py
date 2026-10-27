"""E013: miary grupowania kanałów przy wspólnym operatorze foldingu E012.

Eksperyment: 2026-10-25_E013_alokacja_budzetu_foldingu; pełny opis: OPIS.md.
Wejście: macierz cech ``torch.Tensor[n, d]`` zbudowana jak w E012
(konkatenacja powiązanych parametrów grupy, bez running stats) oraz liczba
klastrów ``n_clusters``. Wyjście: etykiety ``torch.int64`` na CPU, dokładnie
``n_clusters`` niepustych klastrów, ciągłe od 0.

Moduł udostępnia cztery odległości (L2, L1, cosine, correlation) oraz pięć
metod grupowania:

- ``hkmeans_l2``: deleguje do oryginalnego
  ``folding.cluster_group_features`` z E012 (PCA pełny + HKMeans/L2),
- ``average_l2``, ``average_l1``: aglomeracja average na surowych cechach,
  bez dodatkowego PCA; pełny PCA E012 zachowuje odległość euklidesową,
- ``average_cosine``: cosine na surowych wierszach, bez wartości bezwzględnej,
  więc przeciwne kierunki nie są traktowane jako podobne,
- ``average_correlation``: centrowanie każdego wiersza, następnie cosine.

Zerowe wiersze w cosine/correlation: dwa zerowe wiersze mają odległość 0,
a para zerowy-niezerowy ma odległość 1. Dla L1/L2 obowiązuje zwykła definicja
matematyczna (zero-zero = 0, zero-niezero = norma drugiego wiersza). Wartości
NaN/Inf wejścia są zastępowane zerem, jak w E012, aby macierz odległości była
skończona. Uwaga metodologiczna: na sferze jednostkowej porządek cosine jest
monotonicznym przekształceniem L2 (``d_cos = d_l2**2 / 2``); nie jest to
niezależna nowość, tylko własność geometryczna.

``bind_method`` pozwala lokalnie podmienić ``cluster_group_features`` w module
foldingu E012 (np. w jednym workerze), z przywróceniem oryginalnego callable.
Dla ``hkmeans_l2`` opakowanie wywołuje zapisany oryginał, nie samo siebie.
"""

from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform

EXPERIMENT = '2026-10-25_E013_alokacja_budzetu_foldingu'
OPIS_PATH = 'experiments/2026-10-25_E013_alokacja_budzetu_foldingu/OPIS.md'
E012_EXPERIMENT = '2026-10-24_E012_folding_referencja'
E012_FOLDING_PATH = (
    Path(__file__).resolve().parents[1] / E012_EXPERIMENT / 'folding.py'
)

GROUPING_METHODS = (
    'hkmeans_l2',
    'average_l2',
    'average_l1',
    'average_cosine',
    'average_correlation',
)
DISTANCE_METRICS = ('l2', 'l1', 'cosine', 'correlation')
_METHOD_TO_METRIC = {
    'hkmeans_l2': 'l2',
    'average_l2': 'l2',
    'average_l1': 'l1',
    'average_cosine': 'cosine',
    'average_correlation': 'correlation',
}


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


E012_FOLDING = _load_module(E012_FOLDING_PATH, '_e013_e012_folding')
E012_CLUSTER_GROUP_FEATURES = E012_FOLDING.cluster_group_features


def hkmeans_available():
    """Zwraca True, gdy E012 może zaimportować hkmeans i sklearn."""
    return bool(E012_FOLDING.hkmeans_available())


def _require_method(method):
    if method not in GROUPING_METHODS:
        raise ValueError(
            f'Nieznana metoda grupowania {method!r}; '
            f'dopuszczalne: {GROUPING_METHODS}')
    return method


def _resolve_metric(name):
    if name in DISTANCE_METRICS:
        return name
    try:
        return _METHOD_TO_METRIC[name]
    except (KeyError, TypeError):
        raise ValueError(
            f'Nieznana miara {name!r}; dopuszczalne metryki: '
            f'{DISTANCE_METRICS} lub metody: {GROUPING_METHODS}') from None


def _as_feature_matrix(features):
    if not torch.is_tensor(features):
        features = torch.as_tensor(features)
    x = features.detach().to(device='cpu', dtype=torch.float64)
    if x.dim() < 2:
        raise ValueError(
            f'features musi mieć kształt [n, d], otrzymano {tuple(x.shape)}')
    x = x.reshape(x.shape[0], -1).contiguous()
    if not bool(torch.isfinite(x).all()):
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def _cosine_pairwise(rows):
    """Cosine 1 - cos dla wierszy, z jawną obsługą zerowej normy."""
    rows = np.asarray(rows, dtype=np.float64)
    n, d = rows.shape
    distances = np.zeros((n, n), dtype=np.float64)
    if n <= 1 or d == 0:
        return distances

    norms = np.linalg.norm(rows, axis=1)
    zero = norms == 0.0
    nonzero = ~zero
    unit = np.zeros_like(rows)
    if bool(nonzero.any()):
        unit[nonzero] = rows[nonzero] / norms[nonzero, None]

    if bool(nonzero.any()):
        safe = unit.copy()
        if bool(zero.any()):
            safe[zero] = unit[int(np.flatnonzero(nonzero)[0])]
        with np.errstate(invalid='ignore', divide='ignore'):
            distances = squareform(pdist(safe, metric='cosine'))

    if bool(zero.any()):
        z = zero
        distances[np.ix_(z, z)] = 0.0
        distances[np.ix_(z, ~z)] = 1.0
        distances[np.ix_(~z, z)] = 1.0

    np.clip(distances, 0.0, 2.0, out=distances)
    np.fill_diagonal(distances, 0.0)
    return distances


def _finalize_distance(matrix):
    distances = np.array(matrix, dtype=np.float64, copy=True)
    if not bool(np.isfinite(distances).all()):
        raise RuntimeError('Macierz odległości zawiera wartości nieskończone')
    distances = np.maximum(distances, 0.0)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, 0.0)
    if not bool(np.isfinite(distances).all()):
        raise RuntimeError('Macierz odległości zawiera wartości nieskończone')
    return torch.from_numpy(distances)


def distance_matrix(features, method):
    """Zwraca macierz odległości ``[n, n]`` jako ``torch.float64`` na CPU.

    Parametry
    ---------
    features : torch.Tensor albo array-like, kształt ``[n, d]``
        Surowe cechy grupy. Większe wymiary są spłaszczane do ``[n, d]``.
    method : str
        ``'l2'``, ``'l1'``, ``'cosine'``, ``'correlation'`` albo nazwa metody
        grupowania z ``GROUPING_METHODS`` (np. ``'average_l1'``).

    Zwraca
    ------
    torch.Tensor
        Symetryczna macierz ``float64`` na CPU, nieujemna, z zerową
        przekątną i skończonymi wartościami. Cosine/correlation należą do
        ``[0, 2]``; L1/L2 są nieograniczone z góry.
    """
    metric = _resolve_metric(method)
    x = _as_feature_matrix(features)
    n = int(x.shape[0])
    if n <= 1 or int(x.shape[1]) == 0:
        return torch.zeros((n, n), dtype=torch.float64)

    values = x.numpy()
    if metric == 'l2':
        distances = squareform(pdist(values, metric='euclidean'))
    elif metric == 'l1':
        distances = squareform(pdist(values, metric='cityblock'))
    elif metric == 'cosine':
        distances = _cosine_pairwise(values)
    else:
        centered = x - x.mean(dim=1, keepdim=True)
        distances = _cosine_pairwise(centered.numpy())
    return _finalize_distance(distances)


def _agglomerative_labels(distances, n_clusters):
    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(
        n_clusters=int(n_clusters),
        metric='precomputed',
        linkage='average',
    )
    labels = model.fit_predict(np.asarray(distances, dtype=np.float64))
    return torch.as_tensor(labels, dtype=torch.int64)


def _validate_labels(labels, n_samples, n_clusters):
    labels = torch.as_tensor(
        labels, dtype=torch.int64, device='cpu').reshape(-1)
    if int(labels.numel()) != int(n_samples):
        raise RuntimeError(
            f'Etykiety mają {labels.numel()} elementów, '
            f'oczekiwano {n_samples}')
    unique = torch.unique(labels, sorted=True)
    expected = torch.arange(int(n_clusters), dtype=torch.int64)
    if not torch.equal(unique, expected):
        raise RuntimeError(
            f'Etykiety nie tworzą dokładnie {n_clusters} klastrów 0..k-1: '
            f'{unique.tolist()}')
    counts = torch.bincount(labels, minlength=int(n_clusters))
    if int(counts.numel()) != int(n_clusters) or bool((counts < 1).any()):
        raise RuntimeError('Pusty klaster po grupowaniu')
    return labels


def cluster(features, n_clusters, method):
    """Grupuje cechy wybraną metodą i zwraca etykiety ``torch.int64`` na CPU.

    Parametry
    ---------
    features : torch.Tensor, kształt ``[n, d]``
        Cechy pojedynczej grupy, takie jak w E012.
    n_clusters : int
        Żądana liczba klastrów ``1 <= k <= n``.
    method : str
        Jedna z ``GROUPING_METHODS``. ``hkmeans_l2`` wywołuje oryginalne
        ``E012_FOLDING.cluster_group_features``; pozostałe metody używają
        aglomeracji average na macierzy odległości ``distance_matrix``.

    Zwraca
    ------
    torch.Tensor
        Etykiety ``int64`` na CPU, długość ``n``, dokładnie ``k`` niepustych
        klastrów i wartości ciągłe ``0..k-1``. Dla ``k == n`` zwracane jest
        ``arange(n)`` (każdy wiersz osobnym klastrem).
    """
    _require_method(method)
    if not torch.is_tensor(features):
        features = torch.as_tensor(features)
    if features.dim() < 2:
        raise ValueError(
            f'features musi mieć kształt [n, d], otrzymano '
            f'{tuple(features.shape)}')
    n_samples = int(features.shape[0])
    if n_samples < 1:
        raise ValueError('Grupowanie wymaga co najmniej jednego wiersza')
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, (int, np.integer)):
        raise TypeError('n_clusters musi być liczbą całkowitą')
    n_clusters = int(n_clusters)
    if n_clusters < 1 or n_clusters > n_samples:
        raise ValueError(
            f'n_clusters musi spełniać 1 <= k <= n, otrzymano '
            f'{n_clusters} dla n={n_samples}')

    if n_clusters == n_samples:
        return torch.arange(n_samples, dtype=torch.int64)

    if method == 'hkmeans_l2':
        labels = E012_CLUSTER_GROUP_FEATURES(
            features.detach().to(device='cpu'), n_clusters)
    else:
        distances = distance_matrix(features, method)
        labels = _agglomerative_labels(distances, n_clusters)
    return _validate_labels(labels, n_samples, n_clusters)


@contextmanager
def bind_method(folding_module, method) -> Iterator[object]:
    """Lokalnie wiąże ``folding_module.cluster_group_features`` z metodą.

    Parametry
    ---------
    folding_module : module
        Moduł E012 (``folding``) albo zgodny obiekt mający atrybut
        ``cluster_group_features``.
    method : str
        Metoda z ``GROUPING_METHODS``.

    Działanie
    ---------
    Dla ``hkmeans_l2`` opakowanie wywołuje oryginalny callable zapisany przed
    podmianą (brak rekurencji). Dla metod average wywoływane jest
    ``cluster(features, n_clusters, method)``. Po wyjściu z kontekstu
    przywracany jest dokładnie ten sam obiekt funkcji. Kontekst jest lokalny
    dla procesu/werkera i nie zmienia innych modułów.
    """
    _require_method(method)
    if not hasattr(folding_module, 'cluster_group_features'):
        raise TypeError(
            'folding_module musi mieć atrybut cluster_group_features')
    original = folding_module.cluster_group_features

    if method == 'hkmeans_l2':
        def bound(features, n_clusters):
            return original(features, n_clusters)
    else:
        def bound(features, n_clusters):
            return cluster(features, n_clusters, method)

    folding_module.cluster_group_features = bound
    try:
        yield folding_module
    finally:
        folding_module.cluster_group_features = original


def use_method(method):
    """Context manager wiążący metodę z domyślnym modułem E012.

    Równoważne ``bind_method(E012_FOLDING, method)``. Przykład::

        with use_method('average_cosine'):
            folded, diagnostics = E012_FOLDING.fold(model, ratio=0.3)
    """
    return bind_method(E012_FOLDING, method)


__all__ = [
    'EXPERIMENT',
    'OPIS_PATH',
    'GROUPING_METHODS',
    'DISTANCE_METRICS',
    'distance_matrix',
    'cluster',
    'bind_method',
    'use_method',
    'hkmeans_available',
]
