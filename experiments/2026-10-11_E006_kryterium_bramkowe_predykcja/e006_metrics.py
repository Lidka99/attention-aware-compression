"""E006: metryki, sygnaly bramkowe i folding pary kanalow (czyste CPU).

Eksperyment: **2026-10-11_E006_kryterium_bramkowe_predykcja** (protokol v0.2).
Pelny opis zakresu, par, sygnalow i kryterium oceny: ``OPIS.md`` w tym samym
katalogu eksperymentu (link lokalny: ``OPIS.md``). Wzor operatora foldingu
(alpha do SE fc1, beta do projekcji) pochodzi z
``../../2026-10-09_E003_kompresja_10proc_parametrow/folding.py``; tutaj jest
odtworzony samodzielnie dla pojedynczego bloku MBConv, bez importu z E003.

Co robi
-------
Modul daje deterministyczne, czyste funkcje CPU do:

1. policzenia cosinusow wszystkich par kanalow ``c < d`` na globalnie
   sklejonych wektorach kanalow (``all_pair_cosines``);
2. wyboru rozlacznych par rozlozonych po kubelkach kwantylowych
   (rownej liczności) ``s_post`` (``sample_disjoint_pairs``);
3. dopasowania wspolczynnikow scalenia ``alpha``, ``beta`` (``fit_alpha_beta``);
4. policzenia sygnalow par: pierwszorzednego ``g_pert`` i wtornych
   ``s_post``, ``s_pre``, ``g_corr``, ``g_lstd``, ``g_flip``, ``g_amp``
   (``pair_signals``);
5. zbudowania skompresowanej kopii bloku MBConv (``fold_block_for_pair``);
6. porownania wyjsc bloku i bloku zlozonego (``block_outputs``,
   ``relative_mse``);
7. statystyk rangowych i bootstrapu (``spearman``, ``partial_spearman``,
   ``bootstrap_spearman_diff``).

Wejscia
-------
Tablice ``np.ndarray`` FP32/FP64 (statystyki liczone w ``float64``);

- ``maps_a``: ``(N, C, H, W)`` mapy kanalow na zbiorze A;
- ``h_a``, ``z_a``: ``(N, C, H, W)`` aktywacje odpowiednio przed i po SE
  (``z = a * h``);
- ``a_a``: ``(N, C)`` albo ``(N, C, 1, 1)`` bramki SE (wagi atencji);
- ``g_a``: ``(N, C)`` wektor ``avgpool(h)`` podawany do fc1;
- ``fc1_weight``: ``(C_sq, C, 1, 1)`` albo ``(C_sq, C)`` wagi fc1.

Zadna funkcja nie modyfikuje wejsc i wszystkie sa deterministyczne. Folding
przyjmuje modul ``torchvision.models.efficientnet.MBConv`` i zwraca jego
``deepcopy`` (oryginal pozostaje niezmieniony).

Wyjscia
-------
``all_pair_cosines`` zwraca ``(ci, di, s_post)``; ``sample_disjoint_pairs``
zwraca ``(K, 2)`` int; ``fit_alpha_beta`` zwraca ``(alpha, beta)``;
``pair_signals`` zwraca slownik sygnalow; ``fold_block_for_pair`` zwraca nowy
modul; ``block_outputs`` zwraca ``(branch, out)``; ``relative_mse`` zwraca
``float`` albo ``None``; ``spearman``/``partial_spearman`` zwracaja ``float``
(``NaN`` przy degeneracji); ``bootstrap_spearman_diff`` zwraca slownik
``{delta, ci95, ci95_low, ci95_high, reps, valid, invalid, seed}``.

Uruchomienie kontroli: ``python -B e006_metrics.py`` (CPU) wyprintuje
``E006_METRICS_SELF_TEST=passed`` oraz liczby; nie zapisuje zadnych plikow.
Uklad: biblioteka czystych funkcji + ``self_test`` na malych danych
syntetycznych i prawdziwym bloku ``features.4.1`` EfficientNet-B0.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn

EXPERIMENT_ID = "2026-10-11_E006_kryterium_bramkowe_predykcja"
OPIS_PATH = "OPIS.md"
FOLDING_SOURCE_PATH = (
    "../../2026-10-09_E003_kompresja_10proc_parametrow/folding.py"
)
NORM_ZERO_THRESHOLD = 1e-12

__all__ = [
    "EXPERIMENT_ID",
    "OPIS_PATH",
    "FOLDING_SOURCE_PATH",
    "all_pair_cosines",
    "sample_disjoint_pairs",
    "fit_alpha_beta",
    "pair_signals",
    "fold_block_for_pair",
    "block_outputs",
    "relative_mse",
    "spearman",
    "partial_spearman",
    "bootstrap_spearman_diff",
    "self_test",
]


# --------------------------------------------------------------------------- #
# Pomocnicze: konwersje i wektory kanalow
# --------------------------------------------------------------------------- #
def _as_float64(x, name="input"):
    """Zwraca kopie tablicy w ``float64``; akceptuje ndarray i tensory CPU."""
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    else:
        arr = arr.astype(np.float64, copy=True)
    return arr


def _flatten_channel_major(maps):
    """``(N, C, H, W) -> (C, N*H*W)`` (kolejnosc: kanal, obraz, wiersz, kolumna)."""
    return maps.transpose(1, 0, 2, 3).reshape(maps.shape[1], -1)


def _gate_2d(a):
    """Akceptuje ``(N, C)`` lub ``(N, C, 1, 1)``; zwraca ``(N, C)`` bez mutacji."""
    if a.ndim == 4:
        if a.shape[2] != 1 or a.shape[3] != 1:
            raise ValueError(f"a: dozwolone (N, C) lub (N, C, 1, 1), jest {a.shape}")
        return a[:, :, 0, 0]
    if a.ndim == 2:
        return a
    raise ValueError(f"a: dozwolone (N, C) lub (N, C, 1, 1), jest {a.shape}")


def _cosine_pair_1d(x, y):
    """Cosinus dwoch wektorow 1-D; ``NaN`` gdy ktoras norma ``<= 1e-12``."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    nx = float(np.sqrt(np.dot(x, x)))
    ny = float(np.sqrt(np.dot(y, y)))
    if nx <= NORM_ZERO_THRESHOLD or ny <= NORM_ZERO_THRESHOLD:
        return float("nan")
    return float(np.dot(x, y) / (nx * ny))


# --------------------------------------------------------------------------- #
# 1) Cosinusy wszystkich par (gorny trojkat)
# --------------------------------------------------------------------------- #
def all_pair_cosines(maps_a):
    """Cosinusy wszystkich par ``c < d`` na globalnych wektorach kanalow.

    ``maps_a``: ``(N, C, H, W)``. Wektor kanalu to
    ``transpose(1, 0, 2, 3).reshape(C, -1)`` (kanal sklejony przez wszystkie
    obrazy i pozycje przestrzenne). Zwraca ``(ci, di, s_post)`` w porzadku
    leksykograficznym gornego trojkata; cosinus przyciety do ``[-1, 1]``,
    a ``NaN`` tam, gdzie norma ktoregokolwiek kanalu ``<= 1e-12``.
    """
    maps = _as_float64(maps_a, "maps_a")
    if maps.ndim != 4:
        raise ValueError(f"maps_a: oczekiwano (N, C, H, W), jest {maps.shape}")
    _, channels, _, _ = maps.shape
    flat = _flatten_channel_major(maps)
    norms = np.sqrt(np.einsum("cl,cl->c", flat, flat))
    ci, di = np.triu_indices(channels, k=1)
    ci = ci.astype(np.int64)
    di = di.astype(np.int64)
    dot = np.einsum("cl,cl->c", flat[ci], flat[di])
    denom = norms[ci] * norms[di]
    valid = (norms[ci] > NORM_ZERO_THRESHOLD) & (norms[di] > NORM_ZERO_THRESHOLD)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = dot / denom
    cos = np.where(valid, np.clip(cos, -1.0, 1.0), np.nan)
    return ci, di, cos.astype(np.float64)


# --------------------------------------------------------------------------- #
# 2) Rozlaczne pary po kubelkach s_post
# --------------------------------------------------------------------------- #
def sample_disjoint_pairs(ci, di, s_post, bins=5, per_bin=30, seed=20261012,
                          binning="quantile"):
    """Wybor par rozlacznych kanalowo z kubelkow kwantylowych po ``s_post``.

    Podzial na ``bins`` kubelkow **rownej liczności** (kwantyle = rangi):
    pary sa sortowane po ``s_post`` i dzielone na rowne czesci. Wersja 1.0
    uzywala kubelkow rownych szerokosci i nie mogla zebrac ``per_bin`` par
    rozlacznych w skrajnych kubelkach, bo rozklad ``s_post`` jest silnie
    skosny (nieudany bieg E006 1.0). W kazdym kubelku kandydaci sa tasowani
    RNG z ``seed`` i przyjmowani zachlannie, o ile zaden z kanalow nie jest
    juz uzyty (globalny zbior ``used``). Pary ``s_post = NaN`` sa pomijane.
    Wynik (``K, 2``, ``int64``) jest posortowany po ``(c, d)``. Gdy w
    jakimkolwiek kubelku nie udalo sie zebrac ``per_bin`` par, podnoszony
    jest ``ValueError`` z licznikami. ``binning``: obslugiwane tylko
    ``"quantile"``.
    """
    ci = np.asarray(ci, dtype=np.int64).ravel()
    di = np.asarray(di, dtype=np.int64).ravel()
    s = np.asarray(s_post, dtype=np.float64).ravel()
    if not (ci.shape == di.shape == s.shape):
        raise ValueError("ci, di, s_post: niezgodne dlugosci")
    if ci.size == 0:
        raise ValueError("ci, di, s_post: puste wejscie")
    bins = int(bins)
    per_bin = int(per_bin)
    if bins < 1 or per_bin < 1:
        raise ValueError("bins i per_bin musza byc >= 1")
    if binning != "quantile":
        raise ValueError('binning: obslugiwane tylko "quantile", jest %r' % (binning,))

    finite = np.isfinite(s)
    n_finite = int(np.count_nonzero(finite))
    if n_finite == 0:
        raise ValueError("s_post: brak skonczonych wartosci do kubelkowania")
    finite_idx = np.flatnonzero(finite)
    sorted_idx = finite_idx[np.argsort(s[finite_idx], kind="mergesort")]
    ranks = np.empty(n_finite, dtype=np.int64)
    ranks[sorted_idx] = np.arange(n_finite, dtype=np.int64)
    bin_of = np.full(ci.size, -1, dtype=np.int64)
    bin_of[finite_idx] = np.minimum(ranks * bins // n_finite, bins - 1)

    order = np.lexsort((di, ci))
    rng = np.random.default_rng(seed)
    used = set()
    selected = []
    achieved = [0] * bins
    for b in range(bins):
        members = order[bin_of[order] == b]
        perm = rng.permutation(members.size)
        for k in perm:
            idx = int(members[k])
            c = int(ci[idx])
            d = int(di[idx])
            if c in used or d in used:
                continue
            used.add(c)
            used.add(d)
            selected.append((c, d))
            achieved[b] += 1
            if achieved[b] >= per_bin:
                break

    short = [b for b in range(bins) if achieved[b] < per_bin]
    if short:
        raise ValueError(
            "sample_disjoint_pairs: niedobor par w kubelkach "
            f"{short}; osiagnieto {achieved} przy per_bin={per_bin}, "
            f"bins={bins}, seed={seed}, binning={binning}"
        )
    selected.sort(key=lambda pair: (pair[0], pair[1]))
    return np.asarray(selected, dtype=np.int64)


# --------------------------------------------------------------------------- #
# 3) Dopasowanie alpha/beta
# --------------------------------------------------------------------------- #
def fit_alpha_beta(h_a, g_a, z_a, c, d):
    """Wspolczynniki scalenia: ``alpha`` z ``avgpool(h)``, ``beta`` z pelnych map ``z``.

    ``alpha = <g_c, g_d> / <g_c, g_c>`` oraz ``beta = <z_c, z_d> / <z_c, z_c>``
    (sumy po obrazach i pozycjach), w ``float64``, bez clippingu. Gdy ``g_a``
    jest ``None``, ``g = avgpool(h_a)`` liczone z ``h_a``. Zwraca
    ``(alpha, beta)`` jako ``float``.
    """
    z = _as_float64(z_a, "z_a")
    if z.ndim != 4:
        raise ValueError(f"z_a: oczekiwano (N, C, H, W), jest {z.shape}")
    n, channels = z.shape[0], z.shape[1]
    c = int(c)
    d = int(d)
    if not (0 <= c < d < channels):
        raise ValueError(f"para ({c}, {d}) poza zakresem 0 <= c < d < {channels}")

    if g_a is None:
        h = _as_float64(h_a, "h_a")
        if h.ndim != 4:
            raise ValueError(f"h_a: oczekiwano (N, C, H, W), jest {h.shape}")
        g = h.mean(axis=(2, 3))
    else:
        g = _as_float64(g_a, "g_a")
        if g.ndim == 4:
            g = g.reshape(g.shape[0], g.shape[1])
        if g.ndim != 2:
            raise ValueError(f"g_a: oczekiwano (N, C), jest {g.shape}")
    if g.shape != (n, channels):
        raise ValueError(f"g_a: ksztalt {g.shape} niezgodny z (N, C)=({n}, {channels})")

    gc = g[:, c]
    gd = g[:, d]
    zc = z[:, c].reshape(-1)
    zd = z[:, d].reshape(-1)
    alpha = float(np.dot(gc, gd) / np.dot(gc, gc))
    beta = float(np.dot(zc, zd) / np.dot(zc, zc))
    return alpha, beta


# --------------------------------------------------------------------------- #
# 4) Sygnaly pary
# --------------------------------------------------------------------------- #
def pair_signals(h_a, z_a, a_a, g_a, fc1_weight, c, d):
    """Sygnaly pary ``(c, d)``: pierwszorzedny ``g_pert`` i wtorne.

    Wejscia: ``h_a`` ``(N, C, H, W)`` przed SE; ``z_a`` ``(N, C, H, W)`` po SE;
    ``a_a`` ``(N, C)`` lub ``(N, C, 1, 1)`` bramki; ``g_a`` ``(N, C)``
    ``avgpool(h)``; ``fc1_weight`` ``(C_sq, C, 1, 1)`` lub ``(C_sq, C)``.

    Zwraca ``dict`` z kluczami:

    - ``alpha``, ``beta`` — z :func:`fit_alpha_beta`;
    - ``s_post``, ``s_pre`` — cosinus globalnych wektorow odpowiednio ``z`` i
      ``h`` (ta sama linearyzacja co :func:`all_pair_cosines`);
    - ``g_corr`` — cosinus profili bramki ``a_c``, ``a_d`` po obrazach;
    - ``g_lstd`` — odchylenie ``log(clip(a_c)) - log(clip(a_d))`` po obrazach
      (``clip`` dol: ``1e-12``), plus ``g_lstd_clip_count`` (liczba przycięc);
    - ``g_flip`` — czestosc zmiany znaku ``a_c - a_d`` po obrazach;
    - ``g_amp`` — cosinus profili ``h_c``, ``h_d`` wazony amplituda bramki:
      ``cos(h_c, h_d) * min(mean(a_c), mean(a_d))`` (prosta stala regula);
    - ``g_pert`` — srednia po obrazach ``||W1[:, d] (alpha g_c - g_d)|| /
      ||W1 g||``; ``NaN`` gdy mianownik ``<= 1e-12`` (srednia po poprawnych).
    """
    h = _as_float64(h_a, "h_a")
    z = _as_float64(z_a, "z_a")
    a = _gate_2d(_as_float64(a_a, "a_a"))
    g = _as_float64(g_a, "g_a")
    if g.ndim == 4:
        g = g.reshape(g.shape[0], g.shape[1])
    w1 = _as_float64(fc1_weight, "fc1_weight")
    if w1.ndim == 4:
        w1 = w1.reshape(w1.shape[0], w1.shape[1])
    if h.ndim != 4 or z.ndim != 4:
        raise ValueError("h_a i z_a musza byc (N, C, H, W)")
    n, channels = h.shape[0], h.shape[1]
    c = int(c)
    d = int(d)
    if not (0 <= c < d < channels):
        raise ValueError(f"para ({c}, {d}) poza zakresem 0 <= c < d < {channels}")
    if a.shape != (n, channels):
        raise ValueError(f"a_a: ksztalt {a.shape} niezgodny z (N, C)=({n}, {channels})")

    alpha, beta = fit_alpha_beta(h, g, z, c, d)

    s_pre = _cosine_pair_1d(_flatten_channel_major(h)[c], _flatten_channel_major(h)[d])
    s_post = _cosine_pair_1d(_flatten_channel_major(z)[c], _flatten_channel_major(z)[d])

    ac = a[:, c]
    ad = a[:, d]
    g_corr = _cosine_pair_1d(ac, ad)

    clipped = np.clip(a, NORM_ZERO_THRESHOLD, None)
    clip_count = int(np.count_nonzero(a < NORM_ZERO_THRESHOLD))
    log_ratio = np.log(clipped[:, c]) - np.log(clipped[:, d])
    g_lstd = float(np.std(log_ratio, ddof=0))

    diff = ac - ad
    signs = np.sign(diff)
    nonzero = signs[signs != 0.0]
    if nonzero.size < 2:
        g_flip = 0.0
    else:
        g_flip = float(np.count_nonzero(nonzero[1:] != nonzero[:-1]) / (nonzero.size - 1))

    h_cos = _cosine_pair_1d(_flatten_channel_major(h)[c], _flatten_channel_major(h)[d])
    strength = float(min(ac.mean(), ad.mean()))
    g_amp = float(h_cos * strength) if math.isfinite(h_cos) else float("nan")

    if w1.shape[1] != channels:
        raise ValueError(f"fc1_weight: C={w1.shape[1]} niezgodne z {channels}")
    g_c = g[:, c]
    g_d = g[:, d]
    s_full = g @ w1.T
    delta = (alpha * g_c - g_d)[:, None] * w1[:, d][None, :]
    num = np.linalg.norm(delta, axis=1)
    den = np.linalg.norm(s_full, axis=1)
    ratio = np.full(n, np.nan, dtype=np.float64)
    good = den > NORM_ZERO_THRESHOLD
    ratio[good] = num[good] / den[good]
    if np.any(good):
        g_pert = float(np.mean(ratio[good]))
    else:
        g_pert = float("nan")

    return {
        "c": c,
        "d": d,
        "alpha": float(alpha),
        "beta": float(beta),
        "s_post": float(s_post),
        "s_pre": float(s_pre),
        "g_corr": float(g_corr),
        "g_lstd": g_lstd,
        "g_lstd_clip_count": clip_count,
        "g_flip": g_flip,
        "g_amp": g_amp,
        "g_pert": g_pert,
    }


# --------------------------------------------------------------------------- #
# 5) Folding pojedynczego bloku MBConv
# --------------------------------------------------------------------------- #
def _set_param(module, name, tensor):
    """Podmienia parametr, zachowujac flage ``requires_grad``."""
    old = module._parameters.get(name)
    requires_grad = bool(old.requires_grad) if old is not None else False
    module._parameters[name] = nn.Parameter(tensor, requires_grad=requires_grad)


def _slice_batch_norm(bn, keep):
    """Przycina affine i bufory BN do indeksow ``keep``; zachowuje klase i eps."""
    idx = torch.as_tensor(keep, dtype=torch.long)
    for name in ("weight", "bias"):
        param = getattr(bn, name, None)
        if param is not None:
            _set_param(bn, name, param.detach().index_select(0, idx.to(param.device)).clone())
    for name in ("running_mean", "running_var"):
        buf = getattr(bn, name, None)
        if buf is not None:
            setattr(bn, name, buf.detach().index_select(0, idx.to(buf.device)).clone())
    if hasattr(bn, "num_features"):
        bn.num_features = len(keep)


def fold_block_for_pair(block, c, d, alpha, beta):
    """Zwraca ``deepcopy`` bloku MBConv z usunietym kanalem ``d`` i scaleniem ``alpha``/``beta``.

    Operacje (jak w E003 ``_transform_block``):

    - expansion conv: usuniety wiersz ``d``; expansion BN: usuniety ``d``
      (waga, bias, bufory);
    - depthwise conv: usuniety wiersz ``d``; depthwise BN: usuniety ``d``;
    - SE fc1: kolumna ``c += alpha * kolumna d``, nastepnie usunieta kolumna ``d``;
    - SE fc2: usuniety wiersz ``d`` i bias ``d``;
    - projection: kolumna ``c += beta * kolumna d``, nastepnie usunieta kolumna ``d``.

    Zachowane sa atrybuty ``out_channels``/``in_channels``/``groups`` oraz typy
    BN (rowniez ``SyncBatchNorm``). Wejscie nie jest modyfikowane; walidowane
    jest ``0 <= c < d < C``.
    """
    channels = int(block.block[0][0].out_channels)
    c = int(c)
    d = int(d)
    if not (0 <= c < d < channels):
        raise ValueError(f"para ({c}, {d}) poza zakresem 0 <= c < d < {channels}")

    folded = copy.deepcopy(block)
    expansion, depthwise, se, projection = folded.block
    conv, exp_bn = expansion[0], expansion[1]
    dw, dw_bn = depthwise[0], depthwise[1]
    fc1, fc2 = se.fc1, se.fc2
    proj = projection[0]

    keep = [i for i in range(channels) if i != d]
    new_channels = len(keep)
    idx = torch.as_tensor(keep, dtype=torch.long, device=conv.weight.device)

    fc1_weight = fc1.weight.detach().clone()
    fc1_weight[:, c, :, :] = fc1_weight[:, c, :, :] + alpha * fc1_weight[:, d, :, :]
    proj_weight = proj.weight.detach().clone()
    proj_weight[:, c, :, :] = proj_weight[:, c, :, :] + beta * proj_weight[:, d, :, :]

    _set_param(conv, "weight", conv.weight.detach().index_select(0, idx).clone())
    if conv.bias is not None:
        _set_param(conv, "bias", conv.bias.detach().index_select(0, idx).clone())
    conv.out_channels = new_channels
    expansion.out_channels = new_channels
    _slice_batch_norm(exp_bn, keep)

    _set_param(dw, "weight", dw.weight.detach().index_select(0, idx).clone())
    if dw.bias is not None:
        _set_param(dw, "bias", dw.bias.detach().index_select(0, idx).clone())
    dw.groups = new_channels
    dw.in_channels = new_channels
    dw.out_channels = new_channels
    depthwise.out_channels = new_channels
    _slice_batch_norm(dw_bn, keep)

    _set_param(fc1, "weight", fc1_weight.index_select(1, idx).clone())
    fc1.in_channels = new_channels

    _set_param(fc2, "weight", fc2.weight.detach().index_select(0, idx).clone())
    fc2.out_channels = new_channels
    if fc2.bias is not None:
        _set_param(fc2, "bias", fc2.bias.detach().index_select(0, idx).clone())

    _set_param(proj, "weight", proj_weight.index_select(1, idx).clone())
    proj.in_channels = new_channels
    return folded


# --------------------------------------------------------------------------- #
# 6-7) Wyjscia bloku i bled wzgledny
# --------------------------------------------------------------------------- #
def block_outputs(block, inputs):
    """Wyjscia bloku w trybie ``eval``/``no_grad``: ``(branch, out)``.

    ``branch = block.block(inputs)`` (sciezka expansion->depthwise->SE->project),
    ``out = branch + inputs`` gdy ``block.use_res_connect``, inaczej ``branch``.
    Tryb treningowy bloku jest tymczasowo wylaczany i przywracany; stochastic
    depth w ``eval`` jest identycznoscia, wiec wzorzec zgadza sie z E003.
    """
    was_training = block.training
    block.eval()
    try:
        with torch.no_grad():
            branch = block.block(inputs).clone()
            if bool(getattr(block, "use_res_connect", False)):
                out = branch + inputs
            else:
                out = branch
    finally:
        block.train(was_training)
    return branch, out


def relative_mse(student, teacher):
    """``sum((s - t)^2) / sum(t^2)``; ``None`` gdy mianownik ``<= 1e-12``."""
    s = _as_float64(student, "student")
    t = _as_float64(teacher, "teacher")
    if s.shape != t.shape:
        raise ValueError(f"student {s.shape} i teacher {t.shape}: niezgodne ksztalty")
    denom = float(np.sum(t * t))
    if denom <= NORM_ZERO_THRESHOLD:
        return None
    return float(np.sum((s - t) ** 2) / denom)


# --------------------------------------------------------------------------- #
# 8) Statystyki rangowe
# --------------------------------------------------------------------------- #
def _average_ranks(x):
    """Rangi z srednimi dla remisow (``1..n``), bez mutacji wejscia."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    ranks = np.empty(n, dtype=np.float64)
    if n == 0:
        return ranks
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2 or y.size != x.size:
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    dx = float(np.sqrt(np.dot(xc, xc)))
    dy = float(np.sqrt(np.dot(yc, yc)))
    if dx <= 0.0 or dy <= 0.0:
        return float("nan")
    return float(np.dot(xc, yc) / (dx * dy))


def spearman(x, y):
    """Spearman z rangami srednimi dla remisow; ``NaN`` gdy sygnal staly."""
    return _pearson(_average_ranks(x), _average_ranks(y))


def _rank_residual(target, regressor):
    """Reszty regresji liniowej (z wyrazem wolnym) ``rank(target) ~ rank(regressor)``."""
    a = _average_ranks(target)
    b = _average_ranks(regressor)
    bc = b - b.mean()
    denom = float(np.dot(bc, bc))
    if denom <= 0.0:
        return np.full_like(a, np.nan)
    slope = float(np.dot(bc, a - a.mean()) / denom)
    intercept = a.mean() - slope * b.mean()
    return a - (intercept + slope * b)


def partial_spearman(x, y, z):
    """Korelacja czastkowa rang: korelacja reszt ``rank(y) ~ rank(z)`` i ``rank(x) ~ rank(z)``."""
    ex = _rank_residual(x, z)
    ey = _rank_residual(y, z)
    return _pearson(ex, ey)


# --------------------------------------------------------------------------- #
# 9) Bootstrap roznicy Spearmanow
# --------------------------------------------------------------------------- #
def bootstrap_spearman_diff(sig_a, sig_b, damage, reps=10000, seed=20261013):
    """Bootstrap po parach roznicy ``rho(sig_a) - rho(sig_b)`` wzgledem szkody.

    ``rho`` to :func:`spearman`. Losowanie z powtorzeniami indeksow par
    (``reps`` razy, RNG z ``seed``); ``ci95`` to percentyle 2.5 i 97.5.
    Zwraca ``{delta, ci95, ci95_low, ci95_high, reps, valid, invalid, seed}``;
    ``delta`` liczony na pelnym wektorze (bez resamplingu).
    """
    a = np.asarray(sig_a, dtype=np.float64).ravel()
    b = np.asarray(sig_b, dtype=np.float64).ravel()
    y = np.asarray(damage, dtype=np.float64).ravel()
    if not (a.shape == b.shape == y.shape):
        raise ValueError("sig_a, sig_b, damage: niezgodne dlugosci")
    n = y.size
    if n < 2:
        raise ValueError("bootstrap: potrzeba co najmniej 2 par")
    reps = int(reps)
    if reps < 1:
        raise ValueError("reps musi byc >= 1")

    delta = spearman(a, y) - spearman(b, y)
    rng = np.random.default_rng(seed)
    samples = np.empty(reps, dtype=np.float64)
    invalid = 0
    valid = 0
    for i in range(reps):
        pick = rng.integers(0, n, size=n)
        d = spearman(a[pick], y[pick]) - spearman(b[pick], y[pick])
        if math.isfinite(d):
            samples[valid] = d
            valid += 1
        else:
            invalid += 1
    if valid > 0:
        lo, hi = np.percentile(samples[:valid], [2.5, 97.5])
        lo = float(lo)
        hi = float(hi)
    else:
        lo = hi = float("nan")
    return {
        "delta": float(delta),
        "ci95": [lo, hi],
        "ci95_low": lo,
        "ci95_high": hi,
        "reps": reps,
        "valid": int(valid),
        "invalid": int(invalid),
        "seed": int(seed),
    }


# --------------------------------------------------------------------------- #
# 10) Kontrole CPU
# --------------------------------------------------------------------------- #
def _duplicate_channel(module, src, dst):
    """Kopiuje kanal ``src`` w ``dst`` dla weight/bias/running_* (in-place)."""
    for name in ("weight", "bias", "running_mean", "running_var"):
        tensor = getattr(module, name, None)
        if tensor is not None:
            with torch.no_grad():
                tensor[dst] = tensor[src].clone()


def self_test():
    """Kontrole poprawnosci na malych danych i prawdziwym bloku ``features.4.1``.

    Sprawdza: (a) folding zduplikowanego kanalu z ``alpha=beta=1`` zachowuje
    wyjscie bloku; (b) ``g_pert`` analityczne = numeryczna zmiana preaktywacji
    fc1 po foldzie; (c) ``spearman`` rosnacy = 1; (d) bootstrap identycznych
    sygnalow daje ``delta=0``; (e) zerowe normy daja ``NaN``; (f) skosny
    ``s_post`` jak w nieudanym biegu 1.0 daje 5x30 par rozlacznych przy
    kubelkach kwantylowych. Zwraca slownik
    liczb; podnosi ``AssertionError`` przy niepowodzeniu.
    """
    from torchvision.models import efficientnet_b0

    torch.manual_seed(20261011)
    rng = np.random.default_rng(20261011)
    nums = {}

    rho_inc = spearman(np.arange(6), 3.0 * np.arange(6) + 1.0)
    assert abs(rho_inc - 1.0) < 1e-12, f"(c) spearman rosnacy = {rho_inc}"
    nums["spearman_increasing"] = rho_inc

    sig = rng.standard_normal(40)
    damage = rng.standard_normal(40)
    boot = bootstrap_spearman_diff(sig, sig, damage, reps=256, seed=20261013)
    assert abs(boot["delta"]) < 1e-12, f"(d) bootstrap delta = {boot['delta']}"
    assert boot["ci95"] == [0.0, 0.0], f"(d) bootstrap ci95 = {boot['ci95']}"
    nums["bootstrap_delta"] = boot["delta"]
    nums["bootstrap_ci95_low"] = boot["ci95_low"]
    nums["bootstrap_ci95_high"] = boot["ci95_high"]

    maps = rng.standard_normal((4, 5, 2, 2))
    maps[:, 2, :, :] = 0.0
    ci, di, cos = all_pair_cosines(maps)
    touched = (ci == 2) | (di == 2)
    assert np.all(np.isnan(cos[touched])), "(e) zerowa norma nie dala NaN"
    assert np.all(np.isfinite(cos[~touched])), "(e) poprawna para dala NaN"
    nums["zero_norm_nan_pairs"] = int(np.count_nonzero(touched))
    nums["zero_norm_finite_pairs"] = int(np.count_nonzero(~touched))

    n_ch = 480
    ci_t, di_t = np.triu_indices(n_ch, k=1)
    s_t = 0.02 * rng.standard_normal(ci_t.size)
    clique = np.sort(rng.permutation(n_ch)[:24])
    in_clique = np.isin(ci_t, clique) & np.isin(di_t, clique)
    s_t[in_clique] = 0.9 + 0.05 * rng.standard_normal(int(in_clique.sum()))
    pairs_t = sample_disjoint_pairs(ci_t, di_t, s_t, bins=5, per_bin=30, seed=20261012)
    used_t = np.unique(pairs_t)
    assert pairs_t.shape == (150, 2), f"(f) rozlaczne pary = {pairs_t.shape}"
    assert used_t.size == 300, f"(f) uzyte kanaly = {used_t.size}"
    assert np.all(pairs_t[:, 0] < pairs_t[:, 1]), "(f) pary bez c < d"
    assert np.all(np.diff(pairs_t[:, 0] * n_ch + pairs_t[:, 1]) > 0), "(f) kolejnosc par"
    nums["quantile_skew_pairs"] = int(pairs_t.shape[0])
    nums["quantile_skew_channels"] = int(used_t.size)

    base = efficientnet_b0(weights=None, num_classes=200)
    in_ch = base.features[4][1].block[0][0].in_channels
    x = torch.randn(2, in_ch, 6, 6, dtype=torch.float64)

    block_a = copy.deepcopy(base.features[4][1]).double().eval()
    channels = block_a.block[0][0].out_channels
    c, d = 5, 200
    with torch.no_grad():
        expansion, depthwise, se, _ = block_a.block
        _duplicate_channel(expansion[0], c, d)
        _duplicate_channel(expansion[1], c, d)
        _duplicate_channel(depthwise[0], c, d)
        _duplicate_channel(depthwise[1], c, d)
        se.fc2.weight[c] = se.fc2.weight[d].clone()
        se.fc2.bias[c] = se.fc2.bias[d].clone()
    _, before = block_outputs(block_a, x)
    folded_a = fold_block_for_pair(block_a, c, d, 1.0, 1.0)
    _, after = block_outputs(folded_a, x)
    fold_err = float((before - after).abs().max())
    assert fold_err < 1e-5, f"(a) blad foldingu duplikatu = {fold_err}"
    nums["fold_duplicate_max_abs"] = fold_err

    block_b = copy.deepcopy(base.features[4][1]).double().eval()
    se_b = block_b.block[2]
    with torch.no_grad():
        h = block_b.block[1](block_b.block[0](x))
        g = se_b.avgpool(h)
        scale = se_b.fc2(se_b.activation(se_b.fc1(g)))
        a = se_b.scale_activation(scale)
        z = h * a
    sig_pair = pair_signals(
        h.numpy(),
        z.numpy(),
        a.numpy(),
        g.reshape(g.shape[0], g.shape[1]).numpy(),
        se_b.fc1.weight.detach().numpy(),
        c,
        d,
    )
    w1 = se_b.fc1.weight.detach().numpy().reshape(se_b.fc1.out_channels, channels)
    g2d = g.reshape(g.shape[0], g.shape[1]).numpy()
    s_before = g2d @ w1.T
    den = np.linalg.norm(s_before, axis=1)
    folded_b = fold_block_for_pair(block_b, c, d, sig_pair["alpha"], sig_pair["beta"])
    se_f = folded_b.block[2]
    with torch.no_grad():
        h2 = folded_b.block[1](folded_b.block[0](x))
        g2 = se_f.avgpool(h2).reshape(h2.shape[0], h2.shape[1])
        w1f = se_f.fc1.weight.detach().numpy().reshape(
            se_f.fc1.out_channels, h2.shape[1]
        )
        s_after = g2.numpy() @ w1f.T
    numeric_pert = float(np.mean(np.linalg.norm(s_after - s_before, axis=1) / den))
    pert_err = abs(sig_pair["g_pert"] - numeric_pert)
    assert pert_err < 1e-8, (
        f"(b) g_pert analityczne={sig_pair['g_pert']} numeryczne={numeric_pert}"
    )
    nums["g_pert_analytic"] = sig_pair["g_pert"]
    nums["g_pert_numeric"] = numeric_pert
    nums["g_pert_abs_error"] = pert_err

    print("E006_METRICS_SELF_TEST=passed")
    print(
        "  fold_duplicate_max_abs={fold:.3e}  g_pert_analytic={ga:.3e}  "
        "g_pert_numeric={gn:.3e}  g_pert_abs_error={ge:.3e}".format(
            fold=nums["fold_duplicate_max_abs"],
            ga=nums["g_pert_analytic"],
            gn=nums["g_pert_numeric"],
            ge=nums["g_pert_abs_error"],
        )
    )
    print(
        "  spearman_increasing={si:.1f}  bootstrap_delta={bd:.1f}  "
        "ci95=[{lo:.1f}, {hi:.1f}]  zero_norm_nan_pairs={zn}  "
        "quantile_skew_pairs={qp}  quantile_skew_channels={qc}".format(
            si=nums["spearman_increasing"],
            bd=nums["bootstrap_delta"],
            lo=nums["bootstrap_ci95_low"],
            hi=nums["bootstrap_ci95_high"],
            zn=nums["zero_norm_nan_pairs"],
            qp=nums["quantile_skew_pairs"],
            qc=nums["quantile_skew_channels"],
        )
    )
    return nums


if __name__ == "__main__":
    self_test()
