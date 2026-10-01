"""E002: miary par kanałów przed i po SE (czyste funkcje NumPy, tylko CPU).

Eksperyment: **2026-10-01_E002_obserwacja_se**.
Pełny opis zakresu i miar: ``OPIS.md`` w tym samym katalogu. Zamrożona wersja
protokołu, przygotowywana przez agenta głównego, to ``PROTOKOL_v1.md``.

Co robi
-------
Liczy, dla jednego bloku i jednego zbioru obrazów, miary wszystkich par
kanałów ``c < d`` na wektorach aktywacji złożonych z wielu obrazów:

- podobieństwo cosinusowe przed SE (``s_h``) i po SE (``s_z``) oraz ich
  zmianę (``delta_s = s_z - s_h``),
- unormowaną różnicę wektorów przed/po SE (``d_h``, ``d_z``),
- statystyki różnicy wag SE (``b_mean``, ``b_rms``, ``b_std``),
- statystyki logarytmu stosunku wag SE (``l_mean``, ``l_std``).

Nie trenuje, nie zmienia wag i nie dotyka GPU, danych ani checkpointów.
Nie zapisuje plików. Wynik ma być używany przez ``analyze.py`` (agent główny),
który odpowiada za agregację i raport.

Wejścia
-------
``h``, ``z`` : ``np.ndarray`` kształtu ``(N, C, H, W)``, FP32.
    Mapy kanałów odpowiednio przed SE (``h``) i po SE (``z = a * h``).
``a`` : ``np.ndarray`` kształtu ``(N, C)`` albo ``(N, C, 1, 1)``, FP32.
    Wagi SE dla każdego obrazu i kanału; muszą mieścić się w ``[0, 1]``.
``cfg`` : ``dict`` z kluczami:
    - ``norm_zero_threshold`` (wymagany) — progiem normy, poniżej lub równo
      którego wektor kanału jest numerycznie zerowy (``<=`` daje NaN dla
      ``s_h``/``s_z``/``d_h``/``d_z``),
    - ``attention_log_clip_min`` (opcjonalny, domyślnie ``1e-12``) — dolne
      przycięcie wag przy liczeniu ``log(a)``.

Wszystkie wartości wejściowe muszą być skończone; NaN/Inf przerywa obliczenia.
Wszystkie rachunki wykonujemy w ``float64``. Wejścia nie są modyfikowane.

Wyjścia
-------
``pairs`` : ``dict`` z ``np.ndarray`` o długości ``C * (C - 1) / 2`` wpisów
    w porządku leksykograficznym górnego trójkąta (``c`` rosnące, przy równym
    ``c`` ``d`` rosnące). Klucze i znaczenie:

    - ``c``, ``d`` — indeksy kanałów pary (``int64``),
    - ``s_h``, ``s_z`` — cosinus odpowiednio przed i po SE,
    - ``delta_s`` — ``s_z - s_h``,
    - ``d_h``, ``d_z`` — ``||v_c - v_d|| / (||v_c|| + ||v_d||)``,
    - ``b_mean``, ``b_rms``, ``b_std`` — średnia, RMS i odchylenie (``ddof=0``)
      różnicy wag ``b(x) = a_c(x) - a_d(x)``,
    - ``l_mean``, ``l_std`` — średnia i odchylenie (``ddof=0``) logarytmu
      stosunku wag ``log(clip(a_c)) - log(clip(a_d))``.

    Tam, gdzie któraś z norm jest ``<= norm_zero_threshold``, odpowiednie
    ``s``/``d`` są ``NaN`` (nie wpisujemy zera). Miary ``b_*`` i ``l_*``
    zależą wyłącznie od ``a`` i są zawsze skończone.

``channels`` : ``dict`` z wektorami o długości ``C``:
    ``c``, ``norm_h``, ``norm_z``, ``zero_maps_h``, ``zero_maps_z``,
    ``a_mean``, ``a_std``, ``a_min``, ``a_max``.
    ``zero_maps_*`` to liczba map danego kanału, które są dokładnie zerowe.

``diagnostics`` : ``dict`` możliwy do serializacji JSON (tylko ``int``/``float``/
    ``str``): liczba przycięć logarytmu, liczba kanałów numerycznie zerowych
    oraz liczba korekt przycinania cosinusa do ``[-1, 1]``.

Uwagi numeryczne
----------------
Cosinusy i różnice liczymy z macierzy Grama ``v @ v.T`` (jeden iloczyn
macierzowy na ``h`` i jeden na ``z``). Odchylenia różnic wag wyznaczamy
z macierzy scentralizowanej: ``var(b) = var(a_c) + var(a_d) - 2 cov(a_c, a_d)``,
a wynik ujemny tylko od zaokrągleń przycinamy do zera. Analogicznie dla
logarytmów wag. ``b_rms`` liczymy jako ``sqrt(var(b) + mean(b)**2)``.
"""

from __future__ import annotations

import numpy as np

EXPERIMENT_ID = "2026-10-01_E002_obserwacja_se"
OPIS_PATH = "OPIS.md"
PROTOKOL_V1_PATH = "PROTOKOL_v1.md"

_PAIR_KEYS = (
    "c",
    "d",
    "s_h",
    "s_z",
    "delta_s",
    "d_h",
    "d_z",
    "b_mean",
    "b_rms",
    "b_std",
    "l_mean",
    "l_std",
)
_CHANNEL_KEYS = (
    "c",
    "norm_h",
    "norm_z",
    "zero_maps_h",
    "zero_maps_z",
    "a_mean",
    "a_std",
    "a_min",
    "a_max",
)


def _as_float64(x, name):
    """Kopiuje tablicę do ``float64``; wymaga typu zmiennoprzecinkowego."""
    arr = np.asarray(x)
    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(f"{name}: oczekiwano typu zmiennoprzecinkowego, jest {arr.dtype}")
    return arr.astype(np.float64, copy=True)


def _flatten_channel_major(x4):
    """``(N, C, H, W) -> (C, N*H*W)``; kolejność kanał, obraz, wiersz, kolumna."""
    return x4.transpose(1, 0, 2, 3).reshape(x4.shape[1], -1)


def _normalize_a(a):
    """Akceptuje ``(N, C)`` oraz ``(N, C, 1, 1)``; zwraca ``(N, C)`` bez mutacji."""
    if a.ndim == 4:
        if a.shape[2] != 1 or a.shape[3] != 1:
            raise ValueError(
                f"a: dopuszczalne tylko (N, C) lub (N, C, 1, 1), jest {a.shape}"
            )
        return a[:, :, 0, 0]
    if a.ndim == 2:
        return a
    raise ValueError(f"a: dopuszczalne tylko (N, C) lub (N, C, 1, 1), jest {a.shape}")


def compute_pairs(h, z, a, cfg):
    """Liczy miary wszystkich par kanałów ``c < d`` dla jednego bloku.

    Parametry i znaczenie zwracanych wartości opisano w docstringu modułu.
    Funkcja jest czysta: waliduje kształty i zakres ``a``, nie modyfikuje
    wejść i nie wykonuje żadnego zapisu.

    Returns
    -------
    (pairs, channels, diagnostics) : tuple[dict, dict, dict]
    """
    h64 = _as_float64(h, "h")
    z64 = _as_float64(z, "z")
    a64 = _as_float64(a, "a")

    if h64.ndim != 4:
        raise ValueError(f"h: oczekiwano (N, C, H, W), jest {h64.shape}")
    if z64.shape != h64.shape:
        raise ValueError(f"z: kształt {z64.shape} różni się od h {h64.shape}")

    a2 = _normalize_a(a64)

    n, c, hh, ww = h64.shape
    if a2.shape != (n, c):
        raise ValueError(f"a: kształt {a2.shape} niezgodny z (N, C)=({n}, {c})")

    if not np.all(np.isfinite(h64)):
        raise ValueError("h: wykryto NaN/Inf")
    if not np.all(np.isfinite(z64)):
        raise ValueError("z: wykryto NaN/Inf")
    if not np.all(np.isfinite(a2)):
        raise ValueError("a: wykryto NaN/Inf")
    if a2.size and (a2.min() < 0.0 or a2.max() > 1.0):
        raise ValueError("a: wagi poza zakresem [0, 1]")

    if "norm_zero_threshold" not in cfg:
        raise KeyError("cfg: brak wymaganego klucza 'norm_zero_threshold'")
    norm_zero = float(cfg["norm_zero_threshold"])
    if norm_zero < 0.0:
        raise ValueError("cfg['norm_zero_threshold'] musi być >= 0")
    log_clip_min = float(cfg.get("attention_log_clip_min", 1e-12))

    hf = _flatten_channel_major(h64)
    zf = _flatten_channel_major(z64)

    norm_h = np.sqrt(np.einsum("cl,cl->c", hf, hf))
    norm_z = np.sqrt(np.einsum("cl,cl->c", zf, zf))

    gram_h = hf @ hf.T
    gram_z = zf @ zf.T

    h_blocks = h64.reshape(n, c, hh * ww)
    z_blocks = z64.reshape(n, c, hh * ww)
    zero_maps_h = np.count_nonzero(np.all(h_blocks == 0.0, axis=2), axis=0).astype(np.int64)
    zero_maps_z = np.count_nonzero(np.all(z_blocks == 0.0, axis=2), axis=0).astype(np.int64)

    a_mean = a2.mean(axis=0)
    a_std = a2.std(axis=0, ddof=0)
    a_min = a2.min(axis=0)
    a_max = a2.max(axis=0)

    a_centered = a2 - a_mean
    cov_a = (a_centered.T @ a_centered) / n
    var_a = np.diag(cov_a)

    a_clipped = np.clip(a2, log_clip_min, 1.0)
    log_clip_count = int(np.count_nonzero(a2 < log_clip_min))
    log_clip_count += int(np.count_nonzero(a2 > 1.0))
    log_a = np.log(a_clipped)
    log_a_mean = log_a.mean(axis=0)
    log_a_centered = log_a - log_a_mean
    cov_l = (log_a_centered.T @ log_a_centered) / n
    var_l = np.diag(cov_l)

    iu, ju = np.triu_indices(c, k=1)
    ci = iu.astype(np.int64)
    di = ju.astype(np.int64)
    npairs = ci.size

    valid_h = (norm_h[ci] > norm_zero) & (norm_h[di] > norm_zero)
    valid_z = (norm_z[ci] > norm_zero) & (norm_z[di] > norm_zero)

    denom_h = norm_h[ci] * norm_h[di]
    denom_z = norm_z[ci] * norm_z[di]

    s_h = np.full(npairs, np.nan, dtype=np.float64)
    s_z = np.full(npairs, np.nan, dtype=np.float64)
    np.divide(gram_h[ci, di], denom_h, out=s_h, where=valid_h)
    np.divide(gram_z[ci, di], denom_z, out=s_z, where=valid_z)

    clip_h = int(np.count_nonzero(valid_h & ((s_h < -1.0) | (s_h > 1.0))))
    clip_z = int(np.count_nonzero(valid_z & ((s_z < -1.0) | (s_z > 1.0))))
    s_h = np.clip(s_h, -1.0, 1.0)
    s_z = np.clip(s_z, -1.0, 1.0)

    delta_s = s_z - s_h

    diff_sq_h = norm_h[ci] ** 2 + norm_h[di] ** 2 - 2.0 * gram_h[ci, di]
    diff_sq_z = norm_z[ci] ** 2 + norm_z[di] ** 2 - 2.0 * gram_z[ci, di]
    diff_sq_h = np.maximum(diff_sq_h, 0.0)
    diff_sq_z = np.maximum(diff_sq_z, 0.0)

    d_h = np.full(npairs, np.nan, dtype=np.float64)
    d_z = np.full(npairs, np.nan, dtype=np.float64)
    np.divide(
        np.sqrt(diff_sq_h),
        norm_h[ci] + norm_h[di],
        out=d_h,
        where=valid_h,
    )
    np.divide(
        np.sqrt(diff_sq_z),
        norm_z[ci] + norm_z[di],
        out=d_z,
        where=valid_z,
    )

    b_mean = a_mean[ci] - a_mean[di]
    var_b = var_a[ci] + var_a[di] - 2.0 * cov_a[ci, di]
    var_b = np.maximum(var_b, 0.0)
    b_std = np.sqrt(var_b)
    b_rms = np.sqrt(var_b + b_mean ** 2)

    l_mean = log_a_mean[ci] - log_a_mean[di]
    var_l_pair = var_l[ci] + var_l[di] - 2.0 * cov_l[ci, di]
    var_l_pair = np.maximum(var_l_pair, 0.0)
    l_std = np.sqrt(var_l_pair)

    pairs = {
        "c": ci,
        "d": di,
        "s_h": s_h,
        "s_z": s_z,
        "delta_s": delta_s,
        "d_h": d_h,
        "d_z": d_z,
        "b_mean": b_mean,
        "b_rms": b_rms,
        "b_std": b_std,
        "l_mean": l_mean,
        "l_std": l_std,
    }
    channels = {
        "c": np.arange(c, dtype=np.int64),
        "norm_h": norm_h,
        "norm_z": norm_z,
        "zero_maps_h": zero_maps_h,
        "zero_maps_z": zero_maps_z,
        "a_mean": a_mean,
        "a_std": a_std,
        "a_min": a_min,
        "a_max": a_max,
    }
    diagnostics = {
        "experiment": EXPERIMENT_ID,
        "N": int(n),
        "C": int(c),
        "H": int(hh),
        "W": int(ww),
        "n_pairs": int(npairs),
        "norm_zero_threshold": norm_zero,
        "attention_log_clip_min": log_clip_min,
        "log_clip_count": log_clip_count,
        "norm_zero_channels_h": int(np.count_nonzero(norm_h <= norm_zero)),
        "norm_zero_channels_z": int(np.count_nonzero(norm_z <= norm_zero)),
        "norm_zero_channels": int(
            np.count_nonzero((norm_h <= norm_zero) | (norm_z <= norm_zero))
        ),
        "cosine_clip_corrections": int(clip_h + clip_z),
        "cosine_clip_corrections_s_h": clip_h,
        "cosine_clip_corrections_s_z": clip_z,
        "undefined_pairs_s_h": int(np.count_nonzero(~valid_h)),
        "undefined_pairs_s_z": int(np.count_nonzero(~valid_z)),
    }
    return pairs, channels, diagnostics


def select_pairs(pairs, count):
    """Wybiera ``count`` par o największym **skończonym** ``s_h``.

    Kwalifikacja odbywa się wyłącznie po przed-SE ``s_h`` (nie po różnicach
    atencji ani po ``s_z``). Pary z ``s_h`` równym ``NaN`` są niekwalifikowane.
    Remisy rozstrzyga porządek ``(c, d)``. Zwracane są indeksy do tablic
    w ``pairs``, długości ``min(count, liczba kwalifikujących się par)``.
    Wybór jest deterministyczny i nie zależy od kolejności wejścia.
    """
    s_h = np.asarray(pairs["s_h"], dtype=np.float64)
    c = np.asarray(pairs["c"], dtype=np.int64)
    d = np.asarray(pairs["d"], dtype=np.int64)

    valid = np.flatnonzero(np.isfinite(s_h))
    limit = min(int(count), valid.size)
    if limit <= 0:
        return valid[:0]

    order = np.lexsort((d[valid], c[valid], -s_h[valid]))
    return valid[order][:limit]
