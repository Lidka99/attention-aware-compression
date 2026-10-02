"""Rozkład stałej, obrazowej i przestrzennej składowej map E002.

Cel
---
Czysta funkcja NumPy ``decompose_maps`` rozkłada zapisane mapy aktywacji
kanalów (bez SE i po SE, h albo z) na trzy ortogonalne składowe i liczy dla
wybranych par kanałów cosinusy: surowy oraz po centrowaniu globalnym
i przestrzennym. Służy do kontroli, czy wysoki surowy cosinus par wybranych
w E002 wynika głównie ze wspólnej składowej stałej kanałów.

Eksperyment: ``2026-10-01_E002_obserwacja_se``.
Analiza: ``2026-10-02_skladowa_stala`` (post hoc, protokół frozen v1.0).
Pełny opis: lokalny ``OPIS.md`` (``analizy/2026-10-02_skladowa_stala/OPIS.md``)
oraz ``../../OPIS.md`` (``experiments/2026-10-01_E002_obserwacja_se/OPIS.md``).

Wejścia
-------
``maps`` : tablica ``(N, C, H, W)`` zmiennoprzecinkowa, skończona. To wektor
    wszystkich map danego podzbioru (np. 500 obrazów), w kolejności
    odpowiadającej parom. Obliczenia wykonujemy w FP64; wejście nie jest
    modyfikowane.
``pair_ids`` : tablica ``(M, 2)`` liczb całkowitych, ``c < d``, bez powtórzeń,
    ``0 <= c, d < C``.
``cfg`` : dict z kluczami ``norm_zero_threshold``,
    ``near_constant_relative_norm``, ``identity_atol``,
    ``energy_relative_tolerance``, ``energy_absolute_tolerance``.

Wyjścia
-------
``(channels, pairs, diagnostics)``:
``channels`` : dict tablic długości ``C`` — ``c``, ``mean``, ``norm_raw``,
    ``norm_global``, ``norm_spatial``, ``frac_const``, ``frac_image``,
    ``frac_spatial``, ``near_zero_global``, ``near_zero_spatial``.
``pairs`` : dict tablic długości ``M`` — ``c``, ``d``, ``s_raw``,
    ``s_global``, ``s_spatial``, ``delta_global``, ``delta_spatial``,
    ``C_const``, ``C_rest``.
``diagnostics`` : dict gotowy do serializacji JSON z licznikami i błędami
    kontroli numerycznych.

Model rozkładu (FP64, na bezpośrednio wycentrowanych mapach)
------------------------------------------------------------
``mu_c = mean(U_c)`` po ``N*H*W``; ``m_c(x) = mean_p U_c(x, p)``.
``U_c = mu_c + (m_c(x) - mu_c) + (U_c - m_c(x))``, składowe ortogonalne.
``E = E_const + E_image + E_spatial``, z ``E_const = N*P*mu_c^2``,
``E_image = P*sum_x (m_c(x) - mu_c)^2``, ``E_spatial = ||U_c - m_c(x)||^2``.
Centrowane wektory liczymy wprost (``U - mu``, ``U - m(x)``), a nie przez
odejmowanie dużych Gramów, więc nie tracimy cyfr znaczących.

Normy ``<= cfg['norm_zero_threshold']`` oznaczają składową nieokreśloną:
odpowiedni cosinus/udział to ``NaN`` (nie 0, nie 1). Składowe o normie
``<= cfg['near_constant_relative_norm'] * norm_raw`` są oznaczane jako
``near_zero_*`` (bardzo małe względnie), ale pozostają w tabelach.

Funkcja nie wykonuje wejścia/wyjścia, nie dotyka GPU ani checkpointów.
"""

from __future__ import annotations

import numpy as np

_REQUIRED_CFG = (
    "norm_zero_threshold",
    "near_constant_relative_norm",
    "identity_atol",
    "energy_relative_tolerance",
    "energy_absolute_tolerance",
)

# Próg, poniżej którego błąd energii kontrolujemy bezwzględnie (OPIS.md).
_SMALL_ENERGY = 1e-24


def _validate_cfg(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        raise TypeError("cfg musi być typu dict")
    for key in _REQUIRED_CFG:
        if key not in cfg:
            raise KeyError(f"cfg nie zawiera wymaganego klucza: {key}")
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"cfg['{key}'] musi być liczbą, jest {type(value)!r}")
        if not np.isfinite(value):
            raise ValueError(f"cfg['{key}'] musi być skończone")
        if value < 0:
            raise ValueError(f"cfg['{key}'] musi być nieujemne")


def _clip_cosine(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Przycina skończone cosinusy do [-1, 1] i zwraca liczbę korekt."""
    result = np.array(values, dtype=np.float64, copy=True)
    mask = np.isfinite(result)
    if not np.any(mask):
        return result, 0
    clipped = np.clip(result[mask], -1.0, 1.0)
    count = int(np.count_nonzero(clipped != result[mask]))
    result[mask] = clipped
    return result, count


def decompose_maps(maps, pair_ids, cfg):
    """Rozkłada mapy E002 i liczy diagnostykę par; patrz docstring modułu.

    Zwraca ``(channels, pairs, diagnostics)``. Nie modyfikuje wejść.
    Podnosi ``ValueError``/``TypeError`` dla niepoprawnych danych,
    ``KeyError`` dla brakującego klucza ``cfg`` oraz ``ValueError`` przy
    naruszeniu kontroli energii lub tożsamości ``s_raw = C_const + C_rest``.
    """
    _validate_cfg(cfg)

    maps_arr = np.asarray(maps)
    if maps_arr.ndim != 4:
        raise ValueError("maps musi mieć kształt (N, C, H, W)")
    if not np.issubdtype(maps_arr.dtype, np.floating):
        raise TypeError("maps musi być tablicą zmiennoprzecinkową")
    if not np.all(np.isfinite(maps_arr)):
        raise ValueError("maps zawiera NaN lub Inf")
    if min(maps_arr.shape) <= 0:
        raise ValueError("wymiary maps muszą być dodatnie")

    pairs_arr = np.asarray(pair_ids)
    if pairs_arr.ndim != 2 or pairs_arr.shape[1] != 2:
        raise ValueError("pair_ids musi mieć kształt (M, 2)")
    if not np.issubdtype(pairs_arr.dtype, np.integer):
        raise TypeError("pair_ids musi być tablicą liczb całkowitych")

    pairs_arr = np.array(pairs_arr, dtype=np.int64, copy=True)
    n_obs, n_channels, height, width = maps_arr.shape
    n_positions = height * width
    n_pairs = pairs_arr.shape[0]

    c_idx = pairs_arr[:, 0]
    d_idx = pairs_arr[:, 1]
    if n_pairs:
        if np.any(c_idx < 0) or np.any(d_idx >= n_channels):
            raise ValueError("indeks kanału w pair_ids poza zakresem [0, C)")
        if not np.all(c_idx < d_idx):
            raise ValueError("pair_ids musi spełniać c < d")
        if len({(int(a), int(b)) for a, b in pairs_arr}) != n_pairs:
            raise ValueError("pair_ids zawiera powtórzone pary")

    zero_thr = float(cfg["norm_zero_threshold"])
    near_thr = float(cfg["near_constant_relative_norm"])
    energy_rel_tol = float(cfg["energy_relative_tolerance"])
    energy_abs_tol = float(cfg["energy_absolute_tolerance"])
    identity_atol = float(cfg["identity_atol"])

    u = np.array(maps_arr, dtype=np.float64, copy=True)
    flat = u.reshape(n_obs, n_channels, n_positions).transpose(1, 0, 2)
    flat = flat.reshape(n_channels, n_obs * n_positions)

    norm_raw_sq = np.einsum("cp,cp->c", flat, flat)
    norm_raw = np.sqrt(norm_raw_sq)
    mu = flat.mean(axis=1)

    image_mean = u.mean(axis=(2, 3))  # m_c(x), kształt (N, C)
    image_dev = image_mean - mu[None, :]  # m_c(x) - mu_c

    centered_global = u - mu[None, :, None, None]
    centered_spatial = u - image_mean[:, :, None, None]

    glob_flat = centered_global.reshape(n_obs, n_channels, n_positions)
    glob_flat = glob_flat.transpose(1, 0, 2).reshape(n_channels, -1)
    spat_flat = centered_spatial.reshape(n_obs, n_channels, n_positions)
    spat_flat = spat_flat.transpose(1, 0, 2).reshape(n_channels, -1)

    norm_global_sq = np.einsum("cp,cp->c", glob_flat, glob_flat)
    norm_spatial_sq = np.einsum("cp,cp->c", spat_flat, spat_flat)
    norm_global = np.sqrt(norm_global_sq)
    norm_spatial = np.sqrt(norm_spatial_sq)

    energy_const = n_obs * n_positions * mu * mu
    energy_image = n_positions * np.sum(image_dev * image_dev, axis=0)
    energy_spatial = norm_spatial_sq
    energy_total = norm_raw_sq
    energy_sum = energy_const + energy_image + energy_spatial

    energy_err = np.abs(energy_total - energy_sum)
    big = energy_total > _SMALL_ENERGY
    if np.any(big):
        rel_err = energy_err[big] / energy_total[big]
        if np.any(rel_err > energy_rel_tol):
            raise ValueError("kontrola energii: błąd względny przekracza tolerancję")
        max_energy_relative_error = float(np.max(rel_err))
    else:
        max_energy_relative_error = 0.0
    if np.any(~big):
        if np.any(energy_err[~big] > energy_abs_tol):
            raise ValueError("kontrola energii: błąd bezwzględny przekracza tolerancję")

    defined_channel = norm_raw > zero_thr
    frac_const = np.full(n_channels, np.nan)
    frac_image = np.full(n_channels, np.nan)
    frac_spatial = np.full(n_channels, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_const[defined_channel] = (
            energy_const[defined_channel] / energy_total[defined_channel]
        )
        frac_image[defined_channel] = (
            energy_image[defined_channel] / energy_total[defined_channel]
        )
        frac_spatial[defined_channel] = (
            energy_spatial[defined_channel] / energy_total[defined_channel]
        )

    near_zero_global = norm_global <= near_thr * norm_raw
    near_zero_spatial = norm_spatial <= near_thr * norm_raw

    s_raw = np.full(n_pairs, np.nan)
    s_global = np.full(n_pairs, np.nan)
    s_spatial = np.full(n_pairs, np.nan)
    c_const = np.full(n_pairs, np.nan)
    c_rest = np.full(n_pairs, np.nan)

    raw_ok = (norm_raw[c_idx] > zero_thr) & (norm_raw[d_idx] > zero_thr)
    glob_ok = (norm_global[c_idx] > zero_thr) & (norm_global[d_idx] > zero_thr)
    spat_ok = (norm_spatial[c_idx] > zero_thr) & (norm_spatial[d_idx] > zero_thr)

    if n_pairs:
        raw_dot = np.einsum("mp,mp->m", flat[c_idx], flat[d_idx])
        glob_dot = np.einsum("mp,mp->m", glob_flat[c_idx], glob_flat[d_idx])
        spat_dot = np.einsum("mp,mp->m", spat_flat[c_idx], spat_flat[d_idx])

        denom_raw = norm_raw[c_idx] * norm_raw[d_idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            s_raw[raw_ok] = raw_dot[raw_ok] / denom_raw[raw_ok]
            c_const[raw_ok] = (
                n_obs * n_positions * mu[c_idx[raw_ok]] * mu[d_idx[raw_ok]]
                / denom_raw[raw_ok]
            )
            c_rest[raw_ok] = glob_dot[raw_ok] / denom_raw[raw_ok]
            s_global[glob_ok] = glob_dot[glob_ok] / (
                norm_global[c_idx[glob_ok]] * norm_global[d_idx[glob_ok]]
            )
            s_spatial[spat_ok] = spat_dot[spat_ok] / (
                norm_spatial[c_idx[spat_ok]] * norm_spatial[d_idx[spat_ok]]
            )

    s_raw, clips_raw = _clip_cosine(s_raw)
    s_global, clips_global = _clip_cosine(s_global)
    s_spatial, clips_spatial = _clip_cosine(s_spatial)
    cosine_clip_count = clips_raw + clips_global + clips_spatial

    identity_mask = raw_ok & np.isfinite(s_raw) & np.isfinite(c_const) & np.isfinite(c_rest)
    if np.any(identity_mask):
        identity_err = np.abs(s_raw[identity_mask] - (c_const[identity_mask] + c_rest[identity_mask]))
        max_cosine_identity_abs_error = float(np.max(identity_err))
        if max_cosine_identity_abs_error > identity_atol:
            raise ValueError(
                "kontrola tożsamości: |s_raw - (C_const + C_rest)| przekracza identity_atol"
            )
    else:
        max_cosine_identity_abs_error = 0.0

    delta_global = s_global - s_raw
    delta_spatial = s_spatial - s_raw

    channels = {
        "c": np.arange(n_channels, dtype=np.int64),
        "mean": mu,
        "norm_raw": norm_raw,
        "norm_global": norm_global,
        "norm_spatial": norm_spatial,
        "frac_const": frac_const,
        "frac_image": frac_image,
        "frac_spatial": frac_spatial,
        "near_zero_global": near_zero_global,
        "near_zero_spatial": near_zero_spatial,
    }

    pairs = {
        "c": c_idx.copy(),
        "d": d_idx.copy(),
        "s_raw": s_raw,
        "s_global": s_global,
        "s_spatial": s_spatial,
        "delta_global": delta_global,
        "delta_spatial": delta_spatial,
        "C_const": c_const,
        "C_rest": c_rest,
    }

    diagnostics = {
        "max_energy_relative_error": float(max_energy_relative_error),
        "max_cosine_identity_abs_error": float(max_cosine_identity_abs_error),
        "cosine_clip_count": int(cosine_clip_count),
        "undefined_raw_pairs": int(np.count_nonzero(~raw_ok)),
        "undefined_global_pairs": int(np.count_nonzero(~glob_ok)),
        "undefined_spatial_pairs": int(np.count_nonzero(~spat_ok)),
        "zero_raw_channels": int(np.count_nonzero(norm_raw <= zero_thr)),
        "zero_global_channels": int(np.count_nonzero(norm_global <= zero_thr)),
        "zero_spatial_channels": int(np.count_nonzero(norm_spatial <= zero_thr)),
        "near_zero_global_channels": int(np.count_nonzero(near_zero_global)),
        "near_zero_spatial_channels": int(np.count_nonzero(near_zero_spatial)),
    }

    return channels, pairs, diagnostics
