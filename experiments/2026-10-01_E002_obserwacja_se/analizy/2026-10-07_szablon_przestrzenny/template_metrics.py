"""Rozkład szablonu przestrzennego i odpowiedzi zmiennej między obrazami (E002).

Cel
---
Czysta funkcja NumPy ``decompose_templates`` rozkłada zapisane mapy aktywacji
kanałów (bez SE i po SE, h albo z) na cztery ortogonalne składowe: stałą,
obrazową, stały wzór przestrzenny (szablon) i interakcję zależną od obrazu.
Liczy dla wybranych par kanałów cosinusy surowy, szablonu, odpowiedzi zmiennej
i interakcji oraz udziały energii. Służy do kontroli, czy wysoki surowy
cosinus par wybranych w E002 wynika głównie ze wspólnego szablonu.

Eksperyment: **2026-10-01_E002_obserwacja_se**.
Analiza: ``2026-10-07_szablon_przestrzenny`` (post hoc, protokół frozen v1.0).
Pełny opis: lokalny ``OPIS.md`` (``analizy/2026-10-07_szablon_przestrzenny/OPIS.md``)
oraz ``../../OPIS.md``
(``experiments/2026-10-01_E002_obserwacja_se/OPIS.md``).

Wejścia
-------
``maps`` : tablica ``(N, C, H, W)`` zmiennoprzecinkowa, skończona. To mapy
    danego podzbioru (np. 500 obrazów), w kolejności odpowiadającej parom.
    Obliczenia wykonujemy w FP64; wejście nie jest modyfikowane.
``pair_ids`` : tablica ``(M, 2)`` liczb całkowitych, ``c < d``, bez powtórzeń,
    ``0 <= c, d < C``.
``cfg`` : dict z kluczami ``norm_zero_threshold``,
    ``near_constant_relative_norm``, ``identity_atol``,
    ``energy_relative_tolerance``, ``energy_absolute_tolerance``.

Wyjścia
-------
``(channels, pairs, diagnostics)``:
``channels`` : dict tablic długości ``C`` — ``c``, ``mean``, ``norm_raw``,
    ``norm_template``, ``norm_dynamic``, ``norm_interaction``, ``frac_const``,
    ``frac_image``, ``frac_spatial_template``, ``frac_interaction``,
    ``frac_template``, ``frac_dynamic``, ``frac_interaction_in_spatial``,
    ``near_zero_dynamic``, ``near_zero_interaction``.
``pairs`` : dict tablic długości ``M`` — ``c``, ``d``, ``s_raw``,
    ``s_template``, ``s_dynamic``, ``s_interaction``, ``delta_dynamic``,
    ``delta_interaction``, ``C_template``, ``C_dynamic``.
``diagnostics`` : dict gotowy do serializacji JSON (tylko ``int``/``float``)
    z błędami kontroli i licznikami.

Model rozkładu (FP64, na bezpośrednio wycentrowanych tensorach)
---------------------------------------------------------------
``mu = mean_{x,p} U``; ``I(x) = mean_p U(x, p) - mu``;
``S(p) = mean_x U(x, p) - mu``; ``R(x, p) = U(x, p) - mu - I(x) - S(p)``.
Szablon ``T(p) = mean_x U(x, p) = mu + S(p)``; odpowiedź zmienna
``D(x, p) = U(x, p) - T(p) = I(x) + R(x, p)``. W zbalansowanej tabeli
``N x P`` składowe są ortogonalne i
``E = E_const + E_image + E_spatial_template + E_interaction``, gdzie
``E_const = N*P*mu^2``, ``E_image = P*sum_x I(x)^2``,
``E_spatial_template = N*sum_p S(p)^2``, ``E_interaction = sum R^2``.
``E_template = E_const + E_spatial_template`` (stała + szablon),
``E_dynamic = E_image + E_interaction`` (odpowiedź zmienna).

Norma powtórzonego szablonu to ``sqrt(N) * ||T||_P`` (``^``), a surowy
cosinus rozkłada się dokładnie na wkłady szablonu i odpowiedzi zmiennej:
``s_raw = C_template + C_dynamic``, gdzie
``C_template = N * dot(T_c, T_d) / (||U_c|| ||U_d||)`` oraz
``C_dynamic = dot(D_c, D_d) / (||U_c|| ||U_d||)``.

Uwagi numeryczne
----------------
Centrowane tensory ``D`` i ``R`` liczymy wprost (``U - T`` oraz ``D - I``),
a nie przez różnice Gramów, więc nie tracimy cyfr znaczących. Dzielenie
udziałów przez ``E_raw`` wykonujemy tylko tam, gdzie ``norm_raw`` przekracza
``cfg['norm_zero_threshold']``; pozostałe udziały to ``NaN``. Udział
``frac_interaction_in_spatial`` jest określony, gdy
``sqrt(E_spatial_template + E_interaction)`` przekracza próg. Cosinusy par
o normie ``<=`` progu są ``NaN`` (nie 0). Cosinusy przycinamy do ``[-1, 1]``
tylko z powodu zaokrągleń, z licznikiem korekt. Funkcja nie interpretuje
wyniku naukowo, nie wykonuje wejścia/wyjścia, nie dotyka GPU ani checkpointów.
"""

from __future__ import annotations

import numpy as np

EXPERIMENT_ID = "2026-10-01_E002_obserwacja_se"
ANALYSIS_ID = "2026-10-07_szablon_przestrzenny"
OPIS_PATH = "OPIS.md"
EXPERIMENT_OPIS_PATH = "../../OPIS.md"

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


def _energy_check(
    error: np.ndarray,
    total: np.ndarray,
    label: str,
    rel_tol: float,
    abs_tol: float,
) -> float:
    """Kontroluje błąd energii i zwraca maksymalny błąd względny dla dużych E.

    Dla ``E_total > _SMALL_ENERGY`` wymagamy błędu względnego ``<= rel_tol``,
    a poniżej progu błędu bezwzględnego ``<= abs_tol``. Naruszenie przerywa
    obliczenia ``ValueError``.
    """
    big = total > _SMALL_ENERGY
    max_rel = 0.0
    if np.any(big):
        rel = error[big] / total[big]
        max_rel = float(np.max(rel))
        if np.any(rel > rel_tol):
            raise ValueError(f"kontrola energii ({label}): błąd względny przekracza tolerancję")
    if np.any(~big):
        if np.any(error[~big] > abs_tol):
            raise ValueError(f"kontrola energii ({label}): błąd bezwzględny przekracza tolerancję")
    return max_rel


def decompose_templates(maps, pair_ids, cfg):
    """Rozkłada mapy E002 na szablon i odpowiedź zmienną; patrz docstring modułu.

    Zwraca ``(channels, pairs, diagnostics)``. Nie modyfikuje wejść.
    Podnosi ``ValueError``/``TypeError`` dla niepoprawnych danych,
    ``KeyError`` dla brakującego klucza ``cfg`` oraz ``ValueError`` przy
    naruszeniu kontroli energii lub tożsamości ``s_raw = C_template + C_dynamic``.
    Tablice wyjściowe mogą zawierać ``NaN`` tam, gdzie składowa jest
    nieokreślona.
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
    identity_atol = float(cfg["identity_atol"])
    energy_rel_tol = float(cfg["energy_relative_tolerance"])
    energy_abs_tol = float(cfg["energy_absolute_tolerance"])

    u = np.array(maps_arr, dtype=np.float64, copy=True)
    u = u.reshape(n_obs, n_channels, n_positions)

    mu = u.mean(axis=(0, 2))
    template = u.mean(axis=0)
    image = u.mean(axis=2) - mu[None, :]
    spatial = template - mu[:, None]

    dynamic = u - template[None, :, :]
    interaction = dynamic - image[:, :, None]

    energy_total = np.einsum("ncp,ncp->c", u, u)
    energy_const = n_obs * n_positions * mu * mu
    energy_image = n_positions * np.sum(image * image, axis=0)
    energy_spatial = n_obs * np.sum(spatial * spatial, axis=1)
    energy_interaction = np.einsum("ncp,ncp->c", interaction, interaction)

    energy_template = energy_const + energy_spatial
    energy_dynamic = energy_image + energy_interaction

    err_sum4 = np.abs(energy_total - (energy_const + energy_image + energy_spatial + energy_interaction))
    err_td = np.abs(energy_total - (energy_template + energy_dynamic))
    max_energy_relative_error = _energy_check(
        err_sum4, energy_total, "sum4", energy_rel_tol, energy_abs_tol
    )
    max_template_dynamic_energy_relative_error = _energy_check(
        err_td, energy_total, "template+dynamic", energy_rel_tol, energy_abs_tol
    )

    norm_raw = np.sqrt(energy_total)
    norm_template = np.sqrt(n_obs * np.sum(template * template, axis=1))
    norm_dynamic = np.sqrt(np.einsum("ncp,ncp->c", dynamic, dynamic))
    norm_interaction = np.sqrt(energy_interaction)

    frac_const = np.full(n_channels, np.nan)
    frac_image = np.full(n_channels, np.nan)
    frac_spatial_template = np.full(n_channels, np.nan)
    frac_interaction = np.full(n_channels, np.nan)
    frac_template = np.full(n_channels, np.nan)
    frac_dynamic = np.full(n_channels, np.nan)
    frac_interaction_in_spatial = np.full(n_channels, np.nan)

    defined = norm_raw > zero_thr
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_const[defined] = energy_const[defined] / energy_total[defined]
        frac_image[defined] = energy_image[defined] / energy_total[defined]
        frac_spatial_template[defined] = energy_spatial[defined] / energy_total[defined]
        frac_interaction[defined] = energy_interaction[defined] / energy_total[defined]
        frac_template[defined] = energy_template[defined] / energy_total[defined]
        frac_dynamic[defined] = energy_dynamic[defined] / energy_total[defined]

    in_spatial_denom = energy_spatial + energy_interaction
    in_spatial_ok = np.sqrt(in_spatial_denom) > zero_thr
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_interaction_in_spatial[in_spatial_ok] = (
            energy_interaction[in_spatial_ok] / in_spatial_denom[in_spatial_ok]
        )

    near_zero_dynamic = norm_dynamic <= near_thr * norm_raw
    near_zero_interaction = norm_interaction <= near_thr * norm_raw

    s_raw = np.full(n_pairs, np.nan)
    s_template = np.full(n_pairs, np.nan)
    s_dynamic = np.full(n_pairs, np.nan)
    s_interaction = np.full(n_pairs, np.nan)
    c_template = np.full(n_pairs, np.nan)
    c_dynamic = np.full(n_pairs, np.nan)

    raw_ok = (norm_raw[c_idx] > zero_thr) & (norm_raw[d_idx] > zero_thr)
    template_ok = (norm_template[c_idx] > zero_thr) & (norm_template[d_idx] > zero_thr)
    dynamic_ok = (norm_dynamic[c_idx] > zero_thr) & (norm_dynamic[d_idx] > zero_thr)
    interaction_ok = (norm_interaction[c_idx] > zero_thr) & (norm_interaction[d_idx] > zero_thr)

    if n_pairs:
        flat_u = u.transpose(1, 0, 2).reshape(n_channels, -1)
        flat_d = dynamic.transpose(1, 0, 2).reshape(n_channels, -1)
        flat_r = interaction.transpose(1, 0, 2).reshape(n_channels, -1)

        dot_raw = np.einsum("mp,mp->m", flat_u[c_idx], flat_u[d_idx])
        dot_dynamic = np.einsum("mp,mp->m", flat_d[c_idx], flat_d[d_idx])
        dot_interaction = np.einsum("mp,mp->m", flat_r[c_idx], flat_r[d_idx])
        dot_template = np.einsum("mp,mp->m", template[c_idx], template[d_idx])

        denom_raw = norm_raw[c_idx] * norm_raw[d_idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            s_raw[raw_ok] = dot_raw[raw_ok] / denom_raw[raw_ok]
            c_template[raw_ok] = n_obs * dot_template[raw_ok] / denom_raw[raw_ok]
            c_dynamic[raw_ok] = dot_dynamic[raw_ok] / denom_raw[raw_ok]
            s_template[template_ok] = (
                n_obs * dot_template[template_ok]
                / (norm_template[c_idx[template_ok]] * norm_template[d_idx[template_ok]])
            )
            s_dynamic[dynamic_ok] = (
                dot_dynamic[dynamic_ok]
                / (norm_dynamic[c_idx[dynamic_ok]] * norm_dynamic[d_idx[dynamic_ok]])
            )
            s_interaction[interaction_ok] = (
                dot_interaction[interaction_ok]
                / (norm_interaction[c_idx[interaction_ok]] * norm_interaction[d_idx[interaction_ok]])
            )

    s_raw, clips_raw = _clip_cosine(s_raw)
    s_template, clips_template = _clip_cosine(s_template)
    s_dynamic, clips_dynamic = _clip_cosine(s_dynamic)
    s_interaction, clips_interaction = _clip_cosine(s_interaction)
    cosine_clip_count = clips_raw + clips_template + clips_dynamic + clips_interaction

    identity_mask = (
        raw_ok & np.isfinite(s_raw) & np.isfinite(c_template) & np.isfinite(c_dynamic)
    )
    if np.any(identity_mask):
        identity_err = np.abs(
            s_raw[identity_mask] - (c_template[identity_mask] + c_dynamic[identity_mask])
        )
        max_cosine_identity_abs_error = float(np.max(identity_err))
        if max_cosine_identity_abs_error > identity_atol:
            raise ValueError(
                "kontrola tożsamości: |s_raw - (C_template + C_dynamic)| przekracza identity_atol"
            )
    else:
        max_cosine_identity_abs_error = 0.0

    delta_dynamic = s_dynamic - s_raw
    delta_interaction = s_interaction - s_raw

    channels = {
        "c": np.arange(n_channels, dtype=np.int64),
        "mean": mu,
        "norm_raw": norm_raw,
        "norm_template": norm_template,
        "norm_dynamic": norm_dynamic,
        "norm_interaction": norm_interaction,
        "frac_const": frac_const,
        "frac_image": frac_image,
        "frac_spatial_template": frac_spatial_template,
        "frac_interaction": frac_interaction,
        "frac_template": frac_template,
        "frac_dynamic": frac_dynamic,
        "frac_interaction_in_spatial": frac_interaction_in_spatial,
        "near_zero_dynamic": near_zero_dynamic,
        "near_zero_interaction": near_zero_interaction,
    }

    pairs = {
        "c": c_idx.copy(),
        "d": d_idx.copy(),
        "s_raw": s_raw,
        "s_template": s_template,
        "s_dynamic": s_dynamic,
        "s_interaction": s_interaction,
        "delta_dynamic": delta_dynamic,
        "delta_interaction": delta_interaction,
        "C_template": c_template,
        "C_dynamic": c_dynamic,
    }

    diagnostics = {
        "max_energy_relative_error": float(max_energy_relative_error),
        "max_template_dynamic_energy_relative_error": float(
            max_template_dynamic_energy_relative_error
        ),
        "max_cosine_identity_abs_error": float(max_cosine_identity_abs_error),
        "cosine_clip_count": int(cosine_clip_count),
        "undefined_raw_pairs": int(np.count_nonzero(~raw_ok)),
        "undefined_template_pairs": int(np.count_nonzero(~template_ok)),
        "undefined_dynamic_pairs": int(np.count_nonzero(~dynamic_ok)),
        "undefined_interaction_pairs": int(np.count_nonzero(~interaction_ok)),
        "zero_raw_channels": int(np.count_nonzero(norm_raw <= zero_thr)),
        "zero_template_channels": int(np.count_nonzero(norm_template <= zero_thr)),
        "zero_dynamic_channels": int(np.count_nonzero(norm_dynamic <= zero_thr)),
        "zero_interaction_channels": int(np.count_nonzero(norm_interaction <= zero_thr)),
        "near_zero_dynamic_channels": int(np.count_nonzero(near_zero_dynamic)),
        "near_zero_interaction_channels": int(np.count_nonzero(near_zero_interaction)),
    }

    return channels, pairs, diagnostics


def _selfcheck() -> None:
    """Podstawowy test algebraiczny na sztucznych mapach (bez I/O i danych)."""
    cfg = {
        "norm_zero_threshold": 1e-12,
        "near_constant_relative_norm": 1e-6,
        "identity_atol": 1e-10,
        "energy_relative_tolerance": 1e-10,
        "energy_absolute_tolerance": 1e-24,
    }
    template = np.array([-10.0, 10.0])
    perturb = np.array([[1.0, -1.0], [-1.0, 1.0]])
    maps = np.empty((2, 2, 1, 2))
    maps[:, 0, 0, :] = template[None, :] + perturb
    maps[:, 1, 0, :] = template[None, :] - perturb
    pair_ids = np.array([[0, 1]])
    saved = maps.copy()

    channels, pairs, _ = decompose_templates(maps, pair_ids, cfg)
    assert np.array_equal(maps, saved), "wejście zostało zmodyfikowane"
    assert np.isclose(pairs["s_raw"][0], 99.0 / 101.0), "surowy cosinus != 99/101"
    assert np.isclose(pairs["s_dynamic"][0], -1.0), "cosinus D != -1"
    assert np.isclose(pairs["s_interaction"][0], -1.0), "cosinus R != -1"
    assert np.isclose(channels["frac_template"][0], 100.0 / 101.0), "udział szablonu != 100/101"
    assert np.isclose(channels["frac_interaction"][0], 1.0 / 101.0), "udział interakcji != 1/101"
    frac_sum = (
        channels["frac_const"]
        + channels["frac_image"]
        + channels["frac_spatial_template"]
        + channels["frac_interaction"]
    )
    assert np.allclose(frac_sum, 1.0, atol=1e-12), "suma udziałów energii != 1"

    same = np.empty((2, 2, 1, 2))
    same[:, 0, 0, :] = template
    same[:, 1, 0, :] = template
    same_channels, same_pairs, _ = decompose_templates(same, pair_ids, cfg)
    assert np.isnan(same_pairs["s_dynamic"][0]), "zerowe D nie dało NaN"
    assert np.isnan(same_pairs["s_interaction"][0]), "zerowe R nie dało NaN"
    assert np.isclose(same_pairs["s_raw"][0], 1.0), "surowy cosinus szablonu != 1"
    same_frac_sum = (
        same_channels["frac_const"]
        + same_channels["frac_image"]
        + same_channels["frac_spatial_template"]
        + same_channels["frac_interaction"]
    )
    assert np.allclose(same_frac_sum, 1.0, atol=1e-12), "suma udziałów energii != 1"

    print(
        "TEMPLATE_SELFCHECK=passed; raw=99/101, D/R cos=-1, frac_template=100/101; "
        "zero D -> NaN; brak realnych aktywacji"
    )


if __name__ == "__main__":
    _selfcheck()
