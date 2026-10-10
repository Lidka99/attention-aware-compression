"""E003: wybór rozłącznych par kanałów do scalenia (czyste NumPy, tylko CPU).

Eksperyment: **2026-10-09_E003_kompresja_10proc_parametrow**.
Pełny opis zakresu i wspólnego operatora: ``OPIS.md`` w tym samym katalogu.
Zamrożoną wersję protokołu (przygotowywaną osobno przez agenta głównego)
oznacza ``PROTOKOL_v1.md``.

Cel
---
Dla każdego bloku MBConv wybrać z wagowego planu ``rows`` wymaganą liczbę
**rozłącznych** par kanałów ``(c, d)``, ``c < d``, oraz policzyć współczynniki
kompensacji ``alpha`` (wejście SE) i ``beta`` (pełna mapa po SE). Funkcja
obsługuje trzy metody na tych samych danych: ``PRE`` (rangi według cosinusa
przed SE), ``POST`` (rangi według cosinusa po SE) i ``RANDOM`` (kontrola
losowa). Nie trenuje, nie zmienia wag, nie czyta obrazów ani checkpointów,
nie używa GPU i nie zapisuje żadnych plików. ``BASE`` nie ma wyboru par,
więc nie występuje w wyniku.

Wejścia
-------
``moments`` : ``dict`` ``blockpath -> dict`` z kluczami ``h``, ``z``, ``g``.
    Każda wartość to macierz Grama ``C x C`` typu zmiennoprzecinkowego, którą
    funkcja kopiuje do ``float64``. To **sumy** iloczynów, nie macierze
    scentrowane: ``S_h[c,d] = sum_t h_c[t] h_d[t]``, ``S_z`` analogicznie,
    ``S_g[c,d] = sum_b g_c[b] g_d[b]`` gdzie ``g`` jest rzeczywistym wejściem
    ``fc1`` SE. Macierze muszą być skończone i symetryczne (z dokładnością do
    zaokrągleń). Wejście nie jest modyfikowane.

``rows`` : lista wierszy planu struktury w kolejności z ``PLAN_STRUKTURY.json``.
    Każdy wiersz to ``dict`` z co najmniej:
    - ``block`` (``str``) — ścieżka bloku, klucz w ``moments``,
    - ``remove_count`` (``int``) — liczba par do wybrania w tym bloku; jedna
      para usuwa jeden kanał, bo reprezentanta zostawiamy,
    - opcjonalnie ``expanded_channels`` (``int``) — kontrola szerokości ``C``.

``cfg`` : ``dict`` z kluczami:
    - ``norm_zero_threshold`` (``float``) — próg normy; kanał kwalifikuje się
      tylko wtedy, gdy ``sqrt(diag(S_h))``, ``sqrt(diag(S_z))`` oraz
      ``sqrt(diag(S_g))`` są **wszystkie ściśle większe** od progu,
    - ``random_selection_seed`` (``int``) — seed metody RANDOM,
    - ``random_block_seed_stride`` (``int``) — krok seeda między blokami.

Wyjścia
-------
``dict`` o kluczach ``"PRE"``, ``"POST"``, ``"RANDOM"``. Pod każdym kluczem
``dict`` ``blockpath -> list`` wpisów, w kolejności bloków ``rows``, a wpisy
posortowane rosnąco po ``(c, d)``. Każdy wpis to ``dict`` o kluczach
``c``, ``d``, ``alpha``, ``beta``, ``score_pre``, ``score_post``:
    - ``c``, ``d`` (``int``) — oryginalne indeksy kanałów, ``c < d``;
      reprezentantem jest mniejszy indeks (drugi kanał usuwamy),
    - ``alpha = S_g[c,d] / S_g[c,c]`` oraz ``beta = S_z[c,d] / S_z[c,c]``
      (bez przycinania i bez wyrazu wolnego),
    - ``score_pre``, ``score_post`` — cosinus pary odpowiednio z ``h`` i ``z``,
      przycięty do ``[-1, 1]``.

Algorytm
--------
Kwalifikacja jest **wspólna dla wszystkich metod**: ten sam zbiór ``eligible``
(wszystkie trzy normy diagonalne powyżej progu). Cosinusy liczymy jako iloczyn
skalarny z macierzy Grama podzielony przez iloczyn norm diagonalnych, tylko
dla górnego trójkąta ``c < d``. ``PRE`` sortuje kandydatów malejąco po
``score_pre`` (remisy po ``c, d``), ``POST`` malejąco po ``score_post``; obie
metody zachłannie biorą pary rozłączne, aż do ``remove_count`` par. ``RANDOM``
permutuje kanały kwalifikowane generatorem
``default_rng(random_selection_seed + random_block_seed_stride * blockidx)``
(``blockidx`` liczony od 0 w kolejności ``rows``), bierze pierwsze
``2 * remove_count`` kanałów i tworzy kolejne pary, orientując każdą jako
``c < d``. Wejścia są tylko czytane; funkcja nie mutuje ``moments``, ``rows``
ani ``cfg``.

Błędy
-----
Zgłaszamy wyjątek dla: malformed/niesymetrycznej/nieskończonej macierzy Grama,
braku bloku w ``moments``, zbyt małej liczby kanałów kwalifikowanych
(``< 2 * remove_count``) oraz zbyt małej liczby rozłącznych par. Nie zmieniamy
wtedy automatycznie budżetu.

Uwagi
-----
``selection_settings`` zwraca znormalizowane ustawienia, które launcher
(``main``) zapisuje w manifeście. Wynik jest deterministyczny i powtarzalny:
przy tych samych wejściach zwraca identyczny ``dict``.

Sprawdzenie na CPU: ``python -B selection.py`` wykonuje kontrole bez zapisu
plików (bez ``__pycache__``).
"""

from __future__ import annotations

import numpy as np

EXPERIMENT_ID = "2026-10-09_E003_kompresja_10proc_parametrow"
OPIS_PATH = "OPIS.md"
PROTOKOL_V1_PATH = "PROTOKOL_v1.md"

SELECTION_MODES = ("PRE", "POST", "RANDOM")
PAIR_ENTRY_KEYS = ("c", "d", "alpha", "beta", "score_pre", "score_post")
_GRAM_KEYS = ("h", "z", "g")


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _float_setting(cfg, key):
    if key not in cfg:
        raise KeyError(f"cfg: brak wymaganego klucza '{key}'")
    return float(cfg[key])


def _int_setting(cfg, key):
    if key not in cfg:
        raise KeyError(f"cfg: brak wymaganego klucza '{key}'")
    return int(cfg[key])


def _as_gram(value, name):
    """Waliduje macierz Grama: typ zmiennoprzecinkowy, kwadratowa, finite, symetryczna."""
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(
            f"{name}: oczekiwano macierzy zmiennoprzecinkowej, jest {arr.dtype}"
        )
    arr = arr.astype(np.float64, copy=True)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name}: oczekiwano kwadratowej macierzy C x C, jest {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: wykryto NaN/Inf w macierzy Grama")
    if not np.allclose(arr, arr.T, rtol=1e-9, atol=1e-12):
        raise ValueError(f"{name}: macierz Grama nie jest symetryczna")
    return arr


def _diag_norm(gram):
    """Norma kanału z sumy kwadratów: ``sqrt(diag(G))``, przycięta od dołu do 0."""
    return np.sqrt(np.maximum(np.diag(gram), 0.0))


def _cosine_matrix(gram, norm):
    """Cosinusy wszystkich par z macierzy Grama i norm; wynik przycięty do [-1, 1]."""
    denom = np.outer(norm, norm)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = gram / denom
    return np.clip(cosine, -1.0, 1.0)


def _greedy_pairs(width, ci, di, scores, n_pairs, block):
    """Zachłannie wybiera ``n_pairs`` rozłącznych par malejąco po ``scores``.

    Remisy rozstrzyga porządek ``(c, d)``. Zwraca pary posortowane rosnąco po
    ``(c, d)``. Gdy nie da się wybrać wymaganej liczby par, zgłasza błąd.
    """
    order = np.lexsort((di, ci, -scores))
    used = np.zeros(width, dtype=bool)
    chosen = []
    for index in order:
        c = int(ci[index])
        d = int(di[index])
        if used[c] or used[d]:
            continue
        used[c] = True
        used[d] = True
        chosen.append((c, d))
        if len(chosen) == n_pairs:
            break
    if len(chosen) < n_pairs:
        raise RuntimeError(
            f"{block}: za mało rozłącznych par ({len(chosen)} < {n_pairs})"
        )
    return sorted(chosen)


def choose_groups(moments, rows, cfg):
    """Wybiera pary kanałów i współczynniki dla metod PRE, POST i RANDOM.

    Parametry i znaczenie wyniku opisano w docstringu modułu. Funkcja jest
    czysta: nie modyfikuje ``moments``, ``rows`` ani ``cfg``.

    Returns
    -------
    dict
        ``{"PRE": {block: [entry, ...]}, "POST": {...}, "RANDOM": {...}}``,
        gdzie ``entry`` ma klucze ``c``, ``d``, ``alpha``, ``beta``,
        ``score_pre``, ``score_post``.
    """
    norm_threshold = _float_setting(cfg, "norm_zero_threshold")
    if norm_threshold < 0.0:
        raise ValueError("cfg['norm_zero_threshold'] musi być >= 0")
    random_seed = _int_setting(cfg, "random_selection_seed")
    random_stride = _int_setting(cfg, "random_block_seed_stride")

    if not isinstance(moments, dict):
        raise TypeError("moments: oczekiwano dict blockpath -> {h, z, g}")
    if not isinstance(rows, (list, tuple)) or len(rows) == 0:
        raise ValueError("rows: oczekiwano niepustej listy wierszy planu")

    result = {mode: {} for mode in SELECTION_MODES}
    seen_blocks = set()
    for block_idx, row in enumerate(rows):
        if not isinstance(row, dict) or "block" not in row or "remove_count" not in row:
            raise ValueError("rows: każdy wiersz musi mieć 'block' i 'remove_count'")
        block = row["block"]
        if not isinstance(block, str) or not block:
            raise ValueError("rows: 'block' musi być niepustym łańcuchem")
        if block in seen_blocks:
            raise ValueError(f"rows: zduplikowany blok {block!r}")
        seen_blocks.add(block)

        n_pairs = int(row["remove_count"])
        if n_pairs < 0:
            raise ValueError(f"{block}: remove_count musi być >= 0")

        if block not in moments:
            raise KeyError(f"moments: brak bloku {block!r}")
        entry = moments[block]
        if not isinstance(entry, dict) or any(key not in entry for key in _GRAM_KEYS):
            raise ValueError(f"moments[{block!r}]: wymagane klucze {_GRAM_KEYS}")

        gram = {
            key: _as_gram(entry[key], f"moments[{block!r}][{key!r}]")
            for key in _GRAM_KEYS
        }
        width = gram["h"].shape[0]
        for key in _GRAM_KEYS:
            if gram[key].shape != (width, width):
                raise ValueError(
                    f"moments[{block!r}][{key!r}]: kształt {gram[key].shape} "
                    f"niezgodny z ({width}, {width})"
                )
        if "expanded_channels" in row and int(row["expanded_channels"]) != width:
            raise ValueError(
                f"{block}: expanded_channels={row['expanded_channels']} "
                f"niezgodne z szerokością Grama {width}"
            )

        norms = {key: _diag_norm(gram[key]) for key in _GRAM_KEYS}
        eligible = (norms["h"] > norm_threshold)
        eligible &= (norms["z"] > norm_threshold)
        eligible &= (norms["g"] > norm_threshold)
        n_eligible = int(np.count_nonzero(eligible))
        if n_eligible < 2 * n_pairs:
            raise RuntimeError(
                f"{block}: kanałów kwalifikowanych {n_eligible} "
                f"< {2 * n_pairs} wymaganych do {n_pairs} par"
            )

        cosine_h = _cosine_matrix(gram["h"], norms["h"])
        cosine_z = _cosine_matrix(gram["z"], norms["z"])

        upper_c, upper_d = np.triu_indices(width, k=1)
        pair_mask = eligible[upper_c] & eligible[upper_d]
        ci = upper_c[pair_mask]
        di = upper_d[pair_mask]
        score_pre = cosine_h[ci, di]
        score_post = cosine_z[ci, di]
        if ci.size and not (
            np.all(np.isfinite(score_pre)) and np.all(np.isfinite(score_post))
        ):
            raise RuntimeError(f"{block}: niedodatnie cosinusy kwalifikowanych par")

        gram_g = gram["g"]
        gram_z = gram["z"]

        if n_pairs == 0:
            selected = {mode: [] for mode in SELECTION_MODES}
        else:
            selected = {
                "PRE": _greedy_pairs(width, ci, di, score_pre, n_pairs, block),
                "POST": _greedy_pairs(width, ci, di, score_post, n_pairs, block),
            }
            generator = np.random.default_rng(
                random_seed + random_stride * block_idx
            )
            permutation = generator.permutation(np.flatnonzero(eligible))
            drawn = permutation[: 2 * n_pairs]
            random_pairs = []
            for position in range(n_pairs):
                a = int(drawn[2 * position])
                b = int(drawn[2 * position + 1])
                random_pairs.append((a, b) if a < b else (b, a))
            selected["RANDOM"] = sorted(random_pairs)

        for mode in SELECTION_MODES:
            entries = []
            for c, d in selected[mode]:
                denominator_g = gram_g[c, c]
                denominator_z = gram_z[c, c]
                if denominator_g <= 0.0 or denominator_z <= 0.0:
                    raise RuntimeError(f"{block}: niedodatnia norma kanału w parze ({c}, {d})")
                alpha = float(gram_g[c, d] / denominator_g)
                beta = float(gram_z[c, d] / denominator_z)
                if not (np.isfinite(alpha) and np.isfinite(beta)):
                    raise RuntimeError(
                        f"{block}: niepoliczalny współczynnik alpha/beta dla ({c}, {d})"
                    )
                entries.append(
                    {
                        "c": int(c),
                        "d": int(d),
                        "alpha": alpha,
                        "beta": beta,
                        "score_pre": float(cosine_h[c, d]),
                        "score_post": float(cosine_z[c, d]),
                    }
                )
            result[mode][block] = entries

    return result


def selection_settings(cfg):
    """Zwraca znormalizowane ustawienia wyboru par do zapisania przez launcher."""
    return {
        "experiment": EXPERIMENT_ID,
        "modes": list(SELECTION_MODES),
        "pair_entry_keys": list(PAIR_ENTRY_KEYS),
        "norm_zero_threshold": _float_setting(cfg, "norm_zero_threshold"),
        "random_selection_seed": _int_setting(cfg, "random_selection_seed"),
        "random_block_seed_stride": _int_setting(cfg, "random_block_seed_stride"),
        "eligibility": "sqrt(diag(S_h)), sqrt(diag(S_z)), sqrt(diag(S_g)) all > norm_zero_threshold",
        "score_pre": "clip(<h_c,h_d>/(||h_c|| ||h_d||), -1, 1)",
        "score_post": "clip(<z_c,z_d>/(||z_c|| ||z_d||), -1, 1)",
        "alpha_fit": "S_g[c,d]/S_g[c,c]",
        "beta_fit": "S_z[c,d]/S_z[c,c]",
        "representative": "lower_original_channel_index",
    }


def _expect(exception, function):
    try:
        function()
    except exception:
        return
    raise AssertionError(f"oczekiwano {exception.__name__}")


def _self_test():
    """Kontrole CPU: wzory bezpośrednie, rozłączność, kwalifikacja, remisy, RANDOM, błędy."""
    rng = np.random.default_rng(20261009)
    samples, width = 7, 10
    h = rng.normal(size=(samples, width))
    z = rng.normal(size=(samples, width))
    g = rng.normal(size=(samples, width))
    cfg = {
        "norm_zero_threshold": 1e-12,
        "random_selection_seed": 20261009,
        "random_block_seed_stride": 1000,
    }
    rows = [{"block": "features.test.0", "remove_count": 3, "expanded_channels": width}]
    block = "features.test.0"

    def grams(hh, zz, gg):
        return {"h": hh.T @ hh, "z": zz.T @ zz, "g": gg.T @ gg}

    moments = {block: grams(h, z, g)}
    backup = {key: value.copy() for key, value in moments[block].items()}
    groups = choose_groups(moments, rows, cfg)

    assert set(groups) == set(SELECTION_MODES), "zły zestaw metod"
    assert selection_settings(cfg)["modes"] == list(SELECTION_MODES)

    for mode in SELECTION_MODES:
        assert set(groups[mode]) == {block}, "zły zestaw bloków"
        pairs = groups[mode][block]
        assert len(pairs) == 3, (mode, "zła liczba par")
        channels = []
        for e in pairs:
            assert set(e) == set(PAIR_ENTRY_KEYS), "złe klucze wpisu"
            assert e["c"] < e["d"], "kanał c musi być mniejszy od d"
            channels.extend((e["c"], e["d"]))
            c, d = e["c"], e["d"]
            direct_alpha = (g[:, c] @ g[:, d]) / (g[:, c] @ g[:, c])
            direct_beta = (z[:, c] @ z[:, d]) / (z[:, c] @ z[:, c])
            direct_pre = np.clip(
                (h[:, c] @ h[:, d]) / (np.linalg.norm(h[:, c]) * np.linalg.norm(h[:, d])),
                -1.0,
                1.0,
            )
            direct_post = np.clip(
                (z[:, c] @ z[:, d]) / (np.linalg.norm(z[:, c]) * np.linalg.norm(z[:, d])),
                -1.0,
                1.0,
            )
            assert np.isclose(e["alpha"], direct_alpha, atol=1e-12, rtol=1e-12)
            assert np.isclose(e["beta"], direct_beta, atol=1e-12, rtol=1e-12)
            assert np.isclose(e["score_pre"], direct_pre, atol=1e-12, rtol=1e-12)
            assert np.isclose(e["score_post"], direct_post, atol=1e-12, rtol=1e-12)
        assert len(set(channels)) == len(channels) == 6, (mode, "pary nie są rozłączne")

    for key, value in backup.items():
        assert np.array_equal(moments[block][key], value), "wejście zostało zmutowane"

    again = choose_groups(moments, rows, cfg)
    assert again == groups, "wynik nie jest powtarzalny"

    h0, z0, g0 = h.copy(), z.copy(), g.copy()
    h0[:, 0] = 0.0
    z0[:, 0] = 0.0
    g0[:, 0] = 0.0
    zero_groups = choose_groups({block: grams(h0, z0, g0)}, rows, cfg)
    for mode in SELECTION_MODES:
        for e in zero_groups[mode][block]:
            assert e["c"] != 0 and e["d"] != 0, "kanał niekwalifikowany trafił do pary"

    tied = np.full((width, width), 0.5)
    np.fill_diagonal(tied, 1.0)
    tie_groups = choose_groups({block: {"h": tied.copy(), "z": tied.copy(), "g": tied.copy()}}, rows, cfg)
    for mode in ("PRE", "POST"):
        chosen = [(e["c"], e["d"]) for e in tie_groups[mode][block]]
        assert chosen == [(0, 1), (2, 3), (4, 5)], (mode, "remisy nie po (c, d)", chosen)

    identity = np.eye(width)
    _expect(ValueError, lambda: choose_groups(
        {block: {"h": np.zeros((width, width - 1)), "z": identity, "g": identity}}, rows, cfg))
    _expect(TypeError, lambda: choose_groups(
        {block: {"h": np.eye(width, dtype=np.int64), "z": identity, "g": identity}}, rows, cfg))
    asymmetric = identity.copy()
    asymmetric[0, 1] = 0.3
    _expect(ValueError, lambda: choose_groups(
        {block: {"h": asymmetric, "z": identity, "g": identity}}, rows, cfg))
    non_finite = identity.copy()
    non_finite[0, 1] = non_finite[1, 0] = np.inf
    _expect(ValueError, lambda: choose_groups(
        {block: {"h": non_finite, "z": identity, "g": identity}}, rows, cfg))
    _expect(KeyError, lambda: choose_groups({}, rows, cfg))

    sparse = np.zeros((width, width))
    np.fill_diagonal(sparse[:3, :3], 1.0)
    _expect(RuntimeError, lambda: choose_groups(
        {block: {"h": sparse.copy(), "z": sparse.copy(), "g": sparse.copy()}}, rows, cfg))

    print("SELECTION_SELF_TEST=passed; modes=PRE,POST,RANDOM; checks="
          "direct_formulas,disjoint_count,zero_channel,ties,repeat,malformed_reject")


if __name__ == "__main__":
    _self_test()
