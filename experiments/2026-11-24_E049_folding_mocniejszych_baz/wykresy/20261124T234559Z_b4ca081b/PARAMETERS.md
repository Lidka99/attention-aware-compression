# E049 — figura dopasowanych parametrów: TILT vs UNIFORM (30/50/70%)

[PDF wektorowy](e049_matched_parameters_en.pdf) ·
[PNG](e049_matched_parameters_en.png) · [Generator](make_parameter_figure.py) ·
[Wyniki runu](../../../../runs/E049/eval_20261124T234559Z_b4ca081b/WYNIKI.md)

Dodatkowa figura do E049, niezależna od `e049_b10_joint_caps_en.*` (pliki
b10 pozostają bez zmian). Zbudowana **wyłącznie** z
`runs/E049/eval_20261124T234559Z_b4ca081b/per_seed.csv` i `audit.json`
(`split=benchmark`), bez GPU, nowych punktacji i zmian w `paper/`.

## Co pokazuje

Dwa panele: **A CIFAR-100**, **B Tiny ImageNet**. W każdym trzy rzędy
nominalnej redukcji parametrów **30 / 50 / 70%**; wszystkie rzędy i oba
panele mają **tę samą oś x**: `TILT − UNIFORM top-1 (pp)`, ze wspólnym
zerem. Puste kółka to seedy 42–45, romb to średnia (liczba nad rombem).

W prawej kolumnie każdego panelu podano iloraz **MAC TILT/UNIFORM**
policzony z zapisanych kosztów MAC/obraz. To nie jest dopasowanie
kosztu: przy zbliżonej liczbie parametrów TILT ma jawnie większy MAC,
a iloraz rośnie z siłą redukcji (około ×1,105 / ×1,366 / ×1,824).

## Wyniki (benchmark, po CMA, 4 seedy)

| Zbiór | Nominalnie | Średnia TILT−UNIFORM | Pary | P_T/P_U (odchyłka) | MAC U→T [mln] | MAC T/U |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | 30% | **+3,0775 pp** | 4/4 | 0,99936 (0,064%) | 386,29→426,85 | ×1,105 |
| CIFAR-100 | 50% | **+9,7050 pp** | 4/4 | 0,99983 (0,017%) | 275,34→376,14 | ×1,366 |
| CIFAR-100 | 70% | **+11,6525 pp** | 4/4 | 0,99993 (0,007%) | 161,85→295,08 | ×1,823 |
| Tiny ImageNet | 30% | **+3,3050 pp** | 4/4 | 0,99994 (0,006%) | 386,34→426,98 | ×1,105 |
| Tiny ImageNet | 50% | **+9,4625 pp** | 4/4 | 0,99970 (0,030%) | 275,37→376,03 | ×1,366 |
| Tiny ImageNet | 70% | **+9,4850 pp** | 4/4 | 0,99995 (0,005%) | 161,88→295,31 | ×1,824 |

Brak równego MAC jest cechą porównania, nie błędem: TILT wygrywa top‑1
przy niemal identycznej liczbie P, ale przy większym koszcie mnożeń.

## Weryfikacja

`make_parameter_figure.py` przelicza z `per_seed.csv` top‑1, różnice
sparowane i średnie, po czym wymaga zgodności z `audit.json`
(tolerancja 1e-9): `mean_pp`, `per_seed_pp`, `sd_pp`, `positive_pairs = 4`
oraz średnie top‑1 dla `b30/b50/b70` w wariantach `U` i `T`. Sprawdza
`n = 10000`, stałe P i MAC w obrębie wariantu, **dopasowanie P do 0,1%**
(0,999 ≤ P_T/P_U ≤ 1; zmierzona maksymalna odchyłka 0,064%) oraz to, że
MAC TILT jest jawnie większy od MAC UNIFORM. Uruchomienie przerywa się
przy każdej niezgodności. Zapis jest deterministyczny (identyczne bajty
przy powtórzeniu); PDF ma 170,0 × 85,1 mm z osadzonym DejaVu Sans,
PNG 400 dpi.

Wejścia (SHA-256):

- `per_seed.csv`: `c58a1fe5c477cdaeefdfef24a5375b2fa020fc47f254971ec172772075cf5411`
- `audit.json`: `2361839e91cb1d56cc7f7a5e5817575a1e81f2f48a5ff749cb2bdb4df1705810`

## Odtworzenie (tylko CPU)

Z katalogu głównego repozytorium, Conda `attention`, Matplotlib 3.10.9:

```bash
CUDA_VISIBLE_DEVICES='' /home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-11-24_E049_folding_mocniejszych_baz/wykresy/20261124T234559Z_b4ca081b/make_parameter_figure.py
```

SHA-256 wyników:

- `e049_matched_parameters_en.png`: `749eb587284553407730fb6f7302494d261c7854a25003267fe96a21cc2128cb`
- `e049_matched_parameters_en.pdf`: `f80e2b1fe08091f2e6e139686f92ba29c002f4e7cb1a67fe939036fb80546047`
