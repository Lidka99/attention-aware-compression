# Figura b10 — Tiny dev i zewnętrzna próba ImageNetV2

[PDF wektorowy](b10_tiny_imagenetv2.pdf) · [PNG](b10_tiny_imagenetv2.png) ·
[Generator](make_figure.py) · [Wyniki E047](../../WYNIKI.md)

- **A, Tiny dev:** średni top‑1 po CMA względem rzeczywistej redukcji
  parametrów. Pełny model i U/b10–b20 pochodzą z E045, U/b30 i U/b50 z
  E035 dev (udokumentowany reuse w E045). Osobny punkt to L06/b10 z E046.
- **B, ImageNetV2 MatchedFrequency:** 2 000 obrazów, 200 klas Tiny.
  U góry średni top‑1 pełnego, U/b10 i L06/b10. U dołu różnice L06−U po
  seedach 42–45 oraz średnia **+1,35 pp** z 95% przedziałem sparowanego
  bootstrapu **[+0,700; +2,025] pp**, warunkowo na tych checkpointach.
  Strata do pełnego modelu: **3,35 pp dla U** i **2,00 pp dla L06**.

Każdy panel ma własną skalę i fazę danych. Oś redukcji P w A wykorzystuje
niezaokrąglone wartości; w porównaniu b10 L06 ma 10 104 855 P i 132 302 192
MAC/obraz, U ma 10 111 430 P i 132 336 196 MAC/obraz, w obu panelach.

## Źródła i odtworzenie

| Katalog `runs/.../analysis/` | Wykorzystane metryki |
|---|---|
| `E045/20261122T031350Z_a58b56a3` | Pełny model, krzywa U (w tym b30/b50 z E035 dev) |
| `E046/20261122T045655Z_0b0c53b7` | L06/b10 i odtworzenia kontroli Tiny dev |
| `E047/20261122T053804Z_ae2dec73` | Top‑1, trafienia i kontrast z przedziałem na ImageNetV2 |

Generator przypina SHA‑256 każdego `summary.json` i `per_seed.csv`,
przelicza top‑1 i różnice z trafień, sprawdza zgodność kontroli E045/E046
oraz kosztów obu porównań. Rysuje zapisany przedział bootstrap z E047.

Z katalogu głównego repozytorium, Conda `attention`, Matplotlib 3.10.9:

```bash
CUDA_VISIBLE_DEVICES='' /home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-11-22_E047_l06_b10_imagenetv2/wykresy/20261122T053804Z_ae2dec73/make_figure.py
```

SHA‑256 plików:

- `b10_tiny_imagenetv2.png`: `90b31e8c1a33f7e24504c3f0a727089344966bdaecf0a7080b9792c5907bc187`
- `b10_tiny_imagenetv2.pdf`: `5f7421a58314dc983c280c07feb7c72bee1db2ae78f01ea88c920f1916dc4c27`
