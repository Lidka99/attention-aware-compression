# Manuskrypt do recenzji wewnętrznej

[PDF](main.pdf) · [LaTeX](main.tex) · [Abstrakt](abstract.tex) ·
[Bibliografia](references.bib) · [Tabele i CSV](tables/)

**Linear Depth Tilt (TILT)** zwiększa docelową redukcję kanałów wraz z
kolejnością grup. **Frozen Offset Profile (L06)** ponownie wykorzystuje
wybrane na CIFAR-100 dev przesunięcia udziału zachowanych kanałów względem
UNIFORM. W CSV b10 kolumna `model_id` przechowuje identyfikatory z `runs/`.

## Mapa dowodów

| Porównanie | Rola danych | Archiwalne źródło |
|---|---|---|
| E035: TILT–UNIFORM przy dopasowanym P, 15/15 po CMA | Oficjalne testy CIFAR i oficjalny Tiny val; niezależne od doboru alokacji | [`E035/20261107T183831Z_8b0b14e0`](../runs/E035/20261107T183831Z_8b0b14e0/analysis/): `per_seed.csv`, `controls.csv`, `analysis.json`, `verification.json`, `data_verification.json` i `../frozen.json` |
| E036: iloraz MAC TILT/UNIFORM 1,105–1,823 | Kształty modeli; czas forward na wejściach syntetycznych | [`E036/20261107T231027Z_27bac186`](../runs/E036/20261107T231027Z_27bac186/analysis/): `ratios.csv`, `analysis.json`, `verification.json` |
| E037: szerszy UNIFORM wygrywa 10/15 przy zbliżonym MAC | Wspólny dev; U/T z fazy dev E035 | [`E037/20261108T031759Z_855c0b33`](../runs/E037/20261108T031759Z_855c0b33/analysis/): `per_seed.csv`, `analysis.json`, `verification.json` i `../frozen.json` |
| E041: kontrole z oboma kosztami dopasowanymi do TILT | CIFAR-10 i CIFAR-100 dev, ResNet18/b50 | [`E041/20261116T040008Z_9e34f1e9`](../runs/E041/20261116T040008Z_9e34f1e9/analysis/): `per_seed.csv`, `audit.json`, `table_audit.json` |
| E042: wybór L06 z 24 kandydatów | CIFAR-100 dev do wyboru; CIFAR-10 dev jako kontrola | [`E042/20261121T212257Z_207ae03f`](../runs/E042/20261121T212257Z_207ae03f/analysis/): `profiles.csv`, `summary.json`, `audit.json` |
| E043 / E044: sprawdzenie L06 przy b50 i b30 | Osobna rezerwa CIFAR-100 / wcześniej użyty Tiny val | [`E043/20261122T000341Z_9e23e96f`](../runs/E043/20261122T000341Z_9e23e96f/analysis/) oraz [`E044/20261122T022955Z_6c1d2fc5`](../runs/E044/20261122T022955Z_6c1d2fc5/analysis/): `summary.json`, `per_seed.csv`, `audit.json` |
| E045: pełny model i równomierna kompresja b10/b20 | Tiny dev; odniesienie do oceny łagodnej kompresji | [`E045/20261122T031350Z_a58b56a3`](../runs/E045/20261122T031350Z_a58b56a3/analysis/): `summary.json`, `per_seed.csv`, `audit.json` |
| E046: L06 przy b10, +1,1425 pp pod oboma pułapami | Tiny dev, odniesienie UNIFORM z E045 | [`E046/20261122T045655Z_0b0c53b7`](../runs/E046/20261122T045655Z_0b0c53b7/analysis/): `summary.json`, `per_seed.csv`, `audit.json` |
| E047: +1,3500 pp, CI po obrazach [+0,700; +2,025] pp | ImageNetV2 MatchedFrequency, 2 000 obrazów tych samych 200 klas; te same cztery checkpointy co w E046 | [`E047/20261122T053804Z_ae2dec73`](../runs/E047/20261122T053804Z_ae2dec73/analysis/): `summary.json`, `per_seed.csv`, `audit.json` oraz `../gate.json` |

Operator upstream: `folding_as_projection`, commit
`694691556e8314a835178a741938dd3d4d80c171`. Szczegóły implementacji i
podziałów są w zamrożonych planach i raportach eksperymentów.

## Tabele

| LaTeX | Dane |
|---|---|
| [TILT–UNIFORM](tables/tilt_uniform.tex) | [CSV](tables/tilt_uniform.csv) |
| [Kontrole MAC, parametrów i czasu E036–E037](tables/cost_controls.tex) | [CSV](tables/cost_controls.csv) |
| [Kontrole obu kosztów E041](tables/dual_cost_controls.tex) | [CSV](tables/dual_cost_controls.csv) |
| [b10: pełny model, UNIFORM i Frozen Offset Profile](tables/b10.tex) | [Agregaty](tables/b10.csv), [wyniki per seed](tables/b10_per_seed.csv), [sparowane kontrasty i CI](tables/b10_contrasts.csv) |

Odtworzenie z katalogu głównego repozytorium:

```bash
python -B paper/prepare_tables.py
python -B paper/prepare_tables.py --check
```

## Figury

1. E035: [PDF](../experiments/2026-11-07_E035_ocena_niezalezna/figures/publikacja/e035_tilt_minus_uniform_en.pdf) · [PNG](../experiments/2026-11-07_E035_ocena_niezalezna/figures/publikacja/e035_tilt_minus_uniform_en.png)
2. E037: [PDF](../experiments/2026-11-08_E037_kontrola_mac/figures/publikacja/e037_accuracy_cost_en.pdf) · [PNG](../experiments/2026-11-08_E037_kontrola_mac/figures/publikacja/e037_accuracy_cost_en.png)
3. b10: [PDF](../experiments/2026-11-22_E047_l06_b10_imagenetv2/wykresy/20261122T053804Z_ae2dec73/b10_tiny_imagenetv2_en.pdf) · [PNG](../experiments/2026-11-22_E047_l06_b10_imagenetv2/wykresy/20261122T053804Z_ae2dec73/b10_tiny_imagenetv2_en.png)

Odtworzenie z katalogu głównego, w środowisku Conda `attention`
(Matplotlib 3.10.9). Skrypty czytają zapisane metryki i geometrię; nie
uruchamiają treningu ani punktacji obrazów. Opcja `--english-paper` zapisuje
tylko angielskie pliki z przyrostkiem `_en`.

```bash
export CUDA_VISIBLE_DEVICES=''
python -B experiments/2026-11-07_E035_ocena_niezalezna/figures/publikacja/generuj_publikacyjne.py --english-paper
python -B experiments/2026-11-08_E037_kontrola_mac/figures/publikacja/generuj_figure.py --english-paper
python -B experiments/2026-11-22_E047_l06_b10_imagenetv2/wykresy/20261122T053804Z_ae2dec73/make_figure.py --english-paper
```

## Kompilacja

Z katalogu głównego repozytorium (sprawdzono z Tectonic 0.17.0):

```bash
cd paper
tectonic main.tex
```

Kompilacja używa gotowych tabel i wektorowych PDF figur. Nie wymaga
dostępu do `runs/`, jeśli te pliki są już dostępne.
