# Manuskrypt — źródła i odtwarzanie

[PDF](main.pdf) · [Tekst główny](main.tex) · [Abstrakt](abstract.tex) · [Dodatki](appendix.tex) ·
[Bibliografia](references.bib) · [Tabele i CSV](tables/)

[Archiwum wersji 17-stronicowej sprzed redakcji](archiwum/przed_skroceniem_17_stron/README.md).

[Krótki materiał na spotkanie z promotorem](NA_WTOREK.md). Główna oś:
rozdział szerokości przy dopasowanej liczbie parametrów (E049, Tabela 1
i Rysunek 1) oraz koszt MAC. Dodatki A–D zawierają wcześniejsze wyniki,
pełne kontrole kosztu, łagodne cięcia i Frozen offsets.

**Linear Depth Tilt (TILT)** zwiększa docelową redukcję kanałów wraz z
kolejnością grup. **Frozen Width-Offset Profile** (w tabelach i na wykresach
**Frozen offsets**) przenosi wybrane na CIFAR-100 dev przesunięcia udziału
zachowanych kanałów względem UNIFORM. Zamrożone są przesunięcia; szerokości
wyliczamy ponownie pod oba pułapy. Identyfikatory `L06` i `b10_L06` w `runs/`
pozostają w kolumnach technicznych CSV (`model_id`, `comparison`).

## Mapa dowodów

| Porównanie | Rola danych | Archiwalne źródło |
|---|---|---|
| E013–E014: pochodzenie nachylenia TILT 0,30 | Wcześniejsze porównania na CIFAR-10 dev | [E013](../experiments/2026-10-25_E013_alokacja_budzetu_foldingu/PODSUMOWANIE.md) i [E014](../experiments/2026-10-27_E014_replikacja_alokacji/WYNIKI.md) |
| E048: pełne ResNet18 z nowej receptury, 81,125% / 63,465% | CIFAR-100 i Tiny dev przed CMA; na Tiny także nowy stem i 555,52 mln MAC/obraz | [`E048/train_20261124T232822Z_89dfc621`](../runs/E048/train_20261124T232822Z_89dfc621/): `audit.json`, `manifest.json` |
| **E049: TILT–UNIFORM przy dopasowanym P**, 30/50/70%, 6/6 po 4/4; MAC TILT/UNIFORM 1,105–1,824 | Oficjalny test CIFAR-100 i oficjalny Tiny val, oba wcześniej użyte; nowe checkpointy E048 | [`E049/eval_20261124T234559Z_b4ca081b`](../runs/E049/eval_20261124T234559Z_b4ca081b/): `audit.json`, `per_seed.csv`, `geometry.json`, `WYNIKI.md` |
| **E051: łagodne cięcie przy dopasowanym P**, b10 TILT−U +1,7650 / +1,7125 pp, po 4/4; b5 pokazany osobno | Wyłącznie CIFAR-100 i Tiny dev; te same modele E048, kalibracja Tiny z E050 | [Wyniki i audyt](../experiments/2026-11-26_E051_lagodny_tilt/WYNIKI.md) |
| E052: lokalne przesunięcie b10, S04−TILT +0,1350 / +0,2000 pp; strata na Tiny 3,4075 pp | Wybór na rozłącznym podzbiorze train; jednorazowa ocena wyboru na CIFAR-100 i Tiny dev | [Wyniki, manifest, per seed](../experiments/2026-11-26_E052_lokalne_przesuniecie_b10/WYNIKI.md) |
| E049: kontrola obu pułapów w b10, Frozen offsets względem UNIFORM +0,615 / +0,675 pp; kryterium praktyczne niespełnione | Te same benchmarki i nowe checkpointy | Ten sam audyt oraz `b10_transfer` poniżej |
| E050: czułość BN, Frozen offsets +1,320 pp przy niezmienionych kosztach | Tiny dev, kontrola zaprojektowana po E049, łagodniejszy crop pomaga też pozostałym profilom | [`E050/dev_20261125T231044Z_d44c66e4`](../runs/E050/dev_20261125T231044Z_d44c66e4/): `audit.json`, `per_seed.csv`, `WYNIKI.md` |
| E035: TILT–UNIFORM przy dopasowanym P, 15/15 po CMA | Oficjalne testy CIFAR i oficjalny Tiny val; niezależne od doboru alokacji | [`E035/20261107T183831Z_8b0b14e0`](../runs/E035/20261107T183831Z_8b0b14e0/analysis/): `per_seed.csv`, `controls.csv`, `analysis.json`, `verification.json`, `data_verification.json` i `../frozen.json` |
| E036: iloraz MAC TILT/UNIFORM 1,105–1,823 | Kształty modeli; czas forward na wejściach syntetycznych | [`E036/20261107T231027Z_27bac186`](../runs/E036/20261107T231027Z_27bac186/analysis/): `ratios.csv`, `analysis.json`, `verification.json` |
| E037: szerszy UNIFORM wygrywa 10/15 przy zbliżonym MAC | Wspólny dev; U/T z fazy dev E035 | [`E037/20261108T031759Z_855c0b33`](../runs/E037/20261108T031759Z_855c0b33/analysis/): `per_seed.csv`, `analysis.json`, `verification.json` i `../frozen.json` |
| E038: profil TILT pomaga także MAG2 | Kontrola operatora przy tych samych szerokościach na wcześniejszych checkpointach | [Raport E038](../experiments/2026-11-08_E038_operator_alokacja/WYNIKI.md) |
| E041: kontrole z oboma kosztami dopasowanymi do TILT | CIFAR-10 i CIFAR-100 dev, ResNet18/b50 | [`E041/20261116T040008Z_9e34f1e9`](../runs/E041/20261116T040008Z_9e34f1e9/analysis/): `per_seed.csv`, `audit.json`, `table_audit.json` |
| E042: wybór zamrożonego profilu z 24 kandydatów | CIFAR-100 dev do wyboru; CIFAR-10 dev jako kontrola | [`E042/20261121T212257Z_207ae03f`](../runs/E042/20261121T212257Z_207ae03f/analysis/): `profiles.csv`, `summary.json`, `audit.json` |
| E043 / E044: sprawdzenie Frozen offsets przy b50 i b30 | Osobna rezerwa CIFAR-100 / wcześniej użyty Tiny val | [`E043/20261122T000341Z_9e23e96f`](../runs/E043/20261122T000341Z_9e23e96f/analysis/) oraz [`E044/20261122T022955Z_6c1d2fc5`](../runs/E044/20261122T022955Z_6c1d2fc5/analysis/): `summary.json`, `per_seed.csv`, `audit.json` |
| E045: pełny model i równomierna kompresja b10/b20 | Tiny dev; odniesienie do oceny łagodnej kompresji | [`E045/20261122T031350Z_a58b56a3`](../runs/E045/20261122T031350Z_a58b56a3/analysis/): `summary.json`, `per_seed.csv`, `audit.json` |
| E046: Frozen offsets przy b10, +1,1425 pp pod oboma pułapami | Tiny dev, odniesienie UNIFORM z E045 | [`E046/20261122T045655Z_0b0c53b7`](../runs/E046/20261122T045655Z_0b0c53b7/analysis/): `summary.json`, `per_seed.csv`, `audit.json` |
| E047: +1,3500 pp, CI po obrazach [+0,700; +2,025] pp | ImageNetV2 MatchedFrequency, 2 000 obrazów tych samych 200 klas; te same cztery checkpointy co w E046 | [`E047/20261122T053804Z_ae2dec73`](../runs/E047/20261122T053804Z_ae2dec73/analysis/): `summary.json`, `per_seed.csv`, `audit.json` oraz `../gate.json` |

Operator upstream: `folding_as_projection`, commit
`694691556e8314a835178a741938dd3d4d80c171`. Szczegóły implementacji i
podziałów są w zamrożonych planach i raportach eksperymentów.

## Tabele

| LaTeX | Dane |
|---|---|
| [E049: główna tabela TILT–UNIFORM przy dopasowanym P](tables/tilt_new.tex) | [Dokładne P, MAC, top‑1 i kontrasty per seed](tables/tilt_new.csv) |
| [E051: łagodny TILT przy dopasowanym P, dev](tables/tilt_mild_dev.tex) | [P, MAC, top‑1 i kontrasty per seed](tables/tilt_mild_dev.csv) |
| [Kontrola b10 E049: pełny / UNIFORM / T-proj / Frozen offsets](tables/b10_transfer.tex) | [Agregaty](tables/b10_transfer.csv), [per seed](tables/b10_transfer_per_seed.csv), [kontrasty](tables/b10_transfer_contrasts.csv) |
| Kontrola cropu BN E050 (dodatek D) | [Agregaty i różnice per seed](tables/b10_tiny_crop.csv) |
| [TILT–UNIFORM](tables/tilt_uniform.tex) | [CSV](tables/tilt_uniform.csv) |
| [Kontrole MAC, parametrów i czasu E036–E037](tables/cost_controls.tex) | [CSV](tables/cost_controls.csv) |
| [Kontrole obu kosztów E041](tables/dual_cost_controls.tex) | [CSV](tables/dual_cost_controls.csv) |
| [b10 na wcześniejszych modelach Tiny: dev i ImageNetV2](tables/b10.tex) | [Agregaty](tables/b10.csv), [wyniki per seed](tables/b10_per_seed.csv), [sparowane kontrasty i CI](tables/b10_contrasts.csv) |

Odtworzenie z katalogu głównego repozytorium, z dostępem do lokalnych
artefaktów `runs/` wymienionych w mapie dowodów:

```bash
python -B paper/prepare_tables.py
python -B paper/prepare_tables.py --check
```

## Figury

W PDF są Rysunek 1 (E049) oraz Rysunek S1 (E037). Pozostałe figury
są dostępne w katalogach swoich eksperymentów.

1. E049 przy dopasowanym P: [PDF](../experiments/2026-11-24_E049_folding_mocniejszych_baz/wykresy/20261124T234559Z_b4ca081b/e049_matched_parameters_en.pdf) · [PNG](../experiments/2026-11-24_E049_folding_mocniejszych_baz/wykresy/20261124T234559Z_b4ca081b/e049_matched_parameters_en.png)
2. E035: [PDF](../experiments/2026-11-07_E035_ocena_niezalezna/figures/publikacja/e035_tilt_minus_uniform_en.pdf) · [PNG](../experiments/2026-11-07_E035_ocena_niezalezna/figures/publikacja/e035_tilt_minus_uniform_en.png)
3. E037: [PDF](../experiments/2026-11-08_E037_kontrola_mac/figures/publikacja/e037_accuracy_cost_en.pdf) · [PNG](../experiments/2026-11-08_E037_kontrola_mac/figures/publikacja/e037_accuracy_cost_en.png)
4. Kontrola b10 E049: [PDF](../experiments/2026-11-24_E049_folding_mocniejszych_baz/wykresy/20261124T234559Z_b4ca081b/e049_b10_joint_caps_en.pdf) · [PNG](../experiments/2026-11-24_E049_folding_mocniejszych_baz/wykresy/20261124T234559Z_b4ca081b/e049_b10_joint_caps_en.png)
5. b10 E047: [PDF](../experiments/2026-11-22_E047_l06_b10_imagenetv2/wykresy/20261122T053804Z_ae2dec73/b10_tiny_imagenetv2_en.pdf) · [PNG](../experiments/2026-11-22_E047_l06_b10_imagenetv2/wykresy/20261122T053804Z_ae2dec73/b10_tiny_imagenetv2_en.png)

Odtworzenie z katalogu głównego, w środowisku Conda `attention`
(Matplotlib 3.10.9). Skrypty czytają zapisane metryki i geometrię; nie
uruchamiają treningu ani punktacji obrazów. Opcja `--english-paper` zapisuje
tylko angielskie pliki z przyrostkiem `_en`.

```bash
export CUDA_VISIBLE_DEVICES=''
python -B experiments/2026-11-24_E049_folding_mocniejszych_baz/wykresy/20261124T234559Z_b4ca081b/make_parameter_figure.py
python -B experiments/2026-11-24_E049_folding_mocniejszych_baz/wykresy/20261124T234559Z_b4ca081b/make_figure.py
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

## Przed udostępnieniem źródeł

- Do pakietu budowy dołączyć `paper/{main,abstract,appendix}.tex`,
  `paper/references.bib`, siedem plików `.tex` z `paper/tables/` i oba
  PDF figur E049/E037 w ścieżkach oczekiwanych przez LaTeX.
- Do weryfikacji liczb opublikować CSV per seed, zamrożone geometrie,
  manifesty i audyty, podając trwały URL w manuskrypcie. Katalog `runs/`
  jest lokalny i ignorowany przez Git; sam klon nie wystarcza do
  regeneracji tabel z surowych metryk.
- Przed zgłoszeniem uzupełnić autorów i afiliacje zgodnie z wymaganiami
  wybranego miejsca publikacji.
