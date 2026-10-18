# E010 — wersja wykonania 1.0 (projekt)

Eksperyment: `2026-10-18_E010_benchmark_kryteriow`.
Data utworzenia: 2026-10-18 UTC.
**Status protokołu: frozen (1.0)**. Status wykonania podaje manifest biegu.
Testy CPU: [TESTY_CPU_v1.json](TESTY_CPU_v1.json) — passed; hashe kodu
w `config.json`.

## Cel i pytanie

Czy któreś ze standardowych kryteriów wyboru kanałów do scalenia daje po
**krótkiej naprawie** (1–20 epok) istotnie lepszą jakość niż RANDOM?
Kryterium pierwszorzędne: Tiny ImageNet 30%, T=5, przewaga ≥ 1,0 pp nad
RANDOM przy Holm-adjusted p < 0,05. Replikacja: CIFAR-100 10%, T=5 (oraz
Tiny 10%, T=5 dla kryteriów obecnych w tej siatce).
Pełne uzasadnienie i ograniczenia: [OPIS.md](OPIS.md).

## Ustalenia do zamrożenia

### Datasety i wejścia (pinowane SHA-256)

**Tiny ImageNet 224 (primary):**
- Baseline E007: `runs/E007/20261011T061825Z_4de51b3_seed42_24mzpz80/last.pt`,
  `718bbb26…744408`, epoka 100, 4 263 748 param., state_dict `01df27ed…40e2d`.
- Bieg E008: `runs/E008/20261012T042932Z_0fd9962_seed42_tn1zlyim`,
  git `0fd99625…`, status `completed`; `stage1_metrics.json` `1385a2d0…`,
  `model_files.json` `24c09ed6…`.
- Modele 30%: PRE `fa9bfc57…` / `064c83e0…`, POST `8e41d3fd…` /
  `7c0af136…`, RANDOM `03ccb85f…` / `8913a02a…` (plik / state_dict).
- Modele 10%: PRE `4123ad25…` / `34d22462…`, POST `1faf49d4…` /
  `8c0fa4f8…`, RANDOM `a07b6676…` / `5e0c9efc…`.
- Momenty: `stage1/moments_merged.npz` `e0878d8b…`.
- Manifesty: 30% `1bd0c412…`, 10% `e9879c54…`.
- Plany: 30% `PLAN_STRUKTURY_30.json` `fa08dbcd…` (2 981 828 param.,
  3168 usuwanych); 10% `PLAN_STRUKTURY_10.json` `1c23aad9…` (3 837 364;
  1048 usuwanych).
- Dane: splits `9df906f7…`, cache_train `90642db9…`, cache_val `fd0c5fc8…`;
  A = 1000 obrazów (5/klasę, seed 20261015), dev = 10 000.

**CIFAR-100 96 (secondary):**
- Baseline E001: `runs/E001/20260926T013203Z_7cc3f46_seed42_qHKULg/best.pt`,
  `f2ac9c3f…`, epoka 176 (best), 4 135 648 param., git `7cc3f466…`.
- Bieg E003: `runs/E003/20261010T095706Z_5a2f233_seed42__0o_uckk`,
  git `5a2f233a…`, status `completed`; `model_files.json` `8d41d709…`,
  `candidate_manifest.json` `47ae4db3…`, `moments_merged.npz` `2b67fc3d…`.
- Modele 10%: PRE `8186ff81…` / `55d842d3…`, POST `9f4e94dd…` /
  `8ca2e68e…`, RANDOM `c4ba8b4f…` / `0e64b6c2…`.
- Plan: `PLAN_STRUKTURY.json` `fdfa3524…` (3 721 424 param., 1016 usuwanych).
- Kalibracja: `runs/E002/20261001T051649Z_e0b3121_seed42_a8l9uuks/SAMPLES.json`
  `338eb318…`; A = 500 obrazów (5/klasę), dev = 5000.
- Dane: `data/cifar100/splits.json` `ecb8804e…`, train pickle `735e79b0…`.

**Wspólne:** recipe naprawy E009
`runs/E009/20261013T172339Z_954ece0_seed42_nh02g2l8/source/config.json`
`495f065e…`. Zależności kodu: `support_e008.py` `c32689fe…`,
`folding_e008.py` `2ab8c90b…`, E003 `support.py` `e3fa1ee1…`,
`folding.py` `90c0c8f3…`, `selection.py` `53d2c251…`,
`summarize.py` `8007f0aa…`, E002 `common.py` `20a4afbf…`.

### Operator

`folding_e010.py` — fork E008/E003. Semantyka α/β bez zmian. Rozluźnienia:
(1) para `(c,d)` bez wymogu `c < d`; (2) reprezentant `c` może wchłonąć
wiele usuwanych kanałów (dla klastrów k-means), przy czym każdy `d` jest
unikalny, a żaden kanał nie jest jednocześnie `c` i `d`. Dla par
E008/E003 wynik jest identyczny — test CPU odtwarza 9/9 modeli bit w bit.

### Kryteria

Reużyte: **PRE**, **POST**, **RANDOM** (modele etapu 1, zweryfikowane
hashami i odtworzeniem top-1/NLL).
Nowe:
- **L1** — `||W_exp[o]||_1`; usuwane najniższe.
- **BN_GAMMA** — `|γ_exp[o]|`; usuwane najniższe.
- **TAYLOR** — `mean_n |Σ_{h,w} g·a|` na wyjściu expansion conv+BN+SiLU;
  usuwane najniższe (CE bez label smoothing).
- **APOZ** — `1 − zero_fraction(a)` na tym samym miejscu; usuwane najniższe.
- **REC** — score pary `−0,5·(r_g/S_g[d,d] + r_z/S_z[d,d])`,
  `r_x = S_x[d,d] − S_x[c,d]^2/S_x[c,c]`; zachłanne rozłączne pary.
- **KMEANS** — k-means++/Lloyd na wektorach filtrów po BN (Gram), `k` =
  docelowa liczba kanałów, medoid reprezentantem; pozostali scalani do niego.

Wspólne: liczba usuwanych = `remove_count` z planu ratio; kanały
niekwalifikujące się (norma Grama ≤ 1e-12) zostają; kryteria per-kanał
usuwają najsłabsze i przypisują je zachłannie do unikalnych reprezentantów
po cosine filtrów po BN; α = `S_g[c,d]/S_g[c,c]`, β = `S_z[c,d]/S_z[c,c]`.

### Statystyki

Jedny przebieg 4 GPU na A (Tiny 1000, CIFAR 500), BASE w `eval`,
transformacja ewaluacyjna, bez augmentacji, `A[rank::4]`, batch 32/GPU,
FP32, loss = plain CE. Miejsce: wyjście expansion conv+BN+SiLU
(wejście depthwise). Zapis: `stats/<dataset>.npz` + `.json`.

### Scalanie i bramki

6 nowych kryteriów na każdym ratio. Kontrole: parametry dokładnie
2 981 828 / 3 837 364 / 3 721 424; pary rozłączne (`d` unikalne);
α/β skończone; determinizm grup. Zero-shot dev (4 GPU, batch 64/GPU).
Modele reużyte i BASE muszą odtworzyć top-1 dokładnie i NLL z
|Δ| ≤ 1e-5 względem `stage1_metrics.json` (Tiny) / `raw_metrics.json`
(CIFAR), inaczej failed.

### Naprawa

- Tiny 30%: 9 kryteriów, T ∈ {1, 5, 10, 20}; Tiny 10%: 6 kryteriów, T=5.
- CIFAR-100 10%: 9 kryteriów, T ∈ {1, 5, 10, 20}.
- Recipe: AdamW lr 3e-4, wd 0,01, warmup `min(3,T)`, cosine do min_lr 3e-6,
  LS 0,1, clip 1,0, AMP FP16; batch 32/GPU (global 128); 4 GPU DDP;
  seed 42; dev co epokę; wynik = ostatnia epoka; bez early stoppingu.
- Tiny: RRC(224) + ColorJitter + hflip. CIFAR-100: RandomCrop(32, pad 4) +
  hflip + Resize(96, bilinear); normalizacja 0,5. CIFAR nie odtwarza
  receptury SGD z E004; porównania są w obrębie E010.
- Każda komórka startuje z tego samego modelu scalonego; brak wspólnego
  treningu między T. Wznowienie: brak; przerwany bieg powtarzamy od zera.

### Ewaluacja i statystyka

- Dev top-1 i NLL; predykcje per komórka, pokrycie wszystkich obrazów.
- Sparowany bootstrap 10 000, seed 20261018, stratyfikacja po klasach;
  dokładny wariant zgodny z E003 (test CPU) dla primary/replikacji,
  wektorowy dla eksploracji.
- Primary: Tiny 30%, T=5, nowe kryterium vs RANDOM; próg ≥ 1,0 pp;
  jednostronne p; Holm w rodzinie 6; α = 0,05.
- Replikacja: CIFAR-100 10%, T=5 (CI 95%, dolna > 0 i próg ≥ 1,0 pp);
  Tiny 10%, T=5 dla kryteriów w siatce.
- Pozostałe T/pary: eksploracyjne, bez etykiety „survivor".

### Decyzja po E010

- **Survivor + replikacja** → E011: metoda wokół kryterium, 3 seedy,
  T ∈ {5, 20, 60}, ratio 10/30/50, pełne baseline'y.
- **Brak survivora** → zamknięcie wątku selekcji w badanym reżimie;
  publikacja benchmarku + wyniku negatywnego; pivot na naprawę
  (attention-map distillation) lub operator scalania.
- **Wynik mieszany** → powtórka screeningu na drugim seedzie.

### Katalog biegu i koszt

- `runs/E010/<UTC>_<git>_seed42_<unikalny>/` z `stats/`, `groups/`,
  `merge/`, `repair/<dataset>_<ratio>_<kryterium>_T<T>/`, `summary.json`,
  `artifacts.sha256`. Nowy katalog, bez nadpisywania.
- GPU: Tiny 30% ≈ 16,2 h; Tiny 10% ≈ 1,5 h; CIFAR-100 10% ≈ 2,5–3,5 h;
  stats/zero-shot/summary ≈ 1 h → **~21–22 h** pracy 4 GPU.
- Dysk: ~2,2 GB (wolne ~5,2 GB w dniu tworzenia protokołu).

## Kontrole przed zamrożeniem

- [x] Kod: `folding_e010.py`, `support_e010.py`, `criteria_e010.py`,
      `stats_e010.py`, `merge_e010.py`, `repair_e010.py`,
      `summarize_e010.py`, `launch_e010.py`, `check_cpu_e010.py`.
- [x] `folding_e010.py` odtwarza state_dict E008/E003: **9/9** modeli.
- [x] Liczby parametrów i pary grup na syntetycznych statystykach:
      **15/15** plików grup, determinizm, budżety poprawne.
- [x] Bootstrap zgodny z E003 (E008 i E003, delta i CI).
- [x] `py_compile` wszystkich modułów: OK.
- [x] `TESTY_CPU_v1.json` = **passed**.
- [x] `config.json` frozen + hashe kodu; ten plik: status `frozen`.
- [x] Commit protokołu i kodu (ten commit).
- [ ] Osobna zgoda użytkowniczki na uruchomienie.

## Komenda (po zamrożeniu)

```bash
conda run --no-capture-output -n attention python experiments/2026-10-18_E010_benchmark_kryteriow/launch_e010.py
```

## Ograniczenia

Jeden seed (screening); jedna architektura/baseline; operator wspólny
(pairwise LS z rozszerzeniem star-merge), więc KMEANS nie jest średnią
klastra; CIFAR używa recipe AdamW E009, nie SGD E004; statystyki z jednego
zbioru kalibracyjnego; dev użyty do wyboru epoki baseline'u, nie niezależny
test; rezerwa i test nietknięte; brak wniosków publikacyjnych bez E011.
