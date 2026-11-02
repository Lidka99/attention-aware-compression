# 2026-10-27_E016_wrazliwosc_foldingu

Status: **protokół 1.0 zamrożony przed uruchomieniem etapu S1**.
Data utworzenia: 2026-10-27 (UTC). To nowa próba badawcza E016, nie powtórka
E014/E015. Wyniki S1 nie są jeszcze znane w chwili zamrożenia protokołu.

## Cel

Zmniejszyć ryzyko, że dotychczasowa alokacja depth-aware (TILT) działa
przez przypadek: zmierzyć **wrażliwość pojedynczych grup foldingu** na
kompresję, oddzielić szkodę **przed** i **po** korekcie BN oraz sprawdzić,
czy proste predyktory bez etykiet przewidują tę szkodę. Dopiero po tej
diagnostyce zaplanowano automatyczną alokację 50% i porównanie z UNIFORM
oraz TILT. E016 nie zmienia E011–E015 i nie importuje attention_v2 ani
attention_v3.

Ten eksperyment jest rozwojowy i diagnostyczny. Nie obiecuje nowości ani
publikacji. Nie stroi hiperparametrów na dev.

## Kontekst

- **E014** (CIFAR-10/ResNet18, 4 seedy): TILT+0.30 vs UNIFORM
  **+0,87 / +3,77 / +11,27 pp** przy 30/50/70% redukcji, 12/12 par dodatnich.
- **E015** (CIFAR-100/ResNet18, 4 seedy): **+0,81 / +1,96 / +1,60 pp**;
  zamrożone kryterium replikacji primary **nie spełnione** (50/70% poniżej
  +2,0 pp). TILT+0.15 lepszy wtórnie, bez zastąpienia primary.
- Wniosek: kierunek jest dodatni, ale mechanizm nie jest wyjaśniony.
  E016 zaczyna od pomiaru, która grupa i jaki poziom kompresji naprawdę
  szkodzi, zamiast od kolejnego treningu lub kolejnej siły tiltu.

## Etap S1 — diagnostyka pojedynczych grup (ten etap uruchamiamy)

### Wejścia

Istniejące checkpointy, bez nowego treningu:

| Dataset | Checkpoint | Seed | Dev top-1 | Parametry |
|---|---|---:|---:|---:|
| CIFAR-10 | `runs/E011/20261024T192205Z_018f3f2_seeds42-45_b2da2012/seed_42/last.pt` | 42 | 95,28% | 11 173 962 |
| CIFAR-100 | `runs/E015/20261027T190911Z_cb9ffaf_seeds42-45_82c98b12/seed_42/last.pt` | 42 | 74,94% | 11 220 132 |

Hashe checkpointów i splitów są w `configs/stage1.json` i są weryfikowane
przed każdym uruchomieniem. Używamy wyłącznie train/dev. **Oficjalny test
CIFAR-10/CIFAR-100 i rezerwa 5 000 obrazów CIFAR-100 pozostają nietknięte.**
Dla CIFAR-10 split E011 ma 45 000 train / 5 000 dev; dla CIFAR-100 split ma
40 000 train / 5 000 dev / 5 000 rezerwy (rezerwa nieużywana).

### Operator i grupy

Operator foldingu jest importowany bez zmian z E012/E014:

- `experiments/2026-10-24_E012_folding_referencja/folding.py`,
- `experiments/2026-10-24_E012_folding_referencja/evaluate.py`,
- `experiments/2026-10-27_E014_replikacja_alokacji/folding_budget.py`.

Używamy 12 grup E012 w kolejności depth (`relu_conv`, `layer1.0.relu1`, ...,
`layer4.1.relu1`). Dla jednego przebiegu kompresujemy **dokładnie jedną
grupę** do zadanego ratio, a pozostałe grupy mają ratio 0. Dzięki temu
szkoda jest przypisana do jednej grupy, a nie do całej alokacji.

Poziomy kompresji pojedynczej grupy (zamrożone): **25%, 50%, 75%, 90%**
usuniętych kanałów w tej grupie. Ratio 0 nie jest liczony jako poziom —
służy jako kontrola tożsamościowa (jeden przebieg na dataset).

Liczba przebiegów S1:

- 2 datasety × 12 grup × 4 poziomy = **96 przebiegów grupowych**,
- 2 datasety × 1 kontrola ratio 0 = **2 przebiegi kontrolne**,
- razem **98 unikalnych przebiegów**, każdy z własnym katalogiem wyników.

### Pomiary szkody przed/po BN

Dla każdego przebiegu:

1. Baseline: ocena oryginalnego checkpointu na dev (top-1, NLL, logity).
2. Folding jednej grupy na CPU; bez gradientów i bez aktualizacji wag.
3. **Pre-BN**: ocena na dev zaraz po foldingu, z BN odziedziczonym przez
   uśrednienie statystyk klastrów.
4. **Post-BN**: korekta BN na pełnym train (CIFAR-10 45 000, CIFAR-100
   40 000), seed `20261024`, bez etykiet w optymalizacji i bez gradientów;
   następnie ocena na dev.
5. Szkoda: `damage_pre_bn_pp = 100·(baseline_top1 − pre_top1)`,
   `damage_post_bn_pp = 100·(baseline_top1 − post_top1)`;
   `bn_recovery_pp = post_top1 − pre_top1` (dodatnie = BN pomaga).
   Analogicznie dla NLL.

Dev służy **tylko do pomiaru szkody i walidacji predyktorów**. Nie zmieniamy
po zobaczeniu dev poziomów, predyktorów ani kryteriów. Etap S2 będzie
korzystał wyłącznie z predyktorów bez etykiet, nie z dev.

### Predyktory bez etykiet (zamrożone przed biegiem)

Dla każdego przebiegu liczymy trzy rodziny predyktorów:

1. **Głębokość**: `depth_index = g` oraz `depth_norm = g/11` dla grupy
   `g = 0..11`. To predyktor strukturalny, nie wymaga danych.
2. **Błąd rekonstrukcji wag**: względny błąd Frobeniusa projekcji parametrów
   grupy po scaleniu kanałów,
   `||X − Mᵀ(MX)||_F / ||X||_F`, gdzie `X` to cechy parametrów grupy
   (wagi conv/BN/FC w obrębie grupy), a `M` to macierz scalania klastrów.
   Dla ratio 0 predyktor wynosi dokładnie 0. Predyktor nie używa obrazów
   ani etykiet.
3. **Zmiana wyjścia modelu na stałym train-calibration bez etykiet**:
   stały podzbiór **2 048 obrazów train** wybrany przez
   `numpy.random.default_rng(20261027).choice`, posortowany, transformacja
   `dev_transform` (deterministyczny CenterCrop), bez etykiet i bez
   augmentacji losowej. Liczymy zmianę logitów względem baseline:
   `calib_logit_rmse`, `calib_logit_mae`, `calib_prediction_agreement`,
   `calib_kl` — osobno **pre-BN** i **post-BN**. Etykiety tego podzbioru nie
   są używane do wyboru próbek ani do predyktora; predyktor jest liczony
   wyłącznie z logitów.

**Predyktor primary (zamrożony):** `weight_reconstruction_rel_error`.
Predyktory wtórne: `calib_logit_rmse_pre_bn`, `calib_logit_rmse_post_bn`,
`depth_norm`. Wybór primary nie zależy od wyników dev; dlatego S2 nie stroi
się na dev.

### Kontrola jakości S1

- Kontrola ratio 0 musi mieć pre-BN top-1 równy baseline w granicach
  **0,05 pp** i agreement logitów **≥ 0,999999**; błąd rekonstrukcji 0.
- Każdy przebieg: parametry po foldingu ≤ baseline, wszystkie wartości
  skończone, dev count 5 000, brak testu i rezerwy.
- Jeżeli którykolwiek przebieg się nie powiedzie, S1 jest nieukończony;
  nie przechodzimy do S2.

### Budżet obliczeń S1 (zamrożony)

- **4 GPU** (RTX 2080 Ti), 4 niezależne procesy-workerów, po jednym na GPU,
  w dwóch falach: najpierw CIFAR-10, potem CIFAR-100. Każdy worker dostaje
  własny, rozłączny shard przebiegów; każdy przebieg ma unikalny katalog.
- Szacunek: 98 przebiegów × ok. 15–30 s; fala 4 workerów po ok. 12–13
  przebiegów. Oczekiwany wall-time **10–20 min**, twardy limit **120 min**.
- Dysk: logity dev + diagnostyka + snapshoty; limit **1,5 GiB**.
- Brak nowego treningu, brak fine-tuningu, brak gradientów.
- Smoke: 8 przebiegów (4 na dataset), pełny BN i dev, osobny katalog
  `runs/E016/smoke_*`; bez znaczenia naukowego.

### Kryteria kontynuacji do S2 (zamrożone)

S2 (automatyczna alokacja 50% i porównanie UNIFORM/TILT) uruchamiamy tylko
jeżeli **wszystkie** poniższe warunki są spełnione:

- **C1 — kompletność:** 98/98 przebiegów S1 completed, 2/2 kontrole
  poprawne, brak użycia testu/rezerwy.
- **C2 — dynamika szkody:** przy poziomie 75% na **oba** datasety:
  rozstęp średniej szkody post-BN między najlepszą i najgorszą grupą
  ≥ **1,0 pp**, oraz co najmniej **4 grupy** mają średnią szkodę post-BN
  ≥ **0,5 pp**.
- **C3 — BN:** raportujemy średnie `bn_recovery_pp`; nie jest to warunek
  go/no-go, ale jeśli BN średnio pogarsza (recovery < 0), wynik wymaga
  osobnego wyjaśnienia przed S2.
- **C4 — predyktor primary:** `weight_reconstruction_rel_error` ma
  |Spearman ρ| ≥ **0,5** ze szkodą post-BN na **obu** datasetach i ten sam
  znak korelacji. Jeśli nie, S2 **nie jest uzasadniony**; raportujemy wynik
  negatywny i nie uruchamiamy automatycznej alokacji.
- **C5 — budżet:** wall-time ≤ 120 min i rozmiar wyników ≤ 1,5 GiB.

Jeżeli C4 nie przejdzie, nie „naprawiamy” wyniku przez wybór innego
predyktora po zobaczeniu dev. Predyktory wtórne raportujemy opisowo.

## Etap S2 — automatyczna alokacja 50% (zaplanowany, NIE uruchamiany teraz)

S2 pozostaje jawnym następnym etapem. Nie deklarujemy jego wykonania.

- Datasety: CIFAR-10 i CIFAR-100, seed42 (etap rozwojowy).
- Budżet: 50% redukcji; wspólny target = faktyczne parametry UNIFORM,
  tak jak w E014/E015.
- Warianty: **UNIFORM**, **TILT+0.30** (zamrożona hipoteza E014/E015) oraz
  **AUTO50**.
- AUTO50: dla każdej grupy liczymy predyktor primary z S1 (średnia po
  poziomach); standaryzujemy go w obrębie datasetu; alokacja
  `r_g = clip(0.5 + β·z_g, 0.10, 0.90)`, gdzie `β` jest dobierane
  deterministycznie do wspólnego targetu UNIFORM, a następnie naprawiane
  przez `repair_budget` bez przekroczenia targetu. Dev nie jest używany do
  wyboru `β` ani grup.
- Kryterium rozwojowe S2: AUTO50 post-BN top-1 > UNIFORM50 na obu
  datasetach oraz AUTO50 nie gorszy od TILT+0.30 o więcej niż 0,5 pp.
  Przy n=1 seedzie na dataset to sygnał rozwojowy, nie replikacja.
- S2 ma osobny, zamrożony protokół i osobny katalog `runs/E016/`; nie
  przedłużamy S1 automatycznie.

## Czego ten eksperyment nie rozstrzyga

- Nie jest replikacją *Cut Less, Fold More* 1:1; to adaptacja operatora.
- Nie daje niezależnego testu ani rezerwy; ocena jest na dev.
- Nie rozstrzyga o innych architekturach/datasetach (Tiny ImageNet,
  ImageNet) ani o publikacji.
- Nie wybiera „najlepszego” poziomu kompresji po dev; poziomy są zamrożone.
- Nie twierdzi, że predyktory są uniwersalne; sprawdzamy je tylko na dwóch
  checkpointach seed42.

## Artefakty i odtwarzanie

Katalog eksperymentu:

- `OPIS.md` — ten protokół,
- `PLAN.md` — krótki plan etapów i kryteriów,
- `configs/stage1.json` — zamrożone wejścia, poziomy, seedy, kryteria,
- `e016_common.py` — wspólne helpery (dane, kalibracja, predyktory),
- `sensitivity_worker.py` — worker jednego sharda na jednym GPU,
- `launch_sensitivity.py` — preflight, snapshot, 4 GPU, 2 fale,
- `analyze_e016.py` — analiza CPU i decyzja C1–C5,
- `check_cpu_e016.py` — preflight bez GPU.

Uruchomienie (Conda `attention`, z katalogu głównego repo):

```bash
CUDA_VISIBLE_DEVICES='' /home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-27_E016_wrazliwosc_foldingu/check_cpu_e016.py

CUDA_VISIBLE_DEVICES=0,1,2,3 /home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-27_E016_wrazliwosc_foldingu/launch_sensitivity.py --smoke

CUDA_VISIBLE_DEVICES=0,1,2,3 /home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-27_E016_wrazliwosc_foldingu/launch_sensitivity.py

/home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-27_E016_wrazliwosc_foldingu/analyze_e016.py \
  --run-dir runs/E016/<S1_RUN> --output-dir runs/E016/<S1_RUN>/analysis
```

Launcher zapisuje snapshoty użytych źródeł i ich SHA-256, `config.json`,
`manifest.json`, `environment.json`, `pip-freeze.txt`, `nvidia-smi.txt`,
przypisanie GPU, seedy, shardy, logi i `artifacts.sha256`. Każdy przebieg ma
unikalny katalog `items/<dataset>/<item>/`. Launcher wymaga czterech wolnych
GPU 0–3 (≥ 4 GiB VRAM, brak aktywnych procesów), środowiska `attention`
i ≥ 3 GiB wolnego dysku. Kod launchera jawnie importuje `os`; poprawka
launchera E015 (`NameError: os`) jest udokumentowana w
`experiments/2026-10-27_E015_cifar100_alokacja/AUDYT_NAPRAWY_LAUNCHERA.md`
i nie jest powielana.

## Ograniczenia protokołu

- Jeden seed42 na dataset; to etap rozwojowy, nie replikacja międzyseedowa.
- Jeden przepis treningowy na dataset; checkpointy nie są niezależne.
- Ocena tylko na dev; test i rezerwa nietknięte.
- Pojedyncza grupa kompresowana przy pozostałych ratio 0 nie odpowiada
  bezpośrednio alokacji globalnej; to pomiar wrażliwości, nie wynik S2.
- Czasy i rozmiary są szacunkami z E014/E015, nie pomiarem E016.
- Deterministyczność dotyczy tego samego sprzętu i środowiska.
