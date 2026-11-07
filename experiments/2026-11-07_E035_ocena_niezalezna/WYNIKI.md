# E035 — niezależna ocena zamrożonej reguły TILT

**Completed, 2026-11-07. Primary kierunek CMA:15/15, każda komórka4/4
dodatnich seedów. Osobny próg≥2pp przy50/70%:9/10.** Bez nowego treningu.
Wyjątek wielkości efektu: ResNet18/CIFAR-100/50%, **+1,3025pp**.
Wtórny kierunek repair_bn:14/15; ResNet18/CIFAR-100/70% tylko2/4 dodatnie.

## Wykonanie, bramka i pochodzenie

- Protokół [OPIS.md](OPIS.md) zamrożony w **f42c0b8**, przed otwarciem danych
  niezależnych. Implementacja, CPU check i zamrożone wejścia: **5a7ec92**.
- Run: `runs/E035/20261107T183831Z_8b0b14e0`, manifest **completed**.
  Dev360 ocen+60kontroli, test360+60; po cztery exit0. Czas **1018,04s**
  (16,97min), dev490,39s, test515,72s; limit5400s.
- 4×RTX2080Ti, seed42/43/44/45→GPU0/1/2/3. Launcher PID960106;
  dev PID960165–960168, test PID961815–961818. Wszystkie zakończone;
  końcowy pomiar GPU: po1MiB i0% użycia. Brak konkurencyjnych zadań GPU.
- Conda attention, Python3.11.15, Torch2.12.0+cu130, CUDA13.0;
  BLAS/OMP/MKL2, deterministyczne algorytmy, TF32 off,
  CUBLAS_WORKSPACE_CONFIG=:4096:8. Batch128 w ocenie i BN.
- CPU check:10 forward/folding kontroli (5par×U/T przy50%) oraz geometria
  wszystkich15 planów, bias zamrożony, pełne niepuste klastry, provenance.
- Globalna bramka dev **PASS**: top1 dokładnie zgodne ze źródłami360+60,
  maksymalna różnica NLL **1,1920928955078125e-7** (próg1e-5).
  Dev ukończony18:46:42 UTC, dane testowe przygotowane18:46:45,
  test wystartował18:46:46, zakończył18:55:22 UTC.
- `analysis/verification.json` **PASS**: pełna siatka, hash kodu/źródeł/
  checkpointów/danych, niezmienione wagi, brak gradientów, bias zachowany,
  profile/klastry zgodne, stany modeli **identyczne bitowo między dev i testem**
  dla każdej konfiguracji, także teacher. Żadnego doboru BN według testu.
- `analysis/data_verification.json` **PASS**: wszystkie10000 obrazów
  i etykiet Tiny cache zgodne z oficjalnymi JPEG/val_annotations, mapowanie
  klas zgodne ze splitem; hash każdego JPEG, zip i metadanych. Osiem wag
  PreAct dodatkowo zgodnych z historycznymi hashami podsumowań treningu E034.
- ResNet30/70: dokładne źródła E031; CIFAR50:E029; Tiny50:E030;
  PreAct wszystkie:E034. Checkpointy E011/E015/E019/E034, bez ich wyboru
  lub nowego treningu. Dokładne szerokości, parametry i SHA256 w `frozen.json`.
  T≤U i T≥99,9%U; budżety oznaczają redukcję parametrów, nie kanałów.
- Operator upstream6946915 niezmieniony. A4096 i augmentacje tylko train,
  Tiny stałe normalizacji train-only E019/E030. Model Tiny to nasz torchvision
  ResNet18/stem64, z operatorem upstream. Nie jest to pełna reprodukcja CLFM1:1.

### Dane niezależne

CIFAR10/100: pełne oficjalne testy po10000. Tiny: oficjalny val10000,
rezerwa odnosząca się do **osobnego cache_val**, indeksy0..9999;
CIFAR100 reserve nieużyta. Tiny val wcześniej otwierano w E005/E007 do
sprawdzenia kształtów, lecz nie oceniano modeli — nie był „nigdy nieotwarty”.

CIFAR10 test_batch brakował. Znaleziono lokalne oficjalne archiwum
`data/cifar10/cifar-10-python.tar.gz`; bez pobierania (`downloaded=false`).
Zweryfikowano MD5 archiwum `c58f30108f718f92721af3b95e74349a` i test_batch
`40351d587109b95175f43aff81a1287e`, wypakowano wyłącznie test_batch,
bez nadpisania train. CIFAR100 istniejący test: MD5
`f0ef6b0ae62326f3e7ffdfab6717acfc`. Hash train sprawdzony przed i po runie.

SHA256:
- archiwum CIFAR10: `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`;
- test_batch: `f53d8d457504f7cff4ea9e021afcf0e0ad8e24a91f3fc42091b8adef61157831`;
- CIFAR100 test: `4b67687d9933c4db8f0831104447f15b93774f4f464bd0516f0f0f2ac83b7864`;
- Tiny cache_val: `fd0c5fc8e45cff072f45f878b60a33d8f0e0af6340fc11d8f937e1c6c75bf07b`.

## Primary CMA: dev obok niezależnej oceny

R=ResNet18, P=PreActResNet18. Top1 w%, różnice i SD wpp. SD próbne
sparowanych różnic (ddof=1). Każdy wiersz testu: **4/4 dodatnie**.
Primary E035 wymaga mean>0 i≥3/4 w każdej z15 komórek, bez progu2pp.

| Para | Redukcja | Dev T−U | Test U | Test T | Test T−U ±SD | Strata T vs teacher CMA |
|---|---:|---:|---:|---:|---:|---:|
| R/C10 | 30 | +1,0250 | 93,0400 | 93,8500 | +0,8100 ±0,0594 | 0,9550 |
| R/C10 | 50 | +3,9950 | 89,2800 | 92,6925 | +3,4125 ±0,2564 | 2,1125 |
| R/C10 | 70 | +10,9100 | 76,7925 | 87,7800 | +10,9875 ±1,4358 | 7,0250 |
| R/C100 | 30 | +0,6200 | 65,7675 | 66,5950 | +0,8275 ±0,2863 | 7,7775 |
| R/C100 | 50 | +1,6950 | 53,4275 | 54,7300 | +1,3025 ±0,7834 | 19,6425 |
| R/C100 | 70 | +1,6200 | 30,6200 | 32,6800 | +2,0600 ±2,4948 | 41,6925 |
| R/Tiny | 30 | +3,7800 | 34,5575 | 38,0950 | +3,5375 ±1,0847 | 12,6825 |
| R/Tiny | 50 | +9,4075 | 19,6400 | 29,0000 | +9,3600 ±1,0830 | 21,7775 |
| R/Tiny | 70 | +7,7475 | 8,3975 | 15,8800 | +7,4825 ±1,2595 | 34,8975 |
| P/C10 | 30 | +0,9850 | 93,1275 | 94,0875 | +0,9600 ±0,1944 | 0,8150 |
| P/C10 | 50 | +4,8800 | 88,5400 | 93,4900 | +4,9500 ±1,5034 | 1,4125 |
| P/C10 | 70 | +13,1900 | 77,6875 | 91,2650 | +13,5775 ±2,4453 | 3,6375 |
| P/C100 | 30 | +4,4350 | 66,4975 | 70,9850 | +4,4875 ±2,1362 | 5,4850 |
| P/C100 | 50 | +7,2250 | 55,9050 | 63,5475 | +7,6425 ±1,2266 | 12,9225 |
| P/C100 | 70 | +23,0950 | 23,0050 | 45,7425 | +22,7375 ±3,0018 | 30,7275 |

**Osobny próg wielkości efektu≥2pp i≥3/4 przy50/70:9/10.** Nie przechodzi
R/C100/50. R/C100/70 przechodzi zaledwie o0,0600pp i ma SD2,4948pp;
dwa seedy dają tylko+0,20/+0,36pp. Nie przedstawiać tego jako silnego,
jednolitego efektu. Historyczny E034 miał mocniejszy próg primary przy50/70;
pozostaje spełniony6/6. E031 zachowuje historyczny mieszany werdykt11/12.

### Teacher na niezależnym zbiorze

| Para | PRE top1 | CMA top1 | repair_bn top1 |
|---|---:|---:|---:|
| R/C10 | 94,7725 | 94,8050 | 94,7850 |
| R/C100 | 74,3325 | 74,3725 | 74,3650 |
| R/Tiny | 50,4475 | 50,7775 | 50,9275 |
| P/C10 | 94,9125 | 94,9025 | 94,9025 |
| P/C100 | 76,4625 | 76,4700 | 76,4950 |

## Secondary: repair_bn, PRE, NLL

| Para | Redukcja | Repair T−U ±SD [pp] | Dodatnie | PRE T−U [pp] | CMA NLL U/T | Repair ΔNLL T−U |
|---|---:|---:|---:|---:|---|---:|
| R/C10 | 30 | +0,8175 ±0,1297 | 4/4 | +3,0750 | 0,248689/0,216460 | −0,031752 |
| R/C10 | 50 | +3,2725 ±0,1797 | 4/4 | +35,0675 | 0,366186/0,254591 | −0,105946 |
| R/C10 | 70 | +9,4150 ±1,7257 | 4/4 | +48,6475 | 0,783369/0,491631 | −0,234519 |
| R/C100 | 30 | +0,7500 ±0,3687 | 4/4 | +3,3550 | 1,478619/1,474713 | +0,004665 |
| R/C100 | 50 | +0,9700 ±0,8103 | 4/4 | +24,9650 | 2,175717/2,244797 | +0,114011 |
| R/C100 | 70 | +1,1650 ±2,3106 | **2/4** | +12,1800 | 3,292815/3,369166 | +0,161726 |
| R/Tiny | 30 | +3,3975 ±1,0551 | 4/4 | +7,6025 | 3,134771/2,898375 | −0,232580 |
| R/Tiny | 50 | +8,9175 ±0,8512 | 4/4 | +13,2575 | 3,921691/3,299915 | −0,596550 |
| R/Tiny | 70 | +7,2900 ±1,3509 | 4/4 | +3,6775 | 4,593334/4,103748 | −0,445751 |
| P/C10 | 30 | +0,9925 ±0,1436 | 4/4 | +1,9625 | 0,277842/0,235587 | −0,042352 |
| P/C10 | 50 | +4,9725 ±1,5744 | 4/4 | +11,7575 | 0,467342/0,259372 | −0,208504 |
| P/C10 | 70 | +13,7350 ±2,3304 | 4/4 | +27,2875 | 0,927602/0,341878 | −0,589174 |
| P/C100 | 30 | +4,6575 ±2,2719 | 4/4 | +9,8675 | 1,394520/1,206446 | −0,188328 |
| P/C100 | 50 | +7,6775 ±1,1343 | 4/4 | +19,7700 | 1,872117/1,546585 | −0,321058 |
| P/C100 | 70 | +22,5325 ±2,9923 | 4/4 | +17,8475 | 3,470895/2,407754 | −1,064248 |

Niższe NLL jest lepsze. **Top1 i NLL nie dają identycznego wniosku:**
R/C100/50 i70 ma gorsze NLL TILT po CMA; po repair_bn R/C100 ma gorsze
średnie NLL przy wszystkich budżetach. Primary nie wybiera korzystniejszej BN.
PRE ma charakter diagnostyczny, bez werdyktu primary.

### Sparowane różnice top1 per seed42/43/44/45

| Para | Redukcja | CMA [pp] | repair_bn [pp] |
|---|---:|---|---|
| R/C10 | 30 | +0,79 / +0,83 / +0,74 / +0,88 | +0,92 / +0,65 / +0,78 / +0,92 |
| R/C10 | 50 | +3,61 / +3,11 / +3,29 / +3,64 | +3,35 / +3,12 / +3,13 / +3,49 |
| R/C10 | 70 | +10,89 / +9,22 / +11,11 / +12,73 | +8,80 / +7,47 / +9,83 / +11,56 |
| R/C100 | 30 | +0,90 / +1,16 / +0,78 / +0,47 | +0,82 / +1,07 / +0,89 / +0,22 |
| R/C100 | 50 | +2,19 / +0,59 / +0,70 / +1,73 | +1,66 / +0,34 / +0,20 / +1,68 |
| R/C100 | 70 | +0,20 / +0,36 / +2,11 / +5,57 | −1,14 / −0,03 / +1,66 / +4,17 |
| R/Tiny | 30 | +3,05 / +3,61 / +5,01 / +2,48 | +2,90 / +3,33 / +4,89 / +2,47 |
| R/Tiny | 50 | +9,87 / +8,56 / +10,64 / +8,37 | +9,35 / +8,47 / +9,87 / +7,98 |
| R/Tiny | 70 | +5,60 / +7,98 / +8,24 / +8,11 | +5,32 / +8,15 / +8,19 / +7,50 |
| P/C10 | 30 | +1,17 / +0,88 / +1,06 / +0,73 | +1,16 / +0,95 / +1,04 / +0,82 |
| P/C10 | 50 | +3,84 / +4,44 / +4,35 / +7,17 | +3,82 / +4,39 / +4,38 / +7,30 |
| P/C10 | 70 | +12,96 / +14,02 / +16,61 / +10,72 | +13,17 / +14,48 / +16,42 / +10,87 |
| P/C100 | 30 | +2,34 / +6,62 / +2,99 / +6,00 | +2,35 / +6,85 / +3,08 / +6,35 |
| P/C100 | 50 | +7,70 / +8,32 / +8,65 / +5,90 | +7,67 / +8,38 / +8,58 / +6,08 |
| P/C100 | 70 | +26,97 / +20,79 / +20,43 / +22,76 | +26,72 / +20,39 / +20,36 / +22,66 |

Pełne absolute top1/NLL, strata względem teacher o tej samej BN, wszystkie
per-seed i SD sparowanych różnic NLL: `analysis/analysis.json` (90 kontrastów),
`analysis/per_seed.csv` (720 wierszy dev+test), `analysis/controls.csv` (120).
Pomiary źródłowe: `{dev,test}/seed_*/<para>/<budżet>/summary.json` i controls.json.

## Wniosek i ograniczenia

**Kierunek przewagi zamrożonej reguły TILT po CMA utrzymuje się na niezależnych
zbiorach we wszystkich15 komórkach.** Mocny efekt na PreAct i Tiny nie usuwa
słabości ResNet/CIFAR100: mały efekt przy50, duży rozrzut przy70, niespójność
repair_bn i gorsze NLL. Ponadto bezwzględna jakość po70% jest niska:
R/C10032,68%, Tiny15,88%, P/C10045,7425%; straty względem teacher CMA
41,6925/34,8975/30,7275pp. Zysk względny nie oznacza zachowania jakości.

Cztery seedy to cztery powtórzenia treningu, nie10000 niezależnych replikacji;
progi są opisowe, nie testami istotności. Dwie pokrewne architektury,
ustalone checkpointy, reguła wybrana wcześniej na dev. Brak dowodu
uniwersalności, optymalności, mechanizmu preaktywacji lub przewagi nad
wszystkimi metodami kompresji. Po E035 te testy/val są już użyte i nie mogą
być przedstawiane jako niewidziana walidacja przyszłego strojenia.

## Błędy wykonania i odtworzenie

Jeden nieudany CPU check przed runem: błędnie założono E029/indices.json.
Poprawiono na rzeczywiste indeksy E028/E024; [DZIENNIK.md](DZIENNIK.md).
**Brak failed runów GPU**, jeden pełny run, bez restartów/selekcji.
Audyt pochodzenia i formatter dodano po ocenie; nie zmieniają kodu oceny,
metryk, kryteriów ani pierwotnego CPU check (snapshot w source/).

Z katalogu projektu:

```bash
export CONDA_DEFAULT_ENV=attention
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/users/s224574/miniconda3/envs/attention/bin/python
E=experiments/2026-11-07_E035_ocena_niezalezna
R=runs/E035/20261107T183831Z_8b0b14e0
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/audit.py" "$R"
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/audit_data.py" "$R"
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/tables.py" "$R"
# Pełny replay tworzy NOWY run; check aktualizuje hashe bieżącego kodu.
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/experiment.py" check
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$E/experiment.py" run
# Do analizy nowego replay podać ścieżkę nowego runu zamiast $R.
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/experiment.py" analyze --run "$R"
```

Oryginalny protokół i kod są zachowane w `source/` runu; finalne raporty
w repozytorium. Lokalne commity bez push. Brak pozostałych blokerów.
