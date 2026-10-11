# E005 — wynik treningu EfficientNet-B0 od zera na Tiny ImageNet

Data: 2026-10-11 UTC. **Status wykonania: completed.**
Protokół przed uruchomieniem: [PROTOKOL_v1.md](PROTOKOL_v1.md), v1.0.
Bieg: **`runs/E005/20261011T002030Z_3c93ed8_seed42_w7hggu4o/`**,
commit **`3c93ed8`**, dirty=false.

## Wynik końcowy

| Metryka | Wartość |
|---|---:|
| **Dev top-1 (epoka 100, wynik główny)** | **51,53%** (5153/10000) |
| Dev top-1 najlepszy (diagnostyka) | 51,59% (epoka 97) |
| Dev NLL (epoka 100) | 3,2587 |
| Train top-1 (epoka 100) | 98,48% (88615/89984) |
| Train NLL (epoka 100) | 0,0534 |
| Przebieg dev top-1 | 3,0% (ep. 1) → 48,3% (ep. 50) → 51,5% (ep. 100) |

Start 00:20:30 UTC, koniec 03:30:34 UTC — **3 h 10 min** na 4 GPU
(~114 s/epokę). Batch 32/GPU (globalny 128), AMP, SyncBN, DDP.

## Interpretacja

- Baseline jest **znacznie mniej przeuczony niż E001** (CIFAR-100), ale
  wciąż: train 98,5% vs dev 51,5%, dev NLL rośnie od połowy treningu.
  Przyczyny: 200 trudnych, drobnoziarnistych klas, tylko 450 obrazów
  na klasę i efektywne 96×96 (64→96).
- Wynik jest **niższy niż literatura** (Shahriar 2026: ~60,5% scratch
  przy 224×224), co jest zgodne z oczekiwaniem — nasza rozdzielczość to
  świadoma adaptacja. Porównania z literaturą mają charakter jakościowy.
- Dev top-1 nadal delikatnie rósł do końca mimo rosnącego NLL (ten sam
  wzorzec co E001: ranking poprawia się, kalibracja pogarsza).
- Baseline jest wystarczający do badania **względnych** różnic między
  kryteriami scalania (E006), ale do wyników publikacyjnych planowany
  jest osobny baseline 224×224 z mocniejszą recepturą (mixup/cutmix,
  random erasing, label smoothing) — zapisane w docs/DECISIONS.md.

## Dane i wykonanie

- Tiny ImageNet, archiwum SHA-256
  `6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821`;
  podział własny (seed 20261010): train 90 000 (450/klasę), dev 10 000
  (50/klasę); oficjalny val 10 000 **nietknięty jako rezerwa**.
- Cache obrazów: `cache_train.npz`, `cache_val.npz` (hashe w manifeście
  biegu); splits.json SHA-256 zapisany w manifeście.
- Augmentacja: RandomResizedCrop(96, 0,6–1,0) + flip; normalizacja
  mean=std=0,5; dev bez augmentacji. 100 epok, AdamW lr 1e-3, wd 0,01,
  warmup 5, potem cosine. Zero early stoppingu, zero wyboru epoki.
- Kontrole techniczne przed biegiem: smoke `runs/E005/smoke_20261010T235854Z`
  (3 epoki, 2k obrazów) i pilot `runs/E005/pilot_20261011T000010Z`
  (10 epok, 10k, dev 0,5%→8,0%); wcześniejszy nieudany bieg
  `smoke_20261010T235751Z` i przerwany `20261011T001352Z` zachowane.

## Artefakty i hashe

- `artifacts.sha256`: **19/19 plików OK**.

| Artefakt | SHA-256 |
|---|---|
| manifest.json | `a34ef380413b90dcbda7318cf0a0a2a5416c37c8552c37d201280daa540604db` |
| metrics.json | `9b79dfc564be5d723803a72ef485c4d2a4056f2f2c2514f4d359548871bd1218` |
| last.pt | `eb03bb9b9709265134abddceec724243f6a96d2426a30683e2939e0a4531f415` |
| artifacts.sha256 | `d74e62f1b80a9312560cbf52c69555fc5edde41c2e835fbee648de7caf3021cc` |

## Ograniczenia

Jeden seed; dev z tego samego zbioru (oficjalny val nietknięty);
przeuczenie obecne (ogranicza „sufit" dev); brak porównania z literaturą
1:1 (adaptacja rozdzielczości). Nie oceniano testu ani rezerwy.

**Uwaga o losowości (wykryta w przeglądzie po biegu):** w
[train_tiny.py](train_tiny.py) inicjalizacja modelu (linia ~141) następuje
przed `seed_all(cfg['seed'] + rank)` (linia ~147), a seed bazowy nie jest
ustawiony przed budową sieci. Zapisany `last.pt` jest zweryfikowany
hashem i w pełni wystarcza jako baseline do E006, ale **nie twierdzimy,
że ponowny trening z seed=42 odtworzy identyczne wagi**. Kolejny baseline
(224×224) naprawi kolejność seedowania; zakończonego E005 nie zmieniamy.

## Następny krok

[E006 — badanie predykcyjne kryterium z bramek SE](../2026-10-11_E006_kryterium_bramkowe_predykcja/OPIS.md),
na tym właśnie baseline (`last.pt`). Po jego wyniku decyzja o pełnej
kompresji lub o wcześniejszym treningu baseline 224×224 pod publikację.
