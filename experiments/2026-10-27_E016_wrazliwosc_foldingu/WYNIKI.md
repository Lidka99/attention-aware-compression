# E016 — etap S1: wyniki diagnostyki wrażliwości na folding

Status: **S1 completed**. Run:
`runs/E016/s1_20261027T215948Z_cb9ffaf_d98f0ed9`.
Analiza: `runs/E016/analizy/20261027T230435Z_s1/analysis.md`.
Etap S2 (**automatyczna alokacja 50%**) **nie został uruchomiony**; jest
tylko rekomendowany przez bramki C1–C5.

## Co sprawdziliśmy

Na istniejących checkpointach CIFAR-10/ResNet18 (E011 seed42) i
CIFAR-100/ResNet18 (E015 seed42) skompresowaliśmy **pojedynczo każdą z 12
grup E012** do 25/50/75/90%, bez treningu i bez fine-tuningu. Dla każdego
przebiegu zmierzyliśmy szkodę na dev **przed** i **po** korekcie BN oraz
trzy predyktory bez etykiet: głębokość, błąd rekonstrukcji wag i zmianę
logitów na stałym train-calibration 2 048 obrazów. Protokół i kryteria były
zamrożone w [OPIS.md](OPIS.md) przed biegiem.

- 2 datasety × (12 grup × 4 poziomy + 1 kontrola) = **98/98 itemów**,
- 4 GPU, 4 niezależne shardy na falę, 2 fale,
- 4 procesy-workerów na falę; każdy item ma unikalny katalog,
- brak oficjalnego testu i brak rezerwy CIFAR-100,
- brak nowego treningu, brak gradientów.

## Kontrole ratio 0

| dataset | pre-BN top1 pp | agreement | recon | wynik |
|---|---:|---:|---:|---|
| CIFAR-10 | 0,000 | 1,000000 | 0,000000 | OK |
| CIFAR-100 | 0,000 | 1,000000 | 0,000000 | OK |

Kontrola jest tożsamościowa poza `num_batches_tracked` (licznik BN nie
wpływa na eval). To potwierdza, że pipeline nie wprowadza sztucznej szkody.

## Szkoda przed/po BN (średnia po 12 grupach)

### CIFAR-10

| poziom | pre-BN pp | post-BN pp | BN recovery pp | recon |
|---:|---:|---:|---:|---:|
| 25% | 0,547 | 0,152 | 0,395 | 0,4003 |
| 50% | 2,978 | 0,683 | 2,295 | 0,6241 |
| 75% | 13,570 | 2,815 | 10,755 | 0,8349 |
| 90% | 33,170 | 9,870 | 23,300 | 0,9559 |

### CIFAR-100

| poziom | pre-BN pp | post-BN pp | BN recovery pp | recon |
|---:|---:|---:|---:|---:|
| 25% | 1,708 | 0,818 | 0,890 | 0,4273 |
| 50% | 10,117 | 2,980 | 7,137 | 0,6562 |
| 75% | 28,982 | 9,535 | 19,447 | 0,8549 |
| 90% | 46,097 | 22,537 | 23,560 | 0,9648 |

**BN odzyskuje większość szkody przed-BN**, ale nie całą. Na CIFAR-100
szkoda post-BN jest wyraźnie większa niż na CIFAR-10, szczególnie przy
75/90%.

## Najbardziej wrażliwe grupy przy 75%

### CIFAR-10 (post-BN, pp)

| grupa | szkoda post-BN | szkoda pre-BN | BN recovery |
|---|---:|---:|---:|
| layer3.0.relu2 | 11,68 | 24,12 | 12,44 |
| layer2.0.relu2 | 9,38 | 46,12 | 36,74 |
| relu_conv (stem) | 4,56 | 34,36 | 29,80 |
| layer3.0.relu1 | 3,06 | 7,30 | 4,24 |
| layer3.1.relu1 | 1,60 | 2,74 | 1,14 |
| ... | ... | ... | ... |
| layer1.0.relu1 | −0,04 | 5,42 | 5,46 |

### CIFAR-100 (post-BN, pp)

| grupa | szkoda post-BN | szkoda pre-BN | BN recovery |
|---|---:|---:|---:|
| layer4.0.relu2 | 30,14 | 36,16 | 6,02 |
| layer3.0.relu2 | 22,10 | 43,14 | 21,04 |
| layer4.1.relu1 | 19,66 | 34,04 | 14,38 |
| layer4.0.relu1 | 12,02 | 17,18 | 5,16 |
| layer2.0.relu2 | 11,24 | 70,24 | 59,00 |
| relu_conv (stem) | 10,60 | 70,94 | 60,34 |
| ... | ... | ... | ... |
| layer1.1.relu1 | −0,34 | 4,18 | 4,52 |

Wniosek opisowy: **wrażliwość nie jest prostą funkcją głębokości**.
Na CIFAR-10 najmocniej szkodzą grupy `layer3`/`layer2`, na CIFAR-100 —
późne grupy `layer4` oraz `layer3.0.relu2`; wczesne grupy mają dużą szkodę
przed-BN, którą BN w dużej części naprawia.

## Predyktory (Spearman ze szkodą post-BN, 48 przebiegów/dataset)

| predyktor | CIFAR-10 ρ | CIFAR-100 ρ |
|---|---:|---:|
| `weight_reconstruction_rel_error` (primary) | **0,800** | **0,702** |
| `calib_logit_rmse_pre_bn` | 0,786 | 0,883 |
| `calib_logit_rmse_post_bn` | 0,790 | 0,974 |
| `depth_norm` | −0,011 | 0,443 |

Rangi grup między datasetami (12 grup):

| metryka | ρ | p |
|---|---:|---:|
| szkoda post-BN | 0,594 | 0,042 |
| `weight_reconstruction_rel_error` | 0,538 | 0,071 |
| `calib_logit_rmse_pre_bn` | 0,832 | 0,001 |
| `calib_logit_rmse_post_bn` | 0,944 | <0,001 |
| `depth_norm` | 1,000 | — |

`depth_norm` ma ρ=1,000 między datasetami tylko dlatego, że jest identyczny
z definicji; **nie** jest to dowód, że przewiduje szkodę. Wewnątrz datasetów
głębokość praktycznie nie przewiduje szkody post-BN na CIFAR-10 (ρ≈0), a na
CIFAR-100 jest słabsza od pozostałych predyktorów.

## Bramki C1–C5

| bramka | wynik |
|---|---|
| C1 — 98/98 completed, 2/2 kontrole, brak testu/rezerwy | **TAK** |
| C2 — rozstęp przy 75% ≥1,0 pp i ≥4 grupy ≥0,5 pp na obu datasetach | **TAK**: 11,72 pp / 8 grup; 30,48 pp / 10 grup |
| C3 — BN recovery raportowany | **TAK**: średnio 9,19 pp (CIFAR-10), 12,76 pp (CIFAR-100) |
| C4 — primary predictor \|ρ\|≥0,5 na obu datasetach, ten sam znak | **TAK**: +0,800 / +0,702 |
| C5 — wall-time ≤120 min, wyniki ≤1,5 GiB | **TAK**: 63,3 min (z przerwą infra), 309 MB |

**S2 rekomendowany: TAK** — ale **nie uruchomiony** w tym etapie.

## Infrastruktura i zdarzenia

- Smoke na 4 GPU: `runs/E016/smoke_20261027T213755Z_cb9ffaf_a61fab01`,
  8/8 itemów, 4 workerów na falę, exit 0.
- Pierwszy pełny launcher został przerwany przez limit czasu powłoki
  (120 s) po zapisaniu 20 itemów CIFAR-10; bieg zachowany jako nieukończony:
  `runs/E016/s1_20261027T213945Z_cb9ffaf_d066bae8` (`ABORTED.md`).
- Drugi launcher ukończył CIFAR-10 49/49, ale miał błąd w weryfikacji:
  oczekiwał 2 kontroli na dataset zamiast 1. Poprawka: `expected_controls=1`.
  CIFAR-10 zachowano, a falę CIFAR-100 uruchomiono w tym samym biegu przez
  udokumentowany resume. Finalny run to
  `runs/E016/s1_20261027T215948Z_cb9ffaf_d98f0ed9`.
- Poprawka launchera E015 (`import os`) jest obecna i sprawdzona w CPU
  preflight; E016 nie powiela tego błędu.

## Ograniczenia

- Jeden seed42 na dataset; to etap rozwojowy, nie replikacja.
- Ocena na dev; oficjalny test i rezerwa nietknięte.
- Pojedyncza grupa przy pozostałych ratio 0 to pomiar wrażliwości, nie
  wynik globalnej alokacji.
- Predyktory sprawdzone tylko na dwóch checkpointach; brak Tiny ImageNet
  i innych architektur.
- Cross-dataset ranga primary predyktora jest umiarkowana (ρ=0,538,
  p=0,071), mimo mocnych korelacji wewnątrz datasetów.
- Nie ma podstaw do twierdzenia o nowości, publikacji ani o uniwersalności
  predyktorów.

## Następny krok (jawny, niewykonany)

S2: automatyczna alokacja 50% z użyciem **primary predyktora**
(`weight_reconstruction_rel_error`) na obu datasetach seed42, porównanie
UNIFORM / TILT+0.30 / AUTO50. To osobny zamrożony protokół i osobny katalog
`runs/E016/`; nie jest częścią tego biegu.
