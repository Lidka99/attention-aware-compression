# E014 — replikacja alokacji TILT+0.30 na czterech seedach

Status: **completed** (48/48 workerów, exit0). Run:
`runs/E014/20261027T094707Z_267904d_6a0e93eb`. Seedy E011 42–45.

## Cel

Sprawdzić, czy przewaga TILT+0.30 nad UNIFORM z E013 powtarza się na
czterech niezależnych checkpointach, przy tym samym budżecie i bez
fine-tuningu.

## Wynik (dev top-1 po BN, średnia z 4 seedów)

| Budżet | UNIFORM | TILT+0.30 | TILT+0.15 | MAG2 |
|---:|---:|---:|---:|---:|
| 30% | 93,40% | 94,27% | **94,42%** | 92,26% |
| 50% | 89,60% | **93,38%** | 92,02% | 85,18% |
| 70% | 76,65% | **87,92%** | 84,97% | 61,75% |

TILT+0.30 vs UNIFORM, różnica na poszczególnych seedach:

| Budżet | seed42 | seed43 | seed44 | seed45 | średnia |
|---:|---:|---:|---:|---:|---:|
| 30% | +0,58 | +0,94 | +0,88 | +1,06 | **+0,87 pp** |
| 50% | +4,34 | +3,52 | +3,94 | +3,28 | **+3,77 pp** |
| 70% | +10,70 | +9,56 | +11,16 | +13,68 | **+11,27 pp** |

Wszystkie 12 różnic seed×budżet są dodatnie; każdy przedział bootstrapu
95% dla pojedynczego seeda leży powyżej zera. TILT+0.30 bije MAG2 na
każdym budżecie (średnio +2,0 / +8,2 / +26,2 pp). Przy 30% TILT+0.15
jest minimalnie lepszy od TILT+0.30 (94,42 vs 94,27), ale przy 50/70
wyraźnie gorszy.

## Wniosek

**Wynik E013 replikuje się na wszystkich czterech seedach.** Nierówna
alokacja budżetu jest realnym kierunkiem, nie artefaktem jednego
checkpointu. To nadal adaptacja operatora E012, ocena na dev i cztery
checkpointy z jednego przepisu; brak niezależnego testu i drugiego
datasetu. Następny krok: domknięcie metody, analiza mechanizmu i testy
na CIFAR-100/Tiny ImageNet. Szczegóły: [OPIS.md](OPIS.md).
