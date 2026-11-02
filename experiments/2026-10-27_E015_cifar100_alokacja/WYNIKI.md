# E015 — alokacja budżetu na CIFAR-100: wyniki

Status: **completed**. Trening:
`runs/E015/20261027T190911Z_cb9ffaf_seeds42-45_82c98b12` (4/4 seedy).
Folding: `runs/E015/fold_20261027T210324Z_cb9ffaf_36a01249` (48/48 alokacji).
Analiza: `runs/E015/analizy/20261027T211512Z_audit/analysis.md`.

## Co sprawdziliśmy

Czy TILT+0.30 bije UNIFORM na drugim datasecie (ResNet18/CIFAR-100) według
**zamrożonego** kryterium z [OPIS.md](OPIS.md), na czterech niezależnych
treningach od zera (seedy 42–45), przy 30/50/70% redukcji parametrów i bez
fine-tuningu. Protokół i snapshoty pozostały niezmienione.

## Trening

| seed | 42 | 43 | 44 | 45 | średnia |
|---|---:|---:|---:|---:|---:|
| dev top-1, epoka 200 | 74,94% | 75,10% | 75,04% | 75,86% | **75,24%** |

200 epok, 4 niezależne procesy na GPU0–3, batch 128/GPU (globalny 128 na
przebieg). Podział 40 000 train / 5 000 dev / 5 000 rezerwy; oficjalny test
i rezerwa nietknięte.

## Budżety

Wspólny target to **faktyczna** liczba parametrów UNIFORM: 7 846 668 /
5 581 542 / 3 334 845. TILT+0.30, TILT+0.15 i MAG2 były do niego dopasowane.
TILT miał **maks. 253 parametry mniej** niż UNIFORM (19–253; ≤0,003%);
MAG2 był równy UNIFORM. Żaden wariant nie przekroczył targetu. Faktyczna
redukcja: 30,066% / 50,254% / 70,278% (nominalnie 30/50/70%).

## Wynik główny — TILT+0.30 vs UNIFORM

Średnie dev top-1 po BN:

| Budżet | UNIFORM | TILT+0.30 | TILT+0.15 | MAG2 |
|---:|---:|---:|---:|---:|
| 30% | 66,375% | 67,180% | 67,565% | 59,360% |
| 50% | 53,250% | 55,205% | 55,405% | 40,005% |
| 70% | 29,660% | 31,255% | 32,505% | 14,695% |

Parowane różnice TILT+0.30 − UNIFORM (pp), n=4:

| Budżet | seed42 | seed43 | seed44 | seed45 | średnia | SD | CI 95% | dodatnie |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30% | −0,04 | +1,54 | +0,94 | +0,78 | **+0,805** | 0,651 | [−0,232; +1,842] | 3/4 |
| 50% | +3,40 | +1,72 | +0,36 | +2,34 | **+1,955** | 1,270 | [−0,065; +3,975] | 4/4 |
| 70% | +0,86 | +0,82 | +0,72 | +3,98 | **+1,595** | 1,591 | [−0,937; +4,127] | 4/4 |

CI to przedział t dla średniej parowanych różnic, n=4, df=3, t=3,182;
orientacyjny, bez wnioskowania populacyjnego.

## TILT+0.15 (wtórny) vs UNIFORM

| Budżet | seed42 | seed43 | seed44 | seed45 | średnia | SD | CI 95% | dodatnie |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30% | +1,28 | +1,46 | +0,84 | +1,18 | **+1,190** | 0,261 | [+0,775; +1,605] | 4/4 |
| 50% | +3,18 | +1,56 | +0,50 | +3,38 | **+2,155** | 1,372 | [−0,028; +4,338] | 4/4 |
| 70% | +2,74 | +3,46 | +0,70 | +4,48 | **+2,845** | 1,598 | [+0,302; +5,388] | 4/4 |

TILT+0.15 wypada lepiej od TILT+0.30 na każdym budżecie, ale był wariantem
**wtórnym**. Nie może zastąpić primary po zobaczeniu wyników.

## Kryterium replikacji — ocena

| Warunek (zamrożony) | Wynik |
|---|---|
| średnia > 0 na każdym budżecie | spełniony: +0,805 / +1,955 / +1,595 |
| 50%: średnia ≥ +2,0 pp i ≥3/4 seedów dodatnich | **niespełniony**: +1,955 pp; dodatnie 4/4 |
| 70%: średnia ≥ +2,0 pp i ≥3/4 seedów dodatnich | **niespełniony**: +1,595 pp; dodatnie 4/4 |
| 30%: średnia > 0 i ≥3/4 seedów dodatnich | spełniony: +0,805 pp; dodatnie 3/4 |
| TILT+0.30 nie gorszy od MAG2 o >0,5 pp | spełniony: TILT lepszy o 7,8 / 15,2 / 16,6 pp |

**Główne kryterium replikacji NIE jest spełnione.** Na 50% i 70% średnie
różnice są dodatnie i wszystkie seedy dodatnie, ale poniżej progu +2,0 pp.
11/12 dodatnich par pochodzi z **czterech współdzielonych seedów**
(3 budżety × 4 seedy), a nie z 12 niezależnych replikacji; jedyna ujemna
para to seed42 przy 30%.

Metryki wtórne TILT+0.30 są mieszane: ΔNLL −0,031 / +0,042 / +0,095, a
Δagreement +1,84 / +2,86 / +1,80 pp. Poprawa top-1 przy 50/70% nie idzie
w parze z lepszym NLL, więc nie ma podstaw do mocniejszego twierdzenia
o jakości poza top-1.

## Infrastruktura i audyt

Pierwszy folding `runs/E015/fold_20261027T204519Z_cb9ffaf_4ccfe035`
zakończył się `failed` (`NameError: name 'os' is not defined` w
`environment_for`; ten sam brak dotyczył `terminate`). Poprawka launchera
dodała tylko `import os`, jawną stałą `LAUNCHER_REVISION` i audytowalne
`source_revisions`; nie zmieniono innych źródeł, configów ani wyników.
Audyt: [AUDYT_NAPRAWY_LAUNCHERA.md](AUDYT_NAPRAWY_LAUNCHERA.md).

Udany bieg `fold_20261027T210324Z_cb9ffaf_36a01249`: 4 przygotowania configów
+ **48/48 alokacji**, wszystkie exit 0, GPU0–3. Manifest potwierdza wspólny
target, brak użycia testu i rezerwy oraz zgodność snapshotów.

## Wnioski

- Na CIFAR-100 kierunek TILT+0.30 jest dodatni, ale **zamrożone kryterium
  replikacji primary nie zostało spełnione**; efekt przy 50/70% jest mniejszy
  niż w E014 na CIFAR-10.
- 11/12 dodatnich par nie oznacza 12 replikacji: to ten sam zestaw czterech
  checkpointów przy trzech budżetach.
- TILT+0.15 jest lepszy w tym biegu, ale pozostaje wynikiem wtórnym i nie
  zastępuje primary. Nie zmieniamy progu ani hipotezy po wyniku.
- MAG2 jest znacznie gorszy na każdym budżecie.

## Ograniczenia i następny krok

Jeden dataset (CIFAR-100), ocena na dev, cztery checkpointy z jednego przepisu,
brak niezależnego testu; to adaptacja operatora E012, nie reprodukcja liczb
artykułu 1:1. Rekomendowany następny krok: **analiza mechanizmu i siły tiltu
na istniejących wynikach E014/E015** przed kosztownym rozszerzeniem
(Tiny ImageNet, więcej seedów). Nie ogłaszamy nowych uruchomień.

Szczegóły: [OPIS.md](OPIS.md), [PODSUMOWANIE.md](PODSUMOWANIE.md),
[STATUS.md](STATUS.md), [analiza](../../runs/E015/analizy/20261027T211512Z_audit/analysis.md).
