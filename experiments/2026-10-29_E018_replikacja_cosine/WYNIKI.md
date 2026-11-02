# E018 — replikacja cosine: wynik negatywny

## Uruchomienie
- Run: `runs/E018/20261029T233216Z_cb9ffaf_3740f8f4`.
- **24/24 zadania completed**, cztery GPU 0–3, exit 0, 262,4 s.
- Analiza: `runs/E018/analizy/20261029T233800Z_9ac7cdcc/`.
- Brak treningu wag; test i rezerwa nietknięte.
- Pierwsza próba nadzoru nie wystartowała (zła nazwa pliku launchera);
  nie utworzyła katalogu biegu ani wyników. Poprawiono przed uruchomieniem.

## Wyniki — dev top-1 po BN [%]

| Zbiór | Seed | Budżet | Reference | Cosine | Różnica [pp] |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | 43 | 30% | 89,86 | 87,38 | −2,48 |
| CIFAR-10 | 44 | 30% | 89,42 | 86,14 | −3,28 |
| CIFAR-10 | 45 | 30% | 89,70 | 88,00 | −1,70 |
| CIFAR-10 | 43 | 50% | 73,06 | 71,36 | −1,70 |
| CIFAR-10 | 44 | 50% | 70,86 | 72,52 | +1,66 |
| CIFAR-10 | 45 | 50% | 67,42 | 68,88 | +1,46 |
| CIFAR-100 | 43 | 30% | 51,30 | 50,18 | −1,12 |
| CIFAR-100 | 44 | 30% | 52,78 | 51,66 | −1,12 |
| CIFAR-100 | 45 | 30% | 51,04 | 50,02 | −1,02 |
| CIFAR-100 | 43 | 50% | 21,88 | 25,10 | +3,22 |
| CIFAR-100 | 44 | 50% | 23,22 | 28,08 | +4,86 |
| CIFAR-100 | 45 | 50% | 20,40 | 26,42 | +6,02 |

Parametry po kompresji identyczne: 5 459 866 / 2 797 610 (CIFAR-10)
i 5 492 176 / 2 820 740 (CIFAR-100).

## Decyzja

**Kryterium główne niespełnione.** CIFAR-10, 50% kanałów:
średnia **+0,47 pp**, dodatnia na **2/3** seedach (wymagane: wszystkie
trzy dodatnie i średnia ≥ +2,0 pp). `E018_PRIMARY=NO`.

- **30% kanałów:** cosine konsekwentnie gorsza (średnio −2,49 pp
  na CIFAR-10 i −1,09 pp na CIFAR-100).
- **CIFAR-100, 50%:** +3,22/+4,86/+6,02 (średnio +4,70 pp) — ale na seed42
  w E017 cosine była **−3,72 pp** przy tym samym budżecie. Kierunek zależy
  od seeda, więc nie jest stabilną właściwością zbioru.
- **Wniosek:** trop cosine z E017 **nie replikuje się**. Nie ma podstaw,
  by traktować go jako efekt miary; jest to zmienność międzyseedowa
  w reżimie silnej kompresji (reference na CIFAR-10/50% waha się
  67,42–73,06%, czyli 5,6 pp między seedami).

## Co to oznacza dla projektu

1. **Standardowe alternatywy grupowania nie dają potwierdzonej przewagi**
   nad referencją E012: standardized ≈ reference, BN-effective i cosine
   wypadają gorzej lub niestabilnie.
2. Grupowanie samo w sobie ma znaczenie (random znacznie poniżej reference
   w E017), ale **nie znaleźliśmy lepszej miary** w tym zestawie.
3. To zamyka wątek „inna miara podobieństwa” w obecnej, klasycznej postaci.
   Nowa miara wymagałaby **mechanizmu** (np. atencja/SE) i osobnego protokołu,
   a nie kolejnego wariantu tej samej rodziny.

## Ograniczenia
- Jeden budżet główny i dwa zbiory; architektura tylko ResNet18.
- Dev był wcześniej używany; brak niezależnego testu.
- Wniosek dotyczy **tych** miar: cosine, standaryzacja współrzędnych,
  skala BN. Nie wyklucza innych, nieprzetestowanych reprezentacji.

## Następny krok
Nie kontynuować cosine. Priorytetem staje się skonsolidowanie
**najmocniejszego, powtarzalnego wyniku**: alokacji budżetu (TILT)
na CIFAR-10 (4 seedy, 12/12 dodatnich) i jej ograniczonej replikacji
na CIFAR-100. Brakuje trzeciego zbioru i analizy mechanizmu.
