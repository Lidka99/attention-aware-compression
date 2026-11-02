# E015 — krótkie podsumowanie

**Cel:** drugi dataset dla alokacji foldingu — ResNet18/CIFAR-100, cztery
niezależne treningi od zera (seedy 42–45), potem zamrożone porównanie
UNIFORM / TILT+0.30 / TILT+0.15 / MAG2 bez fine-tuningu.
**Co zrobiono:** 4 seedy × 3 budżety × 4 warianty; trening 4/4, folding
48/48 na 4 GPU. Trening dev top-1: 74,94/75,10/75,04/75,86%, średnia
**75,24%**; test i rezerwa nietknięte.
**Wynik primary:** TILT+0.30 vs UNIFORM **+0,81 / +1,96 / +1,60 pp**
(30/50/70%); 11/12 par dodatnich, ale to 4 współdzielone seedy.
**Kryterium:** **NIE spełnione** — przy 50% i 70% średnie poniżej progu
+2,0 pp (1,955 i 1,595), mimo 4/4 seedów dodatnich.
**Wtórny:** TILT+0.15 wypadł lepiej (+1,19 / +2,16 / +2,85 pp), ale nie
zastępuje primary po wyniku.
**Ograniczenie:** dev, jeden dataset, cztery checkpointy z jednego przepisu.
**Szczegóły:** [WYNIKI.md](WYNIKI.md), [OPIS.md](OPIS.md),
[AUDYT_NAPRAWY_LAUNCHERA.md](AUDYT_NAPRAWY_LAUNCHERA.md).
