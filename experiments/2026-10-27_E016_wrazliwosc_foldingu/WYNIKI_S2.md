# E016 S2 — AUTO50: wynik negatywny (2026-10-28)

## Pytanie i decyzja
Czy zamrożona heurystyka alokacji na podstawie błędu rekonstrukcji wag
poprawia folding przy około 50% redukcji parametrów? **Nie spełnia kryteriów**:
G1 (przewaga nad UNIFORM na obu zbiorach) i G2 (nie więcej niż 0,5 pp
straty do TILT+0.30) niespełnione. G3 (integralność) spełnione.
Nie zmieniamy beta, znaku ani reguły po wyniku. To odrzucenie tej konkretnej
heurystyki, nie wszystkich metod automatycznej alokacji.

## Wynik dev top-1 po korekcie BN [%]

| Zbiór, seed42 | UNIFORM | TILT+0.30 | TILT+0.15 | AUTO50 | AUTO−UNIFORM [pp] | AUTO−najlepszy TILT [pp] |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | 88,98 | 93,22 | 91,76 | 91,56 | +2,58 | −1,66 |
| CIFAR-100 | 53,06 | 56,46 | 56,24 | 42,58 | −10,48 | −13,88 |

To jeden seed na zbiór, nie średnie E014/E015. W szczególności przewaga
TILT+0.30 nad +0.15 tutaj nie zmienia wcześniejszego rankingu średnich E015.
Bez przedziałów międzyseedowych: brak replikacji.

## Budżety i poprawność porównania
- CIFAR-10 target UNIFORM: 5 584 580; AUTO: 5 574 567
  (10 013 mniej, −0,1793% względem targetu).
- CIFAR-100 target: 5 581 542; AUTO: 5 580 397
  (1 145 mniej, −0,0205%). Budżety bliskie, nie identyczne.
- TILT+0.30/+0.15: odpowiednio 79/63 parametry poniżej targetu CIFAR-10,
  108/19 poniżej targetu CIFAR-100.
- Alokacje zapisane przed oceną; solver korzysta z błędu rekonstrukcji wag
  S1, nie z accuracy/NLL dev. S1 dev był już oglądany: etap rozwojowy.
- Jednakowa korekta BN: pełny train 45k/40k, bez gradientów i zmiany wag;
  oficjalny test i rezerwa nietknięte.

## Co zawiodło — obserwacja i hipoteza
AUTO kompresuje `layer1.0.relu1` do 6 kanałów na obu zbiorach; na CIFAR-100
także `layer1.1.relu1` do 6, a wejściową grupę `relu_conv` do 19.
To możliwe wąskie gardła. Nie wykazano jeszcze, że są przyczyną całej straty.
Reguła `clip(a - 0.30*z_g, 0.10, 0.90)` wykorzystuje ranking grup, lecz
nie optymalizuje krzywej szkody ani interakcji grup. Sama standaryzacja nie
uzasadnia amplitudy 0,30 ani dopuszczenia 90% kompresji każdej grupy.

Korelacje przy stałym poziomie kompresji nadal są dodatnie: dla 50%
ρ=0,762 / 0,755, lecz przy 90% maleją do 0,483 / 0,357. Nie dowodzi to
przyczyny porażki. Wniosek: predykcja lokalna nie zapewniła skuteczności
tej globalnej reguły alokacji.

## Wykonanie i odtworzenie
- Run: `runs/E016/s2_20261028T225637Z_cb9ffaf_3cb62653`.
- **8/8 workerów completed**, po cztery na zbiór; GPU0–3 po dwa zadania,
  wszystkie returncode=0. Nadzór exit=0, czas 280,3 s (około 4 min 40 s).
- Analiza: [analysis.md](../../runs/E016/analizy/20261028T234407Z_s2_11249ace/analysis.md).
- Weryfikacja: 63 hashe artefaktów, 15 snapshotów źródeł, 8 specyfikacji.
- Log: `runs/E016/s2_full_20261028T222951Z.log`; stan nadzoru:
  `runs/E016/s2_full_20261028T222951Z.status.json`.
- Komenda: jak w `NAPRAWA_S2.md`, bez `--smoke`, z
  `--timeout-seconds 1800` i unikalnym logiem; jawnie GPU0–3.
- Analiza CPU:
  `python -B experiments/2026-10-27_E016_wrazliwosc_foldingu/analyze_s2.py --run-dir runs/E016/s2_20261028T225637Z_cb9ffaf_3cb62653`.
  Interpreter: Conda attention.

## Następny krok — propozycja, nie uruchomiony eksperyment
Nie replikować tej reguły na kolejnych seedach. Najpierw test przyczynowy:
czy ochrona skrajnie zwężonych grup przy tym samym budżecie usuwa pogorszenie?
Następnie ewentualna alokacja na podstawie krzywych szkody mierzonych na train
(bez etykiet/dev w selekcji), zamiast liniowego przekształcenia rankingu.
Każda nowa reguła wymaga osobnego protokołu; wynik S2 pozostaje negatywny.
