# E037 — brak ogólnej przewagi TILT przy zbliżonym MAC

**Completed, primary negatywny.** Dev, bez treningu i bez testu.
Run `runs/E037/20261108T031759Z_855c0b33`:4GPU/exit0,292,28s,
180 ocen U'+60 kontroli, audyt PASS. U/T reuse E035.

Liczba parametrów UNIFORM dopasowanego do MAC TILT (U') wynosi **1,10–1,82×
liczby parametrów T**.
CMA top1: U' bije U we wszystkich15 komórkach; T−U' jest dodatni tylko
5/15, a bramkę przechodzi tylko R/C10/30 (+0,37pp). Na CIFAR-100/Tiny
U' wygrywa do+23,69pp; na CIFAR-10 TILT pozostaje nieco lepszy przy30/50%
(+0,37/+0,48pp), ale przy70% ResNet przegrywa −2,01pp.

**Wniosek:** przewaga TILT przy stałych parametrach nie przenosi się ogólnie
na porównanie ze szerszym UNIFORM przy zbliżonym MAC. U' ma jednak więcej
parametrów, więc wynik nie izoluje wpływu obliczeń od pojemności modelu.
E034/E035 pozostają ważne dla stałego rozmiaru; artykuł musi pokazać oba
porównania. Dev jest używany; w E037 nie oceniano testu ponownie.
[Pełny raport](WYNIKI.md), [wykresy PNG/PDF i dane](figures/README.md).
