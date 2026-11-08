# E037 — przy dopasowanym MAC przewaga TILT znika

**Completed, primary negatywny.** Dev, bez treningu i bez testu.
Run `runs/E037/20261108T031759Z_855c0b33`:4GPU/exit0,292,28s,
180 ocen U'+60 kontroli, audyt PASS. U/T reuse E035.

UNIFORM dopasowany do MAC TILT (U') ma **1,10–1,82× więcej parametrów**.
CMA top1: U' bije U we wszystkich15 komórkach; T−U' jest dodatni tylko
5/15, a bramkę przechodzi tylko R/C10/30 (+0,37pp). Na CIFAR-100/Tiny
U' wygrywa do+23,69pp; na CIFAR-10 TILT pozostaje nieco lepszy przy30/50%
(+0,37/+0,48pp), ale przy70% ResNet przegrywa −2,01pp.

**Wniosek:** zysk TILT przy stałych parametrach jest w dużej mierze
kosztem większego MAC. Nie wolno twierdzić, że alokacja jest lepsza
na jednostkę obliczeń ani pisać „compute-matched”. E034/E035 pozostają
ważne dla stałego rozmiaru sieci; artykuł musi pokazać oba porównania.
Dev jest używany; test po E035 nietknięty w E037. [Pełny raport](WYNIKI.md).
