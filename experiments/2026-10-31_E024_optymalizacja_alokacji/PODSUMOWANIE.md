# E024 — samodzielna optymalizacja alokacji foldingu

Cel: otwarty problem budżetu per warstwa. Samodzielny kod, CIFAR-100,
4 checkpointy, około50% redukcji parametrów, do50 ocen pełnej sieci/seed.
Cel search: KL teacher→candidate na nieetykietowanym train po BN.

**Wynik primary negatywny:** SEARCH−TILT−0,175pp, 1/4 dodatnich seedów.
SEARCH−UNIFORM+1,235pp, 4/4. KL na niewykorzystanym w wyborze train
i NLL na dev poprawione. Funkcja celu i top1 nie są równoważne.
Kontrola BN obniża trafność nieskompresowanego teacher o5,83pp — wymaga
osobnej diagnostyki przed rozszerzaniem wyszukiwania.

Run `runs/E024/20261031T141558Z_971ba86e`:200 ocen,4GPU,4×exit0,
9,77min, bez treningu/testu/rezerwy. Szczegóły: [WYNIKI.md](WYNIKI.md).
