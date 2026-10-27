# E003 — krótkie podsumowanie

**Cel:** fizyczna kompresja 10% parametrów EfficientNet-B0+SE na CIFAR-100.
**Co zrobiono:** PRE/POST/RANDOM ze wspólnym operatorem, bez dostrajania.
**Wynik:** BASE 66,84%; PRE 54,94%; POST 55,24%; RANDOM 49,48%; POST−PRE +0,30 pp (CI obejmuje 0).
**Wniosek:** duża szkoda bez naprawy; brak wyraźnej przewagi kryterium atencji.
**Szczegóły:** [WYNIKI.md](WYNIKI.md), [PROTOKOL_v1.md](PROTOKOL_v1.md).
