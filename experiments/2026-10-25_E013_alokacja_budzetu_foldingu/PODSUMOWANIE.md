# E013 — krótkie podsumowanie

**Cel:** sprawdzić, czy nierówna alokacja globalnego budżetu bije jednolity folding.
**Co zrobiono:** 33 warianty na seed42: UNIFORM, TILT±0.15/±0.30, RANDOM, MAG1/MAG2; budżety 30/50/70%.
**Wynik:** TILT+0.30 vs UNIFORM: **+0,58 / +4,34 / +10,70 pp** przy 30/50/70%; CI > 0; MAG2 znacznie gorzej.
**Wniosek:** wyraźny sygnał, że alokacja ma znaczenie; wymaga replikacji (E014).
**Szczegóły:** [WYNIKI.md](WYNIKI.md), [OPIS.md](OPIS.md).
