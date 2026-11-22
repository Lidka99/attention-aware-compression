# E040 — wynik negatywny, powrót do artykułu

**Completed,4GPU/exit0, audyt PASS.** Run
`runs/E040/20261111T032100Z_07ac54f7`,3032,72s, bez treningu i plików wag.

Pytanie: czy jawne interakcje par grup po BN pozwalają przewidywać całe
alokacje pod równoczesnymi ograniczeniami parametrów i MAC?

Na16zamrożonych dopuszczalnych alokacjach C, po4seedach:
- C10: MAE addytywny1,1330pp, pary1,9997pp, **1,765× gorzej**,0/4poprawy;
- C100:1,6472pp vs3,8406pp, **2,332× gorzej**,0/4poprawy.

Warunek≤0,80× nie został spełniony na żadnym zbiorze. To MAE predykcji
niezgodności z teacher, **nie** różnica top1. Zgodnie z protokołem badanie
zatrzymano przed dev; przewagi jakości i ablacji top1 nie oceniano.
Testy E035 i rezerwa C100 nie były użyte. Stany były w RAM i są już zwolnione.

**Zamykamy tę próbę bez kolejnych wariantów; wracamy do artykułu E035–E039.**
[Wyniki i ograniczenia](WYNIKI.md), [tabele](TABLES.md), [protokół](OPIS.md).
