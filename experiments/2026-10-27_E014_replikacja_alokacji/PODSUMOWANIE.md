# E014 — krótkie podsumowanie

**Cel:** zreplikować TILT+0.30 na czterech niezależnych checkpointach E011.
**Co zrobiono:** 4 seedy × 3 budżety × {UNIFORM, TILT+0.30, TILT+0.15, MAG2}; ten sam operator i BN.
**Wynik:** TILT+0.30 vs UNIFORM: **+0,87 / +3,77 / +11,27 pp** przy 30/50/70%; wszystkie 12 różnic seed×budżet dodatnie.
**Wniosek:** wynik replikuje się; nierówna alokacja to realny kierunek, nie artefakt jednego seeda.
**Ograniczenie:** dev, cztery checkpointy z jednego przepisu, adaptacja operatora; brak drugiego datasetu.
**Szczegóły:** [WYNIKI.md](WYNIKI.md), [OPIS.md](OPIS.md).
