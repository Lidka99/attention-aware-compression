# E008 — krótkie podsumowanie

**Cel:** kompresja 10%/30% na baseline 224 i równa naprawa 60 epok.
**Co zrobiono:** wspólny operator, PRE/POST/RANDOM, bez i z naprawą.
**Wynik:** 30% po naprawie: BASE 59,45; POST 58,32; PRE 57,81; RANDOM 57,55; POST−PRE +0,51 pp (CI obejmuje 0); POST−RANDOM +0,77 pp (na granicy).
**Wniosek:** bez naprawy selekcja > losowa, ale po naprawie przewaga POST słabnie.
**Szczegóły:** [WYNIKI.md](WYNIKI.md), [PROTOKOL_v1.md](PROTOKOL_v1.md).
