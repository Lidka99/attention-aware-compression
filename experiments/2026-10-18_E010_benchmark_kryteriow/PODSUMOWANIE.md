# E010 — krótkie podsumowanie

**Cel:** porównać 9 kryteriów wyboru kanałów po krótkiej naprawie, na dwóch datasetach.
**Co zrobiono:** Tiny 30% (primary), Tiny 10%, CIFAR-100 10%; T≤20; Holm.
**Wynik:** tylko TAYLOR przeszedł primary na Tiny 30%, T=5: +1,03 pp vs RANDOM; brak replikacji na CIFAR-100 (+0,36 pp, CI obejmuje 0).
**Wniosek:** `survivor_not_replicated`; brak podstaw do nowej metody opartej na tych kryteriach.
**Szczegóły:** [WYNIKI.md](WYNIKI.md), [PROTOKOL_v1.md](PROTOKOL_v1.md).
