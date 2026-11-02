# E018 — krótkie podsumowanie

**Cel:** replikacja tropu cosine z E017 na seedach 43–45.
**Wynik:** **negatywny** — CIFAR-10/50% średnio +0,47 pp, 2/3 seedów
dodatnich; kryterium (+2,0 pp, 3/3) niespełnione. Przy 30% cosine gorsza.
CIFAR-100/50% dodatnia na tych seedach, ale sprzeczna z seed42.
**Wniosek:** standardowa zmiana miary grupowania nie daje stabilnej przewagi;
trop cosine zamknięty. Szczegóły: [WYNIKI.md](WYNIKI.md).
