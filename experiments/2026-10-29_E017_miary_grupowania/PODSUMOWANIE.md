# E017 — krótkie podsumowanie

**Cel:** czy inna miara/grupowanie kanałów poprawia folding przy identycznym
operatorze i identycznych szerokościach warstw?
**Zakres:** CIFAR-10 i CIFAR-100, seed42, 10/30/50% kanałów, 7 wariantów.
**Wynik (42/42, 4 GPU):** kryterium **niespełnione**.
Standardized ≈ reference; BN-effective wyraźnie gorsza. Cosine przegrywa
na CIFAR-100, ale przy 50% na CIFAR-10 daje **+8,14 pp** — obserwacja
post hoc, pojedynczy seed, wymaga replikacji.
**Wniosek:** sama zmiana reprezentacji nie daje potwierdzonej przewagi;
grupy mają jednak duże znaczenie (random znacznie poniżej reference).
Trop cosine sprawdzony w E018 na seedach 43–45: **nie replikuje się**.
**Szczegóły:** [WYNIKI.md](WYNIKI.md), [OPIS.md](OPIS.md), [PLAN.md](PLAN.md).
