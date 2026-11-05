# E033 — interakcja późnych grup

**Completed, primary negatywny, secondary dodatni.** 432 oceny+36kontroli,
4GPU/exit0,539,07s; bez treningu/testu/rezerwy. Zgodność LEGACY/UP
dla profili hybrydowych **24/24**; parametry max_abs=0, ARI=1.

CIFAR-100/b70:
- **EP−T: +0,135 pp CMA (2/4)** — primary niespełniony;
  +0,950 pp repair_bn (3/4).
- LC−T: −1,855 CMA / −1,395 repair_bn (eksploracyjne).
- EP−U: +1,755 CMA (3/4) / +1,630 repair_bn (4/4).
- CIFAR-10 i Tiny: EP−T **−7,1 / −5,1 pp** przy b70 CMA, 0/4.

Usunięcie dodatkowej kompresji późnych grup przy dopasowanym budżecie
nie odzyskuje zysku TILT na CIFAR-100 i wyraźnie szkodzi na CIFAR-10/Tiny.
**Prosta hipoteza „CIFAR-100 przez layer4” niepotwierdzona.** Profile są
sprzężone budżetem, więc to nie czysta izolacja późnych grup. Nie budować
na tym nowego alokatora; search pozostaje zamknięty.

Run `runs/E033/20261105T010139Z_f193f1d4`, analiza
`analysis_20261105T011101Z/`; [pełny raport](WYNIKI.md).
