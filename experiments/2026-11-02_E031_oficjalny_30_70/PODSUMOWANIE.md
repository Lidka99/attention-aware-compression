# E031 — oficjalny operator, kompresja30/70%

**Ukończony 2026-11-04, wynik mieszany.** Zgodność LEGACY/LOCAL z upstream
przechodzi24/24 bramki. LEGACY ma identyczne wagi i partycje; LOCAL nie jest
bitowo identyczny (max różnica jakości CMA0,34 pp).

TILT−UNIFORM na upstream, **30/70% redukcji parametrów**:

| Zbiór | CMA [pp] | repair_bn [pp] |
|---|---:|---:|
| CIFAR-10 | +1,025 / +10,910 | +1,035 / +9,260 |
| CIFAR-100 | +0,620 / +1,620 | +0,925 / +0,680 |
| Tiny | +3,780 / +7,7475 | +3,6925 / +7,335 |

**11/12 kryteriów kierunku spełnionych:** CIFAR-100/70%/repair_bn tylko2/4
seedów dodatnie. Łączne kryterium E031 niespełnione. CMA przechodzi6/6,
ale CIFAR-100 słaby, a jakość bezwzględna przy70% niska.

Finalny run `runs/E031/20261104T212951Z_6c3d7356`:504 oceny+36 kontroli,
w tym399+33 reuse i105+3 nowe;4GPU/exit0. Kontynuacja209,85 s.
Protokół i hashe zachowane; test/rezerwa nietknięte. Tiny: operator autorów
na naszym modelu. TILT przy30% różni się od historycznych profili.
[Pełne wyniki i ograniczenia](WYNIKI.md).

Druga architektura wymaga odrębnego kosztorysu/protokołu i wyraźnej zgody
na trening. Decyzja publikacyjna nadal otwarta.
