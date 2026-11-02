# E029 — audyt oficjalnego operatora

**Completed, dodatni wynik audytu**,2026-11-02. Commit upstream
694691556e8314a835178a741938dd3d4d80c171, niezmieniony kod CNN.
CIFAR-10/100,4 seedy,~50% parametrów,7ramion i3stany/model.
168 ocen+24kontrole,4GPU/exit0,3,13min,bez treningu/testu/rezerwy.

LEGACY E012/E014 odtwarza wagi upstream po zachowaniu biasu FC;
LEGACY i LOCAL E024 mają identyczny top1 przy wspólnych profilach/BN.
TILT−UNIFORM na upstream: **+3,995/+1,695pp** z CMA oraz
**+3,745/+1,460pp** z oryginalnym repair_bn (CIFAR-10/100), zawsze4/4.
Losowanie biasu FC nie wyjaśnia przewagi. Brak pełnej reprodukcji paperu:
nasze wagi/dev,2zbiory/1budżet, wspólna kalibracja A4096 crop/flip.

Run `runs/E029/20261102T005941Z_064da5c2`.
[WYNIKI.md](WYNIKI.md). Następny krok: protokół drugiej architektury;
nowy trening wyłącznie po zgodzie użytkowniczki.
