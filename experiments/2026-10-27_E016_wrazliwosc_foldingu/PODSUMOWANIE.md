# E016 — krótkie podsumowanie

**Aktualizacja S2, 2026-10-28: completed, wynik negatywny.** AUTO50 vs
UNIFORM: +2,58 pp na CIFAR-10, −10,48 pp na CIFAR-100; vs najlepszy TILT
−1,66/−13,88 pp. Jeden seed42/dataset, 50% parametrów, 8/8 zadań na 4 GPU.
G1/G2 niespełnione, integralność OK. [WYNIKI_S2.md](WYNIKI_S2.md).
Poniżej podsumowanie wcześniejszej diagnostyki S1.

**Cel:** zmierzyć wrażliwość pojedynczych grup foldingu na CIFAR-10 i
CIFAR-100 (seed42), przed i po BN, oraz sprawdzić predyktory bez etykiet.
**Co zrobiono:** 98/98 itemów (12 grup × 4 poziomy + kontrola × 2 datasety),
4 GPU, bez treningu. Run
`runs/E016/s1_20261027T215948Z_cb9ffaf_d98f0ed9`.
**Wynik:** szkoda post-BN rośnie z poziomem kompresji; BN odzyskuje większość
szkody pre-BN, ale nie całą. Najbardziej wrażliwe grupy to m.in.
`layer3.0.relu2`/`layer2.0.relu2` (CIFAR-10) oraz `layer4.0.relu2`/
`layer3.0.relu2` (CIFAR-100).
**Predyktory:** primary `weight_reconstruction_rel_error` ma ρ=0,800
(CIFAR-10) i 0,702 (CIFAR-100) ze szkodą post-BN; `depth_norm` praktycznie
nie przewiduje szkody na CIFAR-10 (ρ≈0) i jest słabszy na CIFAR-100.
**Bramki:** C1–C5 spełnione; S2 przygotowany, pierwszy smoke failed.
Naprawa techniczna ukończona: nowy smoke **8/8, 4 GPU, exit 0**;
szczegóły w [NAPRAWA_S2.md](NAPRAWA_S2.md).
Pełny wynik AUTO50 zapisano powyżej; łączne korelacje S1 nie dowodzą przewagi alokacji.
**Ograniczenie:** jeden seed42/dataset, dev, brak testu/rezerwy.
**Szczegóły:** [WYNIKI.md](WYNIKI.md), [OPIS.md](OPIS.md),
[analiza](../../runs/E016/analizy/20261027T230435Z_s1/analysis.md).
