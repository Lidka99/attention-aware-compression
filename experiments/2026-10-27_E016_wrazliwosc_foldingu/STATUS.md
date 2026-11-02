# E016 — status

**S2 completed (2026-10-28), 8/8 na czterech GPU, exit 0. Wynik negatywny:**
AUTO−UNIFORM +2,58/−10,48 pp (CIFAR-10/100); AUTO−najlepszy TILT
−1,66/−13,88 pp. G1/G2 NIE, G3 TAK. Szczegóły: [WYNIKI_S2.md](WYNIKI_S2.md).
Brak aktywnych obliczeń. Poniżej historia S1 i smoke.

- Protokół S1 1.0 zamrożony w [OPIS.md](OPIS.md); [PLAN.md](PLAN.md) —
  krótki plan.
- CPU preflight: **OK** (`E016_CPU_PREFLIGHT_OK`), bez GPU.
- Smoke 4 GPU: **OK**, 8/8 itemów,
  `runs/E016/smoke_20261027T213755Z_cb9ffaf_a61fab01`.
- S1: **completed**, 98/98 itemów,
  `runs/E016/s1_20261027T215948Z_cb9ffaf_d98f0ed9`;
  CIFAR-10 49/49, CIFAR-100 49/49, 2/2 kontrole.
- Analiza: `runs/E016/analizy/20261027T230435Z_s1/analysis.md`.
- Bramki S1: C1–C5 **TAK**; nie stanowią potwierdzenia AUTO50.
- S2: pierwszy smoke failed po CIFAR-10; naprawa ukończona, nowy smoke
  **8/8 completed na 4 GPU, exit 0**. Nadzór zakończony po 252,8 s.
  Naprawa i ponowna kontrola
  opisana w [NAPRAWA_S2.md](NAPRAWA_S2.md). Pełny S2 zakończono później.
- Klucz: primary `weight_reconstruction_rel_error` ρ=0,800 / 0,702 ze
  szkodą post-BN; `depth_norm` ρ≈−0,011 / 0,443.
- Infrastruktura: dwa przerwane/poprawione biegi zachowane
  (`s1_20261027T213945Z_cb9ffaf_d066bae8` — abort po 20 itemach;
  `s1_20261027T215948Z_cb9ffaf_d98f0ed9` — CIFAR-10 49/49, poprawka
  `expected_controls=1`, resume CIFAR-100). Poprawka E015 `import os`
  sprawdzona.
- Ograniczenia: seed42, dev, bez testu/rezerwy; brak replikacji S2.
- Pełne tabele: [WYNIKI.md](WYNIKI.md).
