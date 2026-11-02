# E017 — status

- Protokół 1.1: [OPIS.md](OPIS.md); plan: [PLAN.md](PLAN.md).
- **Etap 1 completed:** 42/42 zadania, cztery GPU, exit 0,
  run `runs/E017/20261029T022532Z_cb9ffaf_f50f95ea`.
- **Kryterium niespełnione** (`E017_CONFIRMED=[]`); standardized ≈ reference,
  BN-effective gorsza; cosine +8,14 pp tylko przy 50% na CIFAR-10 (post hoc).
- Kontrole CPU: reference odtwarza E012 bit w bit; równy budżet parametrów.
- Replikacja cosine wykonana w E018: **negatywna** (+0,47 pp, 2/3 seedów).
  Trop zamknięty; nie stroić miar dalej.
- Test i rezerwa nietknięte; brak treningu wag.
- Szczegóły: [WYNIKI.md](WYNIKI.md), analiza
  `runs/E017/analizy/20261029T225633Z_bfd26c4d/`.
