# E016 — krótki plan

Pełny protokół: [OPIS.md](OPIS.md). Zamrożone wejścia:
[configs/stage1.json](configs/stage1.json).

## Etapy

1. **S1 — diagnostyka pojedynczych grup (uruchamiany teraz).**
   - CIFAR-10 seed42 (E011) i CIFAR-100 seed42 (E015).
   - 12 grup E012 × poziomy 25/50/75/90% + kontrola ratio 0.
   - 2 × (12 × 4 + 1) = **98 unikalnych przebiegów**, bez treningu.
   - Dla każdego: szkoda dev przed/po BN, BN recovery, predyktory:
     głębokość, błąd rekonstrukcji wag, zmiana logitów na stałym
     train-calibration 2 048 obrazów bez etykiet.
   - 4 GPU, 4 niezależne shardy, 2 fale (CIFAR-10, CIFAR-100).
   - Budżet: ≤ 120 min wall-time, ≤ 1,5 GiB wyników, brak testu/rezerwy.

2. **S2 — automatyczna alokacja 50% (zaplanowana, NIE uruchamiana).**
   - Tylko jeśli S1 spełni C1–C5 z OPIS.md.
   - CIFAR-10 i CIFAR-100 seed42; UNIFORM vs TILT+0.30 vs AUTO50.
   - AUTO50 używa wyłącznie predyktora primary (błąd rekonstrukcji wag),
     nie dev. Wspólny target = faktyczne parametry UNIFORM.
   - Osobny zamrożony protokół i osobny katalog `runs/E016/`.

## Bramki kontynuacji S1 → S2

- **C1:** 98/98 completed; kontrole ratio 0 poprawne; brak testu/rezerwy.
- **C2:** przy 75% na obu datasetach rozstęp średniej szkody post-BN
  ≥ 1,0 pp i co najmniej 4 grupy ≥ 0,5 pp.
- **C3:** raport BN recovery; średnie pogorszenie wymaga wyjaśnienia.
- **C4:** primary predictor `weight_reconstruction_rel_error` ma
  |Spearman ρ| ≥ 0,5 ze szkodą post-BN na obu datasetach, ten sam znak.
- **C5:** wall-time ≤ 120 min, wyniki ≤ 1,5 GiB.

Brak C4 = wynik negatywny, bez S2 i bez wybierania innego predyktora po dev.

## Czego nie robimy

- Nie trenujemy nowych modeli i nie dostrajamy na dev.
- Nie używamy oficjalnego testu ani rezerwy.
- Nie obiecujemy nowości ani publikacji.
- Nie uruchamiamy S2 w tym etapie; to jawny następny krok.
