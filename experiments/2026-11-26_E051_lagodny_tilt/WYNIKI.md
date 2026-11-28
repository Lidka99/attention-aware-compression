# E051 — TILT poprawia top‑1 przy łagodnej redukcji P

[Run i tabela](../../runs/E051/dev_20261126T213703Z_20c2a1d5/WYNIKI.md) ·
[audyt](../../runs/E051/dev_20261126T213703Z_20c2a1d5/audit.json) ·
[per seed](../../runs/E051/dev_20261126T213703Z_20c2a1d5/per_seed.csv)

**Dev**, checkpointy E048, 56 ocen (w tym 32/32 odtworzenia stanów E049/E050),
4 GPU/exit 0, 238,11 s. Bez treningu, bez ponownego otwierania benchmarków.
Pary mają dopasowane P do 0,1%; Tiny korzysta z kalibracji z E050.

| Zbiór | Cięcie | Pełny [%] | U [%] | TILT [%] | TILT−U [pp] | Strata TILT [pp] | MAC U→TILT [mln] |
|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | b10 | 80,9700 | 77,9850 | 79,7500 | +1,7650 (4/4) | 1,2200 | 494,54 → 536,84 |
| Tiny ImageNet | b10 | 62,6125 | 57,2925 | 59,0050 | +1,7125 (4/4) | 3,6075 | 494,59 → 536,89 |
| CIFAR-100 | b5 | 80,9700 | 79,5600 | 80,4250 | +0,8650 (4/4) | 0,5450 | 523,78 → 546,08 |
| Tiny ImageNet | b5 | 62,6125 | 60,2125 | 61,3250 | +1,1125 (4/4) | 1,2875 | 523,83 → 546,13 |

**b10:** przewaga nad UNIFORM ≥1 pp na obu zbiorach, ale tylko CIFAR‑100
przechodzi limit straty 3 pp. Na Tiny brakuje 0,6075 pp do limitu. W b10
TILT daje ok. 10,4% mniej P i 3,35% mniej MAC niż własny pełny model; liczy
ok. 8,55% więcej MAC niż UNIFORM. Przy b5 redukcja MAC do pełnego wynosi
tylko ok. 1,69%. To łagodniejszy punkt, a nie spełnienie kryterium b10.

[Zamrożony protokół](OPIS.md) · [Odtworzenie](URUCHOMIENIE.md)
