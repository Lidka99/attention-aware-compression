# E049 — stan przygotowania, 2026-11-24

**Geometria 20/20 PASS.** Niezmieniony operator z adapterem odtworzył profile
na checkpointach smoke E048. Rzeczywiste P/MAC z modeli zgadzają się z planami
na obu architekturach. Sprawdzenie na CPU, bez punktacji i otwierania benchmarków.

| Dane | Profil b10 | P | MAC/obraz |
|---|---|---:|---:|
| CIFAR-100 | UNIFORM | 10 055 630 | 494 537 444 |
| CIFAR-100 | L06 | 10 052 748 | 494 053 376 |
| CIFAR-100 | T-proj | 10 051 974 | 494 077 516 |
| Tiny | UNIFORM | 10 104 230 | 494 585 944 |
| Tiny | L06 | 10 104 048 | 494 104 576 |
| Tiny | T-proj | 10 100 626 | 494 294 552 |

[Manifest CPU](../../runs/E049/cpu_20261124T234044Z_7cb0f8c9/manifest.json) ·
[Geometrie](../../runs/E049/cpu_20261124T234044Z_7cb0f8c9/geometry.json) · [Protokół](OPIS.md).

**Punktacja w kolejce:** [manifest E049](../../runs/E049/eval_20261124T234559Z_b4ca081b/manifest.json).
Automatyczny start po ukończeniu i technicznym audycie E048. Przed ukończeniem
nie ma trafności E049. `WYNIKI.md`, `per_seed.csv` i `audit.json` w katalogu
runu pojawią się po ocenie na dev oraz na wcześniej użytych benchmarkach.
