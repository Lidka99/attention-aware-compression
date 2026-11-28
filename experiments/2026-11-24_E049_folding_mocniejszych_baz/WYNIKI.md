# E049 — wyniki foldingu na checkpointach E048

[Pełne wyniki](../../runs/E049/eval_20261124T234559Z_b4ca081b/WYNIKI.md) ·
[audyt](../../runs/E049/eval_20261124T234559Z_b4ca081b/audit.json) ·
[per seed](../../runs/E049/eval_20261124T234559Z_b4ca081b/per_seed.csv) ·
[manifest](../../runs/E049/eval_20261124T234559Z_b4ca081b/manifest.json)

Przy dopasowanym P, na wcześniej użytym teście CIFAR-100 i val Tiny,
TILT wygrywa z UNIFORM we wszystkich sześciu ustawieniach b30/b50/b70,
po 4/4 dodatnie pary. W b50 różnice wynoszą **+9,7050 / +9,4625 pp**,
przy **1,366× MAC** UNIFORM na obu zbiorach. [Tabela i koszty](../../paper/tables/tilt_new.csv).

Pod oboma pułapami UNIFORM w b10 Frozen offsets daje **+0,6150 / +0,6750 pp**
(po 4/4), poniżej celu +1 pp. Straty do pełnego modelu: **1,6575 / 4,5700 pp**.
Bieg ukończony, cztery GPU, audyt techniczny PASS.

## Kontrola geometrii przed punktacją

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
