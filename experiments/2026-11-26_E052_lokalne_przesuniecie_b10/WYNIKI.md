# E052 — mały zysk względem TILT, limit straty na Tiny nadal niespełniony

[Protokół](OPIS.md) · [odtworzenie](URUCHOMIENIE.md) ·
[manifest](../../runs/E052/b10_20261127T001516Z_046b39ba/manifest.json) ·
[audyt](../../runs/E052/b10_20261127T001516Z_046b39ba/audit.json) ·
[per seed](../../runs/E052/b10_20261127T001516Z_046b39ba/per_seed.csv)

**Tiny ImageNet:** S04 odzyskuje **0,2000 pp** względem TILT (3/4), do
**59,2050%**. Strata do pełnego wynosi **3,4075 pp**; do celu 59,6125%
brakuje **0,4075 pp**. Zysk do UNIFORM **+1,9125 pp (4/4)** spełnia próg.
**CIFAR‑100:** **79,8850%**, **+0,1350 pp** względem TILT (2/4), strata
**1,0850 pp**, zysk do UNIFORM **+1,9000 pp (4/4)**. Wszystkie progi na
CIFAR‑100 spełnione, łączne kryterium na dwóch zbiorach niespełnione.

| Zbiór | Wariant | Śr. top‑1 [%] | Strata do pełnego [pp] | P | MAC/obraz |
|---|---|---:|---:|---:|---:|
| Tiny ImageNet | Pełny | 62,6125 | 0,0000 | 11 271 432 | 555 520 000 |
| Tiny ImageNet | UNIFORM | 57,2925 | 5,3200 | 10 104 230 | 494 585 944 |
| Tiny ImageNet | TILT | 59,0050 | 3,6075 | 10 099 600 | 536 890 768 |
| Tiny ImageNet | Frozen offsets | 58,2100 | 4,4025 | 10 104 048 | 494 104 576 |
| Tiny ImageNet | **S04** | **59,2050** | **3,4075** | **10 098 780** | **531 821 776** |
| CIFAR‑100 | Pełny | 80,9700 | 0,0000 | 11 220 132 | 555 468 800 |
| CIFAR‑100 | UNIFORM | 77,9850 | 2,9850 | 10 055 630 | 494 537 444 |
| CIFAR‑100 | TILT | 79,7500 | 1,2200 | 10 052 100 | 536 843 368 |
| CIFAR‑100 | Frozen offsets | 78,9600 | 2,0100 | 10 052 748 | 494 053 376 |
| CIFAR‑100 | **S04** | **79,8850** | **1,0850** | **10 050 480** | **531 773 576** |

S04 daje o **10,4038% / 10,4246%** mniej P niż pełne modele Tiny/CIFAR‑100
i jest w granicy 0,1% P UNIFORM. MAC jest o **0,94% niższy** niż TILT na
obu zbiorach i o **7,53% wyższy** niż UNIFORM. Na Tiny tańszy Frozen
offsets ma 494,10 mln MAC/obraz wobec 531,82 mln dla S04.

| Zbiór | S04 minus referencja [pp] | 42 | 43 | 44 | 45 | Średnia | Dodatnie pary |
|---|---|---:|---:|---:|---:|---:|---:|
| Tiny ImageNet | Pełny | −3,15 | −3,23 | −3,54 | −3,71 | −3,4075 | 0/4 |
| Tiny ImageNet | UNIFORM | +1,94 | +2,08 | +1,91 | +1,72 | +1,9125 | 4/4 |
| Tiny ImageNet | TILT | +0,49 | +0,27 | +0,21 | −0,17 | +0,2000 | 3/4 |
| Tiny ImageNet | Frozen offsets | −0,22 | +2,13 | +1,29 | +0,78 | +0,9950 | 3/4 |
| CIFAR‑100 | Pełny | −0,96 | −1,02 | −0,86 | −1,50 | −1,0850 | 0/4 |
| CIFAR‑100 | UNIFORM | +1,70 | +2,82 | +1,54 | +1,54 | +1,9000 | 4/4 |
| CIFAR‑100 | TILT | +0,22 | +0,42 | −0,06 | −0,04 | +0,1350 | 2/4 |
| CIFAR‑100 | Frozen offsets | +1,10 | +1,48 | +0,64 | +0,48 | +0,9250 | 4/4 |

**Selekcja i audyt.** Kontrola [CPU](../../runs/E052/cpu_20261127T001408Z_ef8e1fa3/manifest.json)
przepuściła sześć profili. Stały podzbiór 8 tys. obrazów train na zbiór
był rozłączny z A4096 i dev. [Reguła NLL](../../runs/E052/b10_20261127T001516Z_046b39ba/selection.json)
wybrała S04 na obu zbiorach (NLL Tiny 0,30723034; CIFAR‑100 0,09341448).
Względem TILT zmiany to `g6: 256→243`, `g8: 256→243`, `g10: 474→482`.
Jeden bieg, 373,40 s; cztery GPU, exit 0 w obu falach, 48 ocen selekcji
i 40 dev, odtworzenie 32/32 kontroli E051 i 8/8 stanów wybranego profilu.
Hashe, trafienia i koszty sprawdzone. Bez uczenia, oficjalnych testów/val,
rezerwy CIFAR‑100 i ImageNetV2.

**Wniosek do manuskryptu.** Krótka kontrola lokalnej alokacji na używanym
wcześniej dev: mały zysk nad TILT przy niższym MAC niż TILT, lecz bez
osiągnięcia limitu 3 pp na Tiny przy b10. Tylko S04 trafił na dev; wynik
nie rozstrzyga o innych profilach. Pula pozostaje zamknięta.
