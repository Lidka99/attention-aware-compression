# E044 — L06 przenosi się na Tiny ImageNet przy około 50% redukcji P

**Ukończony 2026-11-22.** Zamrożony L06 daje na Tiny **+2,100 pp względem
UNIFORM, 4/4 seedów** w b50 — główne kryterium kierunku spełnione. W b30
zysk to **+0,255 pp (3/4)** i pozostaje mały wobec rozrzutu między seedami.

## Wyniki

ResNet18/Tiny64, 4 checkpointy E019, oficjalny val 10 000, wspólne A4096
i reset BN + CMA. Tiny nie uczestniczył w wyborze L06 w E042; val był
wcześniej użyty w E035. To transfer na inną konfigurację, na zbiorze oceny
już używanym w projekcie. Nieskompresowany model po CMA: **50,7775%**.

| Redukcja P | UNIFORM [%] | T-proj [%] | L06 [%] | Strata L06 do pełnego modelu [pp] |
|---:|---:|---:|---:|---:|
| ≈50% (główna) | 19,6400 | 20,7100 | **21,7400** | 29,0375 |
| ≈30% (wtórna) | 34,5575 | 34,6725 | **34,8125** | 15,9650 |

| Budżet | Kontrast | Δ per seed 42/43/44/45 [pp] | Średnia ± SD [pp] | + / 0 / − |
|---|---|---|---:|---:|
| b50 | L06−U | +2,87 / +1,73 / +3,11 / +0,69 | **+2,100 ± 1,116** | 4/0/0 |
| b50 | L06−T-proj | −0,29 / +3,96 / +0,11 / +0,34 | +1,030 ± 1,971 | 3/0/1 |
| b50 | T-proj−U | +3,16 / −2,23 / +3,00 / +0,35 | +1,070 ± 2,550 | 3/0/1 |
| b30 | L06−U | −0,90 / +0,27 / +0,99 / +0,66 | **+0,255 ± 0,824** | 3/0/1 |
| b30 | L06−T-proj | −0,64 / −1,53 / +1,32 / +1,41 | +0,140 ± 1,461 | 2/0/2 |
| b30 | T-proj−U | −0,26 / +1,80 / −0,33 / −0,75 | +0,115 ± 1,144 | 1/0/3 |

Dla b50/L06−U **95% przedział sparowanego bootstrapu po obrazach:
[+1,67; +2,52] pp** (10 000 losowań, stratyfikowanych po klasach, te same
indeksy dla wszystkich modeli i seedów). To niepewność względem obrazów
warunkowo na czterech checkpointach, nie przedział między treningami.

W obu budżetach P i MAC wszystkich profili kompresji mieszczą się w
99,9–100% kosztu U. L06 ma **5 618 309 P / 75 312 564 MAC** w b50 i
**7 838 059 P / 103 449 808 MAC** w b30. Dla kontekstu: TILT bez projekcji
z E035 osiągnął 29,000% w b50 na tym samym val, przy 102,24 mln MAC wobec
pułapu U 75,36 mln (zamrożone koszty E037). Przewaga L06 dotyczy porównania
pod oboma pułapami; nawet wtedy 21,74% top‑1 to duża strata bezwzględna.

## Wykonanie i odtworzenie

- [Manifest](../../runs/E044/20261122T022955Z_6c1d2fc5/manifest.json):
  **28/28 ocen, 124,97 s, 4×exit 0**, cztery RTX 2080 Ti w `attention`,
  seedy 42–45 → GPU 0–3. Bez treningu i aktualizacji wag, bez oceny dev.
- [Audyt PASS](../../runs/E044/20261122T022955Z_6c1d2fc5/analysis/audit.json):
  zgodność wejść, szerokości i kosztów, 28/28 CMA, niezmienione stany w `eval`.
  Dodatkowo pełny model i U w b50/b30 mają **12/12 identycznych hashy stanu
  po CMA** względem E035 na Tiny (`ready.json` vs `test/seed_*/resnet_tiny/`
  `controls.json` i `b{50,30}/summary.json` z runu E035).
- [Trafienia i koszty per seed](../../runs/E044/20261122T022955Z_6c1d2fc5/analysis/per_seed.csv)
  · [Agregaty, bootstrap i straty](../../runs/E044/20261122T022955Z_6c1d2fc5/analysis/summary.json)
  · [Protokół](OPIS.md) · [Odtworzenie](URUCHOMIENIE.md).
