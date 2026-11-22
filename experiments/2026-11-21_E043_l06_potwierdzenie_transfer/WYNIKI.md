# E043 — zamrożony L06 potwierdzony przy 50% redukcji parametrów

**Ukończony 2026-11-22.** Na rezerwie CIFAR-100 L06 poprawił top‑1 względem
UNIFORM o **+1,215 pp (3/4)** przy ~50% redukcji P, spełniając główne kryterium.
Przy ~30% poprawa wyniosła **+0,945 pp (4/4)** — kierunek jest dodatni, lecz
nie osiągnął zamrożonego progu +1,00 pp.

## Wyniki po CMA

ResNet18, te same cztery checkpointy co E042, jedna rezerwa 5 000 obrazów
(50 na klasę), identyczne A4096 dla wszystkich profili. Średnie po seedach
42–45. Koszty podano w [raporcie CPU](PODSUMOWANIE.md) i zmierzono ponownie
przy każdej konstrukcji; P i MAC profili kompresji są w [99,9%, 100%] U.

| Redukcja P | UNIFORM [%] | T-proj [%] | L06 [%] | Strata L06 do nieskompresowanego [pp] | Werdykt L06−U |
|---:|---:|---:|---:|---:|---|
| ~50% (główne) | 53,255 | 53,405 | **54,470** | 19,970 | **Spełnia:** 243 trafienia przy progu 200, 3/4 dodatnie |
| ~30% (transfer) | 66,075 | 66,070 | **67,020** | 7,420 | **Nie spełnia:** 189 trafień przy progu 200, 4/4 dodatnie |

Punkt odniesienia bez kompresji po tej samej CMA: **74,440%**. Rzeczywista
redukcja P L06: **50,299%** w b50 i **30,075%** w b30.

| Budżet | Sparowany kontrast | Δ per seed 42/43/44/45 [pp] | Średnia ± SD [pp] | + / 0 / − |
|---|---|---|---:|---:|
| b50 | L06−UNIFORM | +2,10 / +0,98 / −0,06 / +1,84 | **+1,215 ± 0,975** | 3/0/1 |
| b50 | L06−T-proj | +0,86 / +0,86 / +0,82 / +1,72 | +1,065 ± 0,437 | 4/0/0 |
| b50 | T-proj−UNIFORM | +1,24 / +0,12 / −0,88 / +0,12 | +0,150 ± 0,866 | 3/0/1 |
| b30 | L06−UNIFORM | +1,30 / +1,34 / +0,04 / +1,10 | **+0,945 ± 0,612** | 4/0/0 |
| b30 | L06−T-proj | +2,18 / +1,18 / +0,18 / +0,26 | +0,950 ± 0,937 | 4/0/0 |
| b30 | T-proj−UNIFORM | −0,88 / +0,16 / −0,14 / +0,84 | −0,005 ± 0,713 | 2/0/2 |
| b50 | Nieskompresowany−L06 | +20,36 / +21,32 / +18,28 / +19,92 | +19,970 ± 1,269 | 4/0/0 |
| b30 | Nieskompresowany−L06 | +7,24 / +8,06 / +7,68 / +6,70 | +7,420 ± 0,585 | 4/0/0 |

Straty U/T-proj: **21,185/21,035 pp** w b50 i **8,365/8,370 pp** w b30.
[Pełne kontrasty i straty per seed](../../runs/E043/20261122T000341Z_9e23e96f/analysis/summary.json)
· [Top‑1, liczniki trafień i koszty per seed](../../runs/E043/20261122T000341Z_9e23e96f/analysis/per_seed.csv).

## Wykonanie i audyt

- [Manifest](../../runs/E043/20261122T000341Z_9e23e96f/manifest.json):
  `runs/E043/20261122T000341Z_9e23e96f`, **28/28 ocen, 108,08 s, 4×exit 0**.
  Cztery RTX 2080 Ti, seedy 42–45 → GPU 0–3, Conda `attention`, Python 3.11.15,
  Torch 2.12.0+cu130, CUDA 13.0, sterownik 580.178.04.
- [Audyt PASS](../../runs/E043/20261122T000341Z_9e23e96f/analysis/audit.json):
  zatwierdzone hashe pliku train, indeksów, checkpointów i konfiguracji;
  bramka 4 GPU/CMA przed inferencją, niezmienione stany w `eval()`. Stany po
  CMA teachera i trzech profili b50 **bitowo zgodne z E042 w 16/16** przypadków.
  Rezerwa oceniona raz; bez treningu, fine-tuningu, oceny dev i testu E035.
- [Protokół](OPIS.md) · [Komenda pomiaru](URUCHOMIENIE_GPU.md) ·
  [Odtworzenie audytu bez inferencji](URUCHOMIENIE.md).

## Wniosek

L06 jest potwierdzoną tanią referencją w b50 dla tych checkpointów i dwóch
pułapów. W b30 do progu zabrakło **0,055 pp (11 trafień zsumowanych po
seedach)** mimo poprawy we wszystkich czterech parach. Oba budżety oceniono
na tych samych obrazach i checkpointach; wynik dotyczy generalizacji na
odłożone obrazy i zmiany budżetu, bez uogólniania na nowe modele czy zbiory.
Rezerwa jest odtąd użyta. Ewentualne osobne badanie selektora musi porównywać
go z zamrożonym L06 i mieć własne niewidziane dane oceny.
