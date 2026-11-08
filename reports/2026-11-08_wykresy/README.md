# Wykresy przekrojowe E035–E037

Zestawienie z 2026-11-08. Wszystkie dane pochodzą z istniejących analiz;
skrypt `make_plots.py` nie trenuje i nie otwiera danych/testów.

## Wykresy bezpośrednio w eksperymentach

Poniższe lokalne zestawy mają PNG i PDF, punkty per seed w CSV, skrypt
generujący, `sources.json` z SHA256 oraz `weryfikacja.json`:

- [E034 — PreActResNet18](../../experiments/2026-11-05_E034_preactresnet18/figures/README.md):
  top1 CMA i T−U z punktami seedów, średnią±SD.
- [E035 — niezależna ocena](../../experiments/2026-11-07_E035_ocena_niezalezna/figures/README.md):
  top1, sparowane różnice i porównanie dev/niezależnego zbioru.
- [E036 — koszt](../../experiments/2026-11-07_E036_audyt_kosztu/figures/README.md):
  MAC, przepustowość i latencja batch1 z poprawnym SD0,019–0,100.
- [E037 — kontrola MAC](../../experiments/2026-11-08_E037_kontrola_mac/figures/README.md):
  U/T/U', sparowane T−U' i rzeczywiste budżety parametrów/MAC.

[Audyt wykonania E034–E037](AUDYT_WYKONANIA.md) potwierdza kompletność
lokalnych zapisów i ich przeliczenia. [Dowody w JSON](audit_evidence.json).
Odtworzenie: `python -B reports/2026-11-08_wykresy/audit_zapisow.py` (CPU).

## Pliki

| Plik | Źródło | Co pokazuje |
|---|---|---|
| `e035_quality_dev_test.png` | E035 `analysis.json` | Top-1 CMA: dev (linia przerywana) i test CIFAR/oficjalny val Tiny (ciągła), pięć par×30/50/70%. |
| `e036_cost_ratios.png` | E036 `analysis.json` | Iloraz MAC T/U oraz przepustowości batch128 T/U; ten sam budżet parametrów, inny koszt. |
| `e037_matched_mac.png` | E037 `analysis.json` | Top-1 dev CMA dla U, T i U' (UNIFORM dopasowany MAC); U' ma więcej parametrów. |
| `e037_contrast.png` | E037 `analysis.json` | T−U' po CMA w15 komórkach; wartości dodatnie oznaczają TILT lepszy przy tym samym MAC. |

## Najważniejsze odczyty

- **E035:** kierunek TILT nad U utrzymuje się na dev i teście; CIFAR-100
  ResNet ma mały efekt i gorsze NLL przy50/70%.
- **E036:** TILT ma1,105–1,823× MAC i0,702–0,948× przepustowości b128
  przy dopasowanych parametrach.
- **E037:** gdy UNIFORM ma1,10–1,82× parametrów TILT przy zbliżonym
  MAC, wygrywa na CIFAR-100 i Tiny; TILT pozostaje nieco lepszy głównie
  na CIFAR-10/30–50%. To zawęża twierdzenie o alokacji do budżetu
  **parametrów**, nie obliczeń. Ponieważ zmienia się także pojemność U',
  wynik nie izoluje przyczynowej roli samego compute.

## Odtworzenie

```bash
/home/users/s224574/miniconda3/envs/attention/bin/python -B reports/2026-11-08_wykresy/make_plots.py
```

Wykresy są opisowe; nie zastępują tabel per-seed i audytów w folderach
E035–E037.
