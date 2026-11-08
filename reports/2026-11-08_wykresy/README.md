# Wykresy przekrojowe E035–E037

Zestawienie z 2026-11-08. Wszystkie dane pochodzą z istniejących analiz;
skrypt `make_plots.py` nie trenuje i nie otwiera danych/testów.

## Pliki

| Plik | Źródło | Co pokazuje |
|---|---|---|
| `e035_quality_dev_test.png` | E035 `analysis.json` | Top-1 CMA dla U/T na dev (linia przerywana) i oficjalnym teście (ciągła), pięć par×30/50/70%. |
| `e036_cost_ratios.png` | E036 `analysis.json` | Iloraz MAC T/U oraz przepustowości batch128 T/U; ten sam budżet parametrów, inny koszt. |
| `e037_matched_mac.png` | E037 `analysis.json` | Top-1 dev CMA dla U, T i U' (UNIFORM dopasowany MAC); U' ma więcej parametrów. |
| `e037_contrast.png` | E037 `analysis.json` | T−U' po CMA w15 komórkach; wartości dodatnie oznaczają TILT lepszy przy tym samym MAC. |

## Najważniejsze odczyty

- **E035:** kierunek TILT nad U utrzymuje się na dev i teście; CIFAR-100
  ResNet ma mały efekt i gorsze NLL przy50/70%.
- **E036:** TILT ma1,105–1,823× MAC i0,702–0,948× przepustowości b128
  przy dopasowanych parametrach.
- **E037:** gdy UNIFORM może użyć1,10–1,82× więcej parametrów w tym samym
  MAC, wygrywa na CIFAR-100 i Tiny; TILT pozostaje nieco lepszy głównie
  na CIFAR-10/30–50%. To zawęża twierdzenie o alokacji do budżetu
  **parametrów**, nie obliczeń.

## Odtworzenie

```bash
/home/users/s224574/miniconda3/envs/attention/bin/python -B reports/2026-11-08_wykresy/make_plots.py
```

Wykresy są opisowe; nie zastępują tabel per-seed i audytów w folderach
E035–E037.
