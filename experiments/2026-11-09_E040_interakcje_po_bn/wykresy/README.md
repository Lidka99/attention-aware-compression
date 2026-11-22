# E040 — wykres predykcji szkody

- `e040_mae.png` / `e040_mae.pdf`: MAE modelu addytywnego i modelu par po BN;
  słupki to średnie4seedów, odcinki SD(ddof=1), połączone punkty to sparowane
  seedy. Nie jest to wykres trafności top1 ani przedział ufności.
- `evidence.json`: hashe źródłowego audytu/CSV, skryptu, tabel i wykresów.
- `podsumuj.py`: wyłącznie opisowa analiza po zakończeniu pilota; bez nowych
  ocen modeli, etykiet i doboru wariantów. Nie zmienia zamrożonego kodu pomiarów.

Odtworzenie z katalogu głównego repozytorium:

```bash
CUDA_VISIBLE_DEVICES='' /home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-11-09_E040_interakcje_po_bn/wykresy/podsumuj.py \
  runs/E040/20261111T032100Z_07ac54f7
```

Pełne źródła liczb i ograniczenia: [WYNIKI.md](../WYNIKI.md).
