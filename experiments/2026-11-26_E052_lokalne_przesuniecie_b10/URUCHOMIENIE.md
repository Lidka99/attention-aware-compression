# E052 — odtworzenie

Z katalogu głównego repozytorium, w Conda `attention`:

```bash
source /home/users/s224574/miniconda3/etc/profile.d/conda.sh
conda activate attention
CUDA_VISIBLE_DEVICES='' python -B experiments/2026-11-26_E052_lokalne_przesuniecie_b10/evaluate.py --check
```

Kontrola na CPU liczy P i Conv/Linear MAC z szerokości, sprawdza zgodność
z geometrią modelu i kosztami kontroli E051 oraz zapisuje `CPU=…`.
Każdy skonstruowany model ma ponadto sprawdzane rzeczywiste P/MAC przez
forward CPU przed punktacją. Przed startem sprawdzamy procesy (`ps`,
`nvidia-smi`). Dla jednego biegu wpisujemy dokładny katalog `CPU`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B experiments/2026-11-26_E052_lokalne_przesuniecie_b10/evaluate.py --execute runs/E052/cpu_<ID>
```

Skrypt weryfikuje hashe zamrożonych wejść, blokuje powtórne uruchomienie
E052 i wymaga czterech wolnych RTX 2080 Ti. Dwie kolejne fale w jednym
katalogu `b10_*`: selekcja, `selection.json`, pomiar na dev. Limit 900 s
łącznie. Po zakończeniu sprawdzamy `manifest.json` (dwie fale po cztery
exit 0), `audit.json` (odtworzenie stanów, koszty i progi) oraz tabele
`train_per_seed.csv`, `per_seed.csv`. Nie powstają nowe pliki wag.
