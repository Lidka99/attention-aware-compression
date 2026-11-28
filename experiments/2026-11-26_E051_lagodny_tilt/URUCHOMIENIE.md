# E051 — odtworzenie

Z katalogu głównego repozytorium, w Conda `attention`:

```bash
source /home/users/s224574/miniconda3/etc/profile.d/conda.sh
conda activate attention
CUDA_VISIBLE_DEVICES='' python -B experiments/2026-11-26_E051_lagodny_tilt/evaluate.py --check
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B experiments/2026-11-26_E051_lagodny_tilt/evaluate.py
```

Pierwsza komenda liczy tylko geometrię. Druga wymaga czterech wolnych
RTX 2080 Ti i uruchamia cztery procesy z limitem 900 s. Zamrożone
wejścia, manifest, logi, `per_seed.csv`, `audit.json` i `WYNIKI.md`
powstają w nowym katalogu pod `runs/E051/`. Bez zapisu checkpointów.
