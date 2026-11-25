# E048 — uruchomienie i odtworzenie

Z katalogu głównego repozytorium, w Conda `attention`. Kopia `config.json`, kodu i protokołu oraz ich SHA-256 są w każdym manifeście biegu.

```bash
source /home/users/s224574/miniconda3/etc/profile.d/conda.sh
conda activate attention
E=experiments/2026-11-24_E048_mocniejsze_modele_bazowe
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -B "$E/train.py" check
# Podstaw wypisaną ścieżkę po CHECK=.
CHECK=runs/E048/cpu_…
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B "$E/train.py" smoke --check "$CHECK"
# Podstaw ukończony run po RUN=.
SMOKE=runs/E048/smoke_…
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B "$E/train.py" train --check "$CHECK" --smoke-run "$SMOKE" --detach
```

Run `train_…`: `manifest.json` ma ścieżkę, PID, dwie fale i kody wyjścia; `logs/` rejestruje postęp. `audit.json` podaje kompletność, progi i porównania. Po wykonaniu można ponowić odczyt audytu:

```bash
CUDA_VISIBLE_DEVICES='' python -B "$E/train.py" audit --run-dir runs/E048/train_…
```

Sprawdzenie CPU jest niezależne od treningu. Smoke używa 8 train i 2 dev na klasę, jednej epoki na zbiór oraz wszystkich czterech GPU. Techniczny PASS smoke umożliwia pełny trening. Cztery osobne procesy, jeden model/seed na urządzenie i aktywna tylko jedna fala. Maks. 24 h na dwie fale. `last.pt` zapisuje ostatni punkt pomiaru dla danego seeda i zbioru, nie maksimum dev. W pełnym biegu rejestrujemy pierwszą i co piątą epokę dev, a pełny audyt wymaga wszystkich 240 epok i ośmiu końcowych checkpointów.
