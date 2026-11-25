# E049 — uruchomienie

Z katalogu głównego repozytorium. Każda geometria i każde wykonanie mają własną ścieżkę pod `runs/E049/`.

```bash
source /home/users/s224574/miniconda3/etc/profile.d/conda.sh
conda activate attention
E=experiments/2026-11-24_E049_folding_mocniejszych_baz
# Opcjonalnie --smoke-run z ukończonego smoke E048 sprawdza konstrukcje
# ze stanami dopasowanymi przez jedną epokę, bez pomiaru trafności.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -B "$E/evaluate.py" check
# Podstaw ścieżkę po CHECK= oraz run treningowy E048 (może jeszcze trwać).
CHECK=runs/E049/cpu_…
TRAIN=runs/E048/train_…
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B "$E/evaluate.py" queue --check "$CHECK" --training "$TRAIN"
```

`queue` zwraca ścieżkę runu i PID, a proces nadzorujący czeka na konkretny manifest E048. Wystartuje dopiero po technicznym audycie i zwolnieniu wszystkich czterech GPU. Limit oczekiwania 25 h, limit obliczeń E049 30 min. E049 nie czeka na korzystny werdykt trafności E048. Nie wznawia się sam po błędzie.

`manifest.json` przechowuje status i PIDs, `benchmark_release.json` powstaje po pełnym dev, `per_seed.csv` i `WYNIKI.md` po zakończeniu. `audit.json` zawiera przeliczone kontrasty i werdykt. Skrypt nie zapisuje nowych checkpointów; przechowuje stany foldingu w RAM na czas obu faz.
