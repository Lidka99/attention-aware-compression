# E045 — odtworzenie pomiaru łagodnych budżetów

Z katalogu głównego repozytorium, z 4 wolnymi GPU do fazy jakości:

```bash
export CONDA_DEFAULT_ENV=attention
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/users/s224574/miniconda3/envs/attention/bin/python
E=experiments/2026-11-22_E045_lagodny_uniform_tiny
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/eksperyment.py" prepare
CPU_RUN='runs/E045/<cpu_id_wypisany_przez_prepare>'
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$E/eksperyment.py" run "$CPU_RUN"
RUN='runs/E045/<identyfikator_run>'
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/eksperyment.py" analyze "$RUN"
```

Każdy przebieg ma własny katalog, manifest, hashe danych i kodu oraz logi.
Faza CPU buduje dwa modele na syntetycznym wejściu, bez obrazów. Launcher
sprawdza wykorzystanie GPU (PID→UUID; seedy 42–45 po jednym na kartę) oraz
barierę zgodności z historycznym pełnym modelem. Punktacja dev korzysta z
indeksów E035 w cache train; cache val nie jest odczytywany.

Zapis wykonania: `runs/E045/cpu_20261122T030856Z_83060c6f` i
`runs/E045/20261122T031350Z_a58b56a3`. Drugi katalog zawiera gotową
`analysis/`; ponowne generowanie figury z jej plików nie wymaga pomiaru.
[Figura i generator](wykresy/20261122T031350Z_a58b56a3/README.md).
