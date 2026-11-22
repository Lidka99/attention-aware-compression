# Odtworzenie E042

Z katalogu `badania_atencja_cnn`. Używamy Conda `attention` i istniejących
artefaktów E035/E037/E041 oraz checkpointów E011/E015. Geometria i kontrola
operatora są na CPU, jakość na **czterech** GPU. Jeden worker na seed; na
każdym oba zbiory w kolejności, batch 128. Nie wymaga nowego treningu.

```bash
export CONDA_DEFAULT_ENV=attention
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/users/s224574/miniconda3/envs/attention/bin/python
E=experiments/2026-11-21_E042_zapas_pod_uniform

# Przygotowanie przed pierwszym pomiarem; użyj rzeczywistej ścieżki cpu_….
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/geometria.py"
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/experiment.py" freeze --cpu-run runs/E042/cpu_20261121T211306Z_957fea21
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/experiment.py" check

# Replay: użyj istniejących frozen.json i check.json (bez ponownego freeze).
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$E/experiment.py" run
# Podaj nową ścieżkę z wypisanego RUN; analiza tworzy analysis/ raz w danym runie.
RUN=runs/E042/20261121T212257Z_207ae03f  # przy powtórce podstaw nowy run
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/analiza.py" "$RUN"
```

`manifest.json` i `monitor.jsonl` zapisują status, czas, PID-y, wersje,
telemetrię i kody zakończenia. `source/` zachowuje dokładny protokół i kod;
hash zamrożonej geometrii i hashe wejść są w manifeście i `frozen.json`.
`analysis/` zawiera audyt, wyniki per seed i agregaty profili. Metryki są
wyłącznie z wcześniej używanego dev, przy wykluczonych indeksach rezerwy.
