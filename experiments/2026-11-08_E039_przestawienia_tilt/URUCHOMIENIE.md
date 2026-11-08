# E039 — replay i analiza istniejącego runu

`freeze` w protokole dotyczy pierwszego przygotowania. Jeśli `frozen.json`
już istnieje, zachowujemy go i pomijamy tę komendę; skrypt celowo odmawia
nadpisania zamrożonych wejść. Poniższy replay tworzy **nowy** katalog runu.

```bash
export CONDA_DEFAULT_ENV=attention
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/users/s224574/miniconda3/envs/attention/bin/python
E=experiments/2026-11-08_E039_przestawienia_tilt
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/experiment.py" check
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$E/experiment.py" run
# R wskazuje NOWY run wypisany przez launcher:
R=runs/E039/<nowy-run-ID>
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/experiment.py" analyze --run "$R"
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/audit.py" "$R"
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/tables.py" "$R"
```

Sama analiza zakończonego runu: ostatnie trzy komendy z
`R=runs/E039/20261108T223038Z_9d2d039e`. Nie uruchamia GPU i nie ocenia
ponownie modeli. Szczegóły podziału pracy i źródeł: [OPIS.md](OPIS.md).
