# E038 — replay i analiza istniejącego runu

`freeze` z protokołu wykonuje się tylko przy pierwszym przygotowaniu.
Istniejący `frozen.json` zachowujemy; poniższy replay go używa i tworzy
**nowy** run. Nowe skrypty audytu/wykresów są analizą po zakończeniu,
nie zmianą operatora ani kryteriów pierwotnego runu.

```bash
export CONDA_DEFAULT_ENV=attention
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/users/s224574/miniconda3/envs/attention/bin/python
E=experiments/2026-11-08_E038_operator_alokacja
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/experiment.py" check
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$E/experiment.py" run
# R wskazuje NOWY run wypisany przez launcher:
R=runs/E038/<nowy-run-ID>
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/experiment.py" analyze --run "$R"
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/audit.py" "$R"
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/audit_operator.py" "$R"
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/tables.py" "$R"
```

Sama analiza: ostatnie cztery komendy z
`R=runs/E038/20261108T215451Z_a5912565`; bez GPU i ponownej oceny jakości.
Pełny opis czterech GPU, seedów, batcha i wejść: [OPIS.md](OPIS.md).
