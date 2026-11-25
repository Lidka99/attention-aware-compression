# E046 — ocena Tiny dev z zamrożonym profilem b10

Z katalogu repozytorium, z czterema wolnymi GPU w Conda `attention`:

```bash
export CONDA_DEFAULT_ENV=attention
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/home/users/s224574/miniconda3/envs/attention/bin/python
E=experiments/2026-11-22_E046_l06_b10_tiny
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$E/ocena_dev.py" run
RUN='runs/E046/<identyfikator_z_run>'
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/ocena_dev.py" analyze "$RUN"
```

Seedy 42/43/44/45 → GPU 0/1/2/3, jeden proces na kartę. Launcher wiąże
`frozen.json` z ukończoną kontrolą CPU (patrz [WYNIKI.md](WYNIKI.md)) i
sprawdza zapis E045. Przed oceną L06 wymaga wszystkich ośmiu dokładnych
odtworzeń stanu po CMA i liczby trafień pełnego modelu oraz U/b10; sprawdza
PID → UUID GPU. Limit 30 minut. Analiza korzysta wyłącznie z artefaktów runu.
Szczegóły danych i kryterium kierunkowego są w [protokole](OPIS.md).
