# E047 — odtworzenie audytu i analiza

Z głównego katalogu repozytorium, środowisko Conda `attention`:

```bash
export CONDA_DEFAULT_ENV=attention
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/users/s224574/miniconda3/envs/attention/bin/python
E=experiments/2026-11-22_E047_l06_b10_imagenetv2
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/przygotuj.py"
```

Kontrola CPU pobiera przypiętą wersję archiwum (~1,26 GB w pliku tymczasowym
w `/tmp/opencode`), nie wypakowuje plików w repozytorium i nie ocenia modeli.
Zapisuje skrót cache, mapę etykiet i audyt w nowym katalogu CPU; faza GPU
jest przypięta do zatwierdzonego `cpu_20261122T053249Z_169cdc5b`.

Polecenia zatwierdzonej jednorazowej oceny (4 GPU, GPU0–3 → seedy 42–45):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$E/ocena_v2.py" run
RUN='runs/E047/<id-wypisany-przy-run>'
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/ocena_v2.py" analyze "$RUN"
```

`run` przed dostępem do testu wymaga odtworzenia 12 stanów E046 i zgodności
widoków A4096 oraz mapy PID→GPU UUID. Wcześniejsze otwarcie punktacji blokuje
znacznik `runs/E047/imagenetv2_b33_first_use.json`. `analyze` działa na
utrwalonych trafieniach per obraz; porównuje już istniejące agregaty przy
odtworzeniu. Kryterium i ograniczenia: [OPIS.md](OPIS.md).
