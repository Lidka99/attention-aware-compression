# E040 — odtworzenie implementacji

Protokół: [OPIS.md](OPIS.md) i [uzupełnienie przed oceną](UZUPELNIENIE_WYKONAWCZE.md).
Polecenia z katalogu głównego repozytorium, Conda attention. Wagi nie są
zapisywane; stany terminalne pozostają w RAM czterech żyjących workerów.
`completed` w manifeście oznacza prawidłowe wykonanie ustalonej procedury,
nie spełnienie kryterium naukowego. `execution_outcome=stopped_C` oznacza
zatrzymanie przed dev po negatywnej bramce predykcji.

```bash
export CONDA_DEFAULT_ENV=attention
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export LOKY_MAX_CPU_COUNT=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/users/s224574/miniconda3/envs/attention/bin/python
E=experiments/2026-11-09_E040_interakcje_po_bn
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/check_cpu.py"
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/experiment.py" freeze
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/experiment.py" check
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$E/experiment.py" run --gpus 0 1 2 3
# R ustaw na dokładny unikalny katalog RUN wypisany przez launcher.
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/audit.py" "$R"
```

`freeze` odmawia nadpisania istniejącego frozen.json. Do odtworzenia
archiwalnego uruchomienia użyć jego snapshotu kodu/wejść i osobnego katalogu
roboczego, nie usuwać obecnego zamrożenia ani nie nadpisywać starego runu.
Zmiana progów, menu lub celu po obejrzeniu wyników nie jest odtworzeniem E040.

CPU check używa wyłącznie syntetycznych wejść do kontroli architektury,
geometrii i zgodności MAG2 z E038. Freeze odczytuje obrazy train A/B, aby
utrwalić hashe widoków; nie ocenia jakości. Worker dopuszcza C dopiero po
globalnej barierze B, a dev tylko po pozytywnej zagregowanej bramce C.
Limit7200s biegnie od startu workerów. Cztery GPU i przypisanie42–45→0–3
są wymagane; nie ma automatycznego zejścia na mniejszą liczbę urządzeń.
