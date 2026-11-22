# E041 — odtworzenie testu geometrii

Środowisko Conda `attention`. Z katalogu głównego projektu, po utworzeniu
katalogu nadrzędnego `runs/E041`:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-11-15_E041_kontrola_p_mac/geometry.py
```

Każde uruchomienie tworzy unikalny `runs/E041/cpu_<UTC>_<id>/`, zawierający
manifest, kopię kodu/protokołu, `geometry.json` i tabelę `profiles.csv`.
Skrypt nie importuje Torch, nie otwiera wag ani danych i nie używa GPU.
Odczytuje tylko zamrożone pliki JSON E035/E037/E039, z których wykorzystuje
plany, koszty i profile; nie wykorzystuje pól dawnych wyników jakości.

## Zamrożenie i kontrola operatora na CPU

Pierwsze zamrożenie zostało wykonane. `freeze` odmawia nadpisania istniejącego
`frozen.json`; przy odtwarzaniu zachowujemy ten plik i pomijamy `freeze`.
Historyczna komenda pierwszego zamrożenia:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-11-15_E041_kontrola_p_mac/experiment.py freeze \
  --cpu-run runs/E041/cpu_20261115T232256Z_44602061
```

Kontrola8rzeczywistych konstrukcji operatora na CPU, bez obrazów i oceny
jakości (zarezerwować co najmniej300s limitu polecenia):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-11-15_E041_kontrola_p_mac/experiment.py check
```

Tworzy osobny run `runs/E041/check_<UTC>_<id>/` i aktualny `check.json`.
Każda zmiana plików `.py` lub `OPIS.md` wymaga ponownej kontroli przed GPU.

## Ocena jakości na czterech GPU

Wymaga czterech wolnych GPU0–3, dodatniego check i niezmienionych hashy.
Launcher sprawdza stan urządzeń i brak konkurencyjnych procesów przez
`nvidia-smi`, zapisuje ich modele i telemetrię. Nie przechodzi na mniej GPU.
Seed42/43/44/45→GPU0/1/2/3; limit30min. To komenda nowego uruchomienia,
nie dowód jego wykonania:

```bash
export CONDA_DEFAULT_ENV=attention
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/users/s224574/miniconda3/envs/attention/bin/python
E=experiments/2026-11-15_E041_kontrola_p_mac
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$E/experiment.py" run
# R to NOWY run wypisany przez launcher:
R=runs/E041/<run-ID>
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/experiment.py" analyze --run "$R"
```

`analyze` sprawdza manifest/kody zakończenia, odtworzenie referencji,
profile, faktyczne P/MAC, komplet32kontroli i zapisane dowody niezmienności
wag. Zapisuje `analysis/analysis.json`, `per_seed.csv` i `audit.json`.
Nie wykonuje ponownej inferencji ani nie odtwarza top1 z surowych logitów.
Wyniki geometrii i check CPU nie są wynikami jakości.

## Tabele i wykres z zakończonego runu

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-11-15_E041_kontrola_p_mac/analizy/tabele.py \
  runs/E041/20261116T040008Z_9e34f1e9
```

Wyłącznie zapisane metryki: przeliczenie30kontrastów, TABLES.md/cells.csv,
PNG/PDF pod `figures/<run-ID>/`, `analysis/table_audit.json` w runie.
