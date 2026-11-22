# E044 — odtworzenie

Z katalogu głównego repozytorium, Conda `attention`:

```bash
export CONDA_DEFAULT_ENV=attention
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/home/users/s224574/miniconda3/envs/attention/bin/python
E=experiments/2026-11-22_E044_l06_tiny
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/kontrola_cpu.py"
```

Każdy replay tworzy nowy `runs/E044/cpu_<UTC>_<id>/`. Skrypt sprawdza
metadane podziałów i wag, dopasowuje P/MAC oraz buduje modele na CPU.
Przy sukcesie `frozen.json` zawiera wektory, koszty, hashe wejść i regułę
późniejszej oceny; `manifest.json` zawiera dowód kontroli i log dostępu.
CPU nie otwiera `cache_train.npz` ani `cache_val.npz`.

## Ocena na 4 GPU i agregacja

W tym samym środowisku i z `PY`, `E` ustawionymi jak powyżej. Poniżej
komendy jednorazowego pomiaru z 2026-11-22:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$E/ocena.py" run
RUN=runs/E044/20261122T022955Z_6c1d2fc5
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/analiza.py" "$RUN"
```

Jedna karta na worker: GPU0–3→seed42–45. Przypisanie potwierdza telemetryczny
PID→UUID GPU. Przed dawaniem punktów wszystkim modelom launcher sprawdza
koszty, CMA, widoki A i wspólną bramkę. Limit 30 min. Jednorazowe otwarcie
punktacji zapisuje `runs/E044/tiny_val_release.json`. W manifeście są hashe
obu cache, kodu i metryk; `correctness.npz` zachowuje jedynie indeksy,
etykiety i binarną poprawność na potrzeby sparowanego bootstrapu.

Wykorzystujemy [zamrożony protokół](OPIS.md) i `frozen.json` z
`runs/E044/cpu_20261122T020819Z_00821a3f/`, SHA-256
`b145cf9d1c0e7e6785963dfe425a4c3df694db6ef8f4422d6ec1bdf70c191908`.

## Odtworzenie analizy bez nowych ocen

Wyniki są już w `analysis/` zakończonego runu. Przeliczenie i porównanie
z zapisami, bez nadpisywania plików:

```bash
RUN=runs/E044/20261122T022955Z_6c1d2fc5
CUDA_VISIBLE_DEVICES='' "$PY" -B -c 'import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); from analiza import audit,paired_bootstrap,summarize,read; p=Path(sys.argv[2]).resolve(); f,rows,correct,labels,proof=audit(p); assert proof==read(p/"analysis/audit.json") and summarize(f,rows,paired_bootstrap(f,correct,labels))==read(p/"analysis/summary.json"); print("E044: audyt i agregaty zgodne")' "$E" "$RUN"
```
