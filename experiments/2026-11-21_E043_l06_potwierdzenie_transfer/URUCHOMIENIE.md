# Odtworzenie E043

## Teraz: audyt i geometria wyłącznie na CPU

Z katalogu głównego repozytorium. Wejścia to zamrożony bank i kopia solvera
E042, plany E035/E037, lokalny upstream E029, cztery checkpointy E015 oraz
metadane splitów i runów. Konkretne ścieżki sprawdza `kontrola_cpu.py`.
Wersje projekcji z E042: Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1.

```bash
export CONDA_DEFAULT_ENV=attention
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/home/users/s224574/miniconda3/envs/attention/bin/python
E=experiments/2026-11-21_E043_l06_potwierdzenie_transfer
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/kontrola_cpu.py"
```

Każde wykonanie tworzy nowy `runs/E043/cpu_<UTC>_<id>/`. Oczekiwany końcowy
komunikat: `CPU PASS 24 modeli skompresowanych + 4 teachery`. `manifest.json`
zapisuje przebieg, P/MAC z hooków, metadane rozłączności, ścieżki odczytanych
plików danych/wag i SHA-256 konfiguracji. `frozen.json` zawiera dokładne wektory
szerokości i przesunięć, wejścia z hashami oraz zamrożoną decyzję dla przyszłej
oceny. Porównaj jego skrót z raportem kontroli CPU. `source/` to snapshot kodu
i protokołu; wcześniejsze artefakty pozostają na ścieżkach wskazanych przez
hashe. Skrypt nie ma trybu uruchomienia oceny jakości.

## Odtworzenie audytu zakończonego pomiaru bez inferencji

Jednorazowy pomiar na czterech GPU wykonano zgodnie z [komendą i instrukcją
GPU](URUCHOMIENIE_GPU.md), po odrębnej zgodzie. Run
`runs/E043/20261122T000341Z_9e23e96f` zawiera zapisane wyniki. Rezerwa jest
wykorzystana; nie otwierać ponownie bramki `reserve_release.json`.

Z istniejących metadanych i liczników można przeliczyć audyt i kontrasty na
CPU, bez nowych plików i bez nadpisywania `analysis/`:

```bash
RUN=runs/E043/20261122T000341Z_9e23e96f
CUDA_VISIBLE_DEVICES='' "$PY" -B -c 'import json,sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); from analiza import audit,summarize; f,rows,proof=audit(Path(sys.argv[2]).resolve()); print(json.dumps({"audit":proof,"summary":summarize(f,rows)},indent=2))' "$E" "$RUN"
```
