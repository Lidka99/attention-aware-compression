# E046 — odtworzenie bramki CPU

Z katalogu głównego repozytorium, w środowisku Conda `attention`:

```bash
export CONDA_DEFAULT_ENV=attention
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/home/users/s224574/miniconda3/envs/attention/bin/python
E=experiments/2026-11-22_E046_l06_b10_tiny
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/sprawdz_geometrie.py"
```

Każde uruchomienie tworzy `runs/E046/cpu_<UTC>_<id>/`: kopię protokołu i
kodu, `manifest.json` z wersjami, hashami i kontrolą dostępu, `geometry.json`
z diagnostyką i `construction.log` (jeżeli dochodzi do składania modeli).
`frozen.json` powstaje tylko po przejściu całej bramki. `stopped_geometry`
oznacza zakończoną kontrolę z niespełnioną bramką, `failed` — błąd wykonania
lub audytu. Skrypt nie zawiera trybu oceniania. Późniejsza ocena na czterech
GPU jest opisana w [protokole](OPIS.md).
