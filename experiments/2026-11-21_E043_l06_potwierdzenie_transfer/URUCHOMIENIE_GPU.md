# E043 — jednorazowa ocena na rezerwie, po zgodzie

Zgoda w rozmowie: „kontynuuj i uruchom”. Obowiązuje [zamrożony protokół](OPIS.md)
i `frozen.json` z `runs/E043/cpu_20261121T230452Z_8b558297/`, SHA-256
`f84a40c4d4b435495904723aef61664ef11b359de4bee12ecc9160d6d81ded81`.

Z katalogu głównego repozytorium, z dokładnie czterema wolnymi GPU:

```bash
export CONDA_DEFAULT_ENV=attention
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/users/s224574/miniconda3/envs/attention/bin/python
E=experiments/2026-11-21_E043_l06_potwierdzenie_transfer
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$E/ocena_rezerwy.py" run --zgoda-rezerwa
# Podstaw jedyny nowy run wypisany przez launcher (po COMPLETED).
RUN='runs/E043/<identyfikator>'
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/analiza.py" "$RUN"
```

Launcher zapisuje zgody, hashe i snapshot kodu przed wykonaniem. Jeden
worker na GPU: 42→0, 43→1, 44→2, 45→3. Po zweryfikowaniu checkpointów,
geometrii i CMA wszystkich 28 modeli potwierdza przypisanie PID do GPU
w `nvidia-smi` i dopiero wtedy otwiera wspólną bramkę inferencji. Ponowne
otwarcie blokuje `runs/E043/reserve_release.json`. Czas do 30 min, bez
automatycznego restartu. Operacja na pliku oficjalnego `train` jest wspólna
dla A i rezerwy; po sprawdzeniu hashy wybierają je dwie odrębne listy indeksów.
Stany po CMA modeli b50 i teachera są porównywane z zapisanymi hashami E042
jako diagnostyka techniczna niezależna od jakości.

Do sprawdzenia: `manifest.json`, `monitor.jsonl`, `inference_gate.json`,
`seed_*/{status,ready,metrics}.json` i `worker_*.log`. `analiza.py` działa
na zapisanych licznikach, nie otwiera danych ani wag. W `analysis/` zapisuje
audyt, wyniki per seed, kontrasty i werdykty zgodne z kryterium protokołu.
