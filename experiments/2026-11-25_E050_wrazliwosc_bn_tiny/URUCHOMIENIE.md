# E050 — odtworzenie kontroli BN

Z katalogu głównego repozytorium:

```bash
source /home/users/s224574/miniconda3/etc/profile.d/conda.sh
conda activate attention
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B experiments/2026-11-25_E050_wrazliwosc_bn_tiny/evaluate.py
```

Skrypt używa wyłącznie archiwów E048/E049 wskazanych w kodzie i cache train
Tiny. Wymaga czterech wolnych RTX 2080 Ti; limit części GPU to 900 s.
Pod `runs/E050/<run-ID>/` zapisuje manifest (w tym hashe i komendę), zamrożone
wejścia, logi czterech procesów, `audit.json`, `per_seed.csv` i raport.
Nie zapisuje stanów modeli.
