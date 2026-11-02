# E019 — kontrole przed foldingiem

## CPU (`check_cpu_e019.py`, bez GPU i bez dev)

- Dane: `cache_train.npz` (100k, 64×64), split train/dev 90k/10k, rezerwa
  w osobnym `cache_val.npz` (nietknięta). Hashe zgodne.
- Normalizacja policzona **tylko z 90k train**; kontrola odtwarza ją
  z danych z tolerancją 5e-7.
- Model: 11 279 112 parametrów, 200 klas, logity (2, 200).
- Transformacje: train/dev dają (3, 64, 64).
- Folding: 12 grup E012 = grupy E014; UNIFORM 50% = 2 850 280 params
  (deterministyczny fold); wszystkie 4 warianty × 3 budżety mają wspólny
  target = faktyczny UNIFORM; TILT odchyla się maks. 0,003% poniżej targetu.
- Smoke run jest odrzucany przez folding.
- Wynik: `E019_CPU_CHECKS=passed`.

## Zdarzenia przed wynikami
- Bieg `20261030T000316Z...` przerwano po <1 epoce: normalizacja liczyła się
  z 100k (z dev/rezerwy). Nowy bieg `20261030T000950Z...` używa 90k train.
  Przerwany bieg zachowany, bez wyników naukowych.
- Kod foldingu zapisano po starcie treningu; hashe rejestruje manifest
  foldingu, a wejścia treningu są weryfikowane bez zmian.

## Czego nie sprawdzono
- Jakości po treningu i foldingu (to robi bieg na GPU).
