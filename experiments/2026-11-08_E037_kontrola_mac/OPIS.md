# 2026-11-08_E037_kontrola_mac — zamrożony protokół

Kontrola kosztowa po E036: czy przewaga TILT nad UNIFORM utrzymuje się,
gdy UNIFORM dostaje **więcej parametrów**, ale nie więcej MAC. Bez treningu,
bez testu i bez nowych danych. Protokół i kod commitowane przed oceną dev.

## Pytanie

E036 pokazał, że przy dopasowanych parametrach TILT ma1,105–1,823× MAC
UNIFORM. Czy zysk jakości TILT wynika z lepszej alokacji, czy głównie
z większego kosztu obliczeń? Odpowiedź: porównanie z UNIFORM dopasowanym
do MAC TILT, kosztem większej liczby parametrów.

## Zakres

- Pięć par E035: R/C10, R/C100, R/Tiny, P/C10, P/C100; seedy42–45.
- Nominalne redukcje parametrów30/50/70%; T_b z zamrożonych planów E035.
- Dla każdej pary/budżetu: **U'_b** = profil UNIFORM o maksymalnej liczbie
  parametrów, dla którego MAC(U'_b)≤MAC(T_b). Wagi i operator E035,
  upstream6946915, bias klasyfikatora zachowany.
- Ocena wyłącznie **dev** (CIFAR5000, Tiny10000), A4096 crop/flip,
  PRE/CMA/`repair_bn`; bez testu, rezerwy i nowych danych.
- Oryginalne U_b/T_b w analizie są **reuse E035 dev** z weryfikacją hashy.
- Bez treningu, fine-tuningu, search i zmiany profili po wynikach.

## Dopasowanie MAC

MAC liczony hookami na rzeczywistym forward CPU (Conv2d/Linear;
1 multiply-add=1MAC; BN/ReLU poza MAC), zgodnie z E036. Dla profilu
UNIFORM szerokości rosną monotonicznie; szukamy największej liczby
parametrów, której MAC nie przekracza MAC(T_b). Wynik zaokrąglony do
pełnych kanałów, więc MAC(U')≤MAC(T), zwykle blisko. Dokładne szerokości,
parametry, MAC i hashe stanów zapisane w `frozen.json` przed oceną.
U' jest większy od T pod względem parametrów; to kontrola **compute**,
nie kontrola rozmiaru sieci.

## Kryteria zamrożone

1. Integralność: pełna siatka15 profili U'×4seedy×3 tryby,4GPU/exit0,
   MAC(U')≤MAC(T), dokładne parametry/szerokości, niepuste klastry,
   skończone metryki, bias zachowany, brak dostępu do testu.
2. Primary CMA T−U': przy30% średnia>0 i≥3/4 dodatnich; przy50/70%
   średnia≥+2,0pp i≥3/4. To progi E034/E035 przeniesione na kontrolę MAC.
3. Secondary: te same kontrasty po `repair_bn`, NLL, top1 bezwzględny;
   raportujemy też oryginalne T−U i U'−U.
4. Jeśli T−U' nie przechodzi, mówimy wprost, że przy dopasowanym MAC
   przewaga TILT znika lub słabnie; nie wybieramy korzystniejszej BN.
5. Dev jest używany wcześniej i nie jest niezależny; test po E035 jest
   użyty i nie będzie otwierany. To kontrola mechanizmu kosztu, nie nowa
   walidacja potwierdzająca.

## Wykonanie i limity

4GPU, seed42/43/44/45→GPU0/1/2/3; jeden proces/seed. 5par×3budżety×4seedy
=60 profiles U' ×3 tryby =180 ocen +60 kontroli teacher. Limit90min,
artefakty<0,5GiB. Przed runem CPU check15 U' i audyt MAC; po runie analyze
i niezależny audit. Jeden run; błędy wykonania zachowane, bez retry
wybierającego wynik. Bez nowego treningu.

## Odtworzenie

```bash
export CONDA_DEFAULT_ENV=attention
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=/home/users/s224574/miniconda3/envs/attention/bin/python
E=experiments/2026-11-08_E037_kontrola_mac
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/experiment.py" check
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$E/experiment.py" run
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/experiment.py" analyze --run runs/E037/<run>
```
