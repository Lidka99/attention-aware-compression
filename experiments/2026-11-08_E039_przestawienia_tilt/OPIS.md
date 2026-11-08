# 2026-11-08_E039_przestawienia_tilt — protokół przed oceną

## Pytanie

Czy uporządkowanie profilu TILT według głębokości daje lepszą jakość niż
cztery z góry ustalone przestawienia jego odchyleń? Kontrola alokacji
na istniejących wagach, bez treningu, search i wyboru profilu po wynikach.

## Zamrożony zakres

- ResNet18 na CIFAR-10, CIFAR-100 i Tiny ImageNet64; seedy42–45.
  PreAct nie wchodzi do tej próby: jej13-grupowa reprezentacja wymaga
  związania stem i pierwszego wyjścia residualnego. Nie przenosimy
  arbitralnej permutacji12 grup na inną reprezentację.
- Nominalne redukcje parametrów30/50/70%, budżety U i referencje T z E035.
- Odchylenia `d=0.30*linspace(-1,1,12)`; permutacje indeksów generuje
  `np.random.default_rng(s).permutation(12)` dla **s=101,102,103,104**.
  Ten sam zestaw czterech permutacji na wszystkich zbiorach/budżetach
  i seedach modeli. Cztery permutacje nie są czterema seedami treningu.
- Dla każdej permutacji rozwiązujemy wyłącznie skalar `base` metodą
  bisekcji60 iteracji, `ratio=clip(base+d_perm,0.1,0.9)`, szerokość
  `floor(n*(1−ratio))`, następnie deterministyczne `Geometry.fill` E024.
  Cel: parametry≤P(U), nie mniej niż99,9%P(U). Wagi nie wpływają na dobór.
- Permutujemy **odchylenia przed clippingiem**, nie gotowe liczby kanałów.
  Clipping, nierówne szerokości grup, przesunięcie base i naprawa budżetu
  zmieniają rozkład finalnych ratio. To nie idealna kontrola zachowująca
  histogram finalnych ratio. Jawnie zapisujemy profile i MAC.
- Operator: upstream6946915, ten sam co E035, bias FC zachowany.
- Tylko dotychczasowy dev5000/5000/10000; A4096 train/crop/flip do BN.
  PRE/CMA/repair_bn, batch128, bez aktualizacji gradientowych.
  Test i rezerwa nie są otwierane. Siła TILT+0.30 pozostaje zamrożona.

## Hipoteza i analiza

Primary dla każdej z9 komórek zbiór×budżet po CMA:
`D_seed=Acc(TILT)−mean_4_permutations(Acc(SHUFFLED))`, w pp.
Najpierw średnia po permutacjach w obrębie seeda; agregacja po4 seedach
modelu. Kierunek: mean(D)>0 i≥3/4 dodatnie; łączna bramka H_ORDER wymaga
wszystkich9 komórek. Praktyczna wielkość≥1pp raportowana osobno.

Wtórne: D po repair_bn/PRE, NLL, TILT−każda permutacja osobno,
SHUFFLED−UNIFORM, absolute top1 i parametry/MAC. Raportujemy całą pulę,
bez wybierania najlepszego profilu i bez kolejnego strojenia. Średnie,
SD(ddof=1), wszystkie seedy, opisowe t(df3) przedziały. Brak formalnego
testu permutacyjnego: cztery próbki permutacji nie dają takiej podstawy.
Niespełnienie bramki nie dowodzi równoważności.

Parametry są dopasowane; MAC między profilami nie jest dopasowany.
Różnica nie identyfikuje czystego wpływu kolejności niezależnie od compute.
Dev używano wcześniej: nowa kontrola diagnostyczna, nie niezależny test.

## Kontrole i wykonanie

CPU: odtworzenie uporządkowanego T ze wzoru (identyczne szerokości E035),
komplet36 nowych planów, dopasowanie budżetu, jedno rzeczywiste złożenie
operatora dla każdej pary przy50%/permutacja101 (parametry, MAC, forward).
W runie kontrola teacher we wszystkich trybach i odtworzenie T przy50%
na wszystkich seedach (metryki i stany zgodne E035); hash widoków A zgodny.
Każdy nowy profil: niepuste klastry, rzeczywisty MAC zgodny z geometrią,
parametry zgodne, bias zachowany, brak gradientów i niezmienione wagi BN.

4GPU, seed42/43/44/45→GPU0/1/2/3, jeden proces/seed. 432 nowych ocen
SHUFFLED +36 kontroli teacher +36 odtworzeń T. Limit90min, artefakty<1GiB.
Kolejny eksperyment po ukończonym E038, bez konkurencyjnych zadań GPU.
Protokół, kod i wejścia SHA256 przed oceną; snapshot i środowisko w runie.
Bez treningu: globalny batch treningowy i akumulacja nie dotyczą.

## Odtworzenie

```bash
export CONDA_DEFAULT_ENV=attention
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/users/s224574/miniconda3/envs/attention/bin/python
E=experiments/2026-11-08_E039_przestawienia_tilt
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/experiment.py" freeze
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/experiment.py" check
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$E/experiment.py" run
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/experiment.py" analyze --run runs/E039/<run-ID>
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/audit.py" runs/E039/<run-ID>
```

Każde run ma nowy katalog; błędy zachowane, bez selekcji po jakości.
