# 2026-11-08_E038_operator_alokacja — protokół przed oceną

## Pytanie i zakres

Czy korzyść TILT jest większa dla foldingu niż dla magnitude pruningu?
Porównanie 2×2: FOLD/MAG2 × UNIFORM/TILT, **identyczne szerokości dla
obu operatorów w obrębie profilu**, bez treningu. Zgoda użytkowniczki
2026-11-08 na eksperymenty wzmacniające. Nie zmieniamy siły TILT ani wag.

- Pięć par E035: ResNet18/CIFAR-10, CIFAR-100, Tiny64 oraz
  PreActResNet18/CIFAR-10, CIFAR-100; seedy 42–45.
- Profile U/T i nominalne **redukcje parametrów** 30/50/70% dokładnie z
  `runs/E035/20261107T183831Z_8b0b14e0/frozen.json`.
- Dane: tylko dotychczasowy dev (5000 CIFAR, 10000 Tiny) i nieetykietowane
  A4096 train/crop/flip do BN. Testy i rezerwa nie są otwierane.
- PRE, CMA_reset (primary), oryginalne repair_bn (secondary), batch128.
  Cztery niezależne checkpointy, nie cztery powtórzenia tych samych wag.

## Operator i kontrola zgodności

FOLD to operator upstream6946915 zweryfikowany w E029–E035. Jego metryki
dev są reuse E035; dla obu profili przy50% odtwarzamy je na wszystkich
seedach i parach (top1 dokładnie, NLL tolerancja1e-5, zgodny hash stanu).
Kontrole teacher są również odtwarzane na wszystkich parach i seedach.

MAG2: norma L2 oryginalnych filtrów wyjściowych conv, przed przycięciem
wejść, zgodnie z kryterium upstream. Wspólne osie kanałów są wycinane
jednocześnie. Stem wybierany według conv1, wyjścia stage2–4 według conv2
pierwszego bloku, kanały wewnętrzne według conv1 danego bloku. Stabilny
ranking malejący, remisy: niższy indeks. PreAct `relu_conv` i `relu1_out`
muszą mieć te same szerokości i indeksy (brak projekcji skrótu w layer1).
Wszystkie profile E035 spełniają ten warunek. Bias klasyfikatora zachowany.

Adapter MAG2 jest nową implementacją per grupa; nie nazywamy go
niezmienionym upstream. CPU check obejmuje równość z oficjalnym MAG2 dla
UNIFORM/50% na pięciu parach, brak kompresji, ranking i remisy, reszty,
kształty parametrów oraz MAC FOLD=MAG2 dla każdego profilu. Ta sama BN
oznacza wspólną procedurę i dane, nie kopiowanie buforów między operatorami.
Skalarny keep_ratio do kontroli upstream wyznaczamy z przedziału realizującego
dokładnie U; pomocnicze `ratio` E035/PreAct nie odtwarza całego wektora U.

## Zamrożona analiza

Dla każdego seeda i komórki:

- ΔF=Acc(FOLD,T)−Acc(FOLD,U), ΔM=Acc(MAG2,T)−Acc(MAG2,U).
- **I=ΔF−ΔM** (interakcja operator×alokacja), w pp.
- Primary: I przy50% po CMA, osobno dla pięciu par. Kierunek wspierany,
  gdy mean(I)>0 i co najmniej3/4 dodatnie. Mocny kierunek we wszystkich
  pięciu parach to łączna bramka H_F; osobno liczba efektów≥1pp.
- Równolegle raportujemy kierunek ΔM: korzyść również dla pruningu
  oznacza komponent ogólny alokacji, nie unieważnia ΔF.
- 30/70%, repair_bn, PRE, NLL i różnice FOLD−MAG2: wtórne.
- Średnie, SD(ddof=1), wszystkie sparowane seedy i opisowe przedziały
  t(df3) dla średniej. Progi nie są testami istotności; brak ekwiwalencji
  z niespełnienia H_F. Nie łączymy15 komórek w15 niezależnych replikacji.
- Parametry U/T dopasowane jak E035 (≤0,1% różnicy). Raportujemy MAC;
  między profilami U/T compute nie jest dopasowany. Niska jakość MAG2
  może powodować efekt podłogi, ograniczający interpretację interakcji.

## Wykonanie i odtworzenie

4×GPU: seed42/43/44/45→GPU0/1/2/3, jeden proces/seed, osobne zadania
bez DDP. Brak akumulacji i globalnego batcha treningowego (brak treningu).
360 nowych ocen MAG2 +120 kontroli odtworzenia FOLD +60 teacher;
360 ocen FOLD reuse. Limit90min i artefakty<1GiB.
Protokół, kod i wejścia zamrażamy SHA256 przed oceną; snapshot źródeł,
git HEAD i stan drzewa trafiają do runu. Nie jest wymagany nowy commit.

```bash
export CONDA_DEFAULT_ENV=attention
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/users/s224574/miniconda3/envs/attention/bin/python
E=experiments/2026-11-08_E038_operator_alokacja
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/experiment.py" freeze
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/experiment.py" check
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$E/experiment.py" run
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/experiment.py" analyze --run runs/E038/<run-ID>
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/audit.py" runs/E038/<run-ID>
```

Każde `run` tworzy nowy katalog. Błędy zachowujemy; bez wyboru runu po
jakości. Kontrola na używanym dev jest diagnostyczna, nie niezależnym testem.
