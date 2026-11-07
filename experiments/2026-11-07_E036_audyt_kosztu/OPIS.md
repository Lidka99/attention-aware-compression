# 2026-11-07_E036_audyt_kosztu — zamrożony protokół

Audyt opisowy istniejących TILT+0.30 vs UNIFORM przy matched params E035.
Bez treningu, danych rzeczywistych, kalibracji BN, oceny jakości, strojenia
profili lub selekcji runu. Protokół i kod commitowane przed pomiarami GPU.

## Zakres i kryterium opisowe

R/C10, R/C100, R/Tiny, P/C10, P/C100; redukcje parametrów30/50/70;
seedy42–45. Teacher raz na parę/seed, U i T dla każdego budżetu:140 modeli.
Dokładne profile, checkpointy i operator E035 (upstream6946915), hashy
nie zmieniamy. Odtwarzamy PRE i porównujemy stan bitowo z dowodem E035.
E035 nie zachował modeli CMA; bez kalibracji na danych nie odtwarzamy ich
BN. PRE i CMA mają tę samą geometrię/operator; pomiar PRE nie jest pomiarem
bitowo identycznego stanu CMA. Jakość CMA wyłącznie reuse wyników E035,
nie nowa walidacja. Teacher także PRE.

Zamrożone kryterium: kompletna siatka i audyt zgodności, a następnie
opis tradeoff T/U MAC, latency, throughput, pamięć vs istniejąca jakość.
Brak hipotezy skuteczności, progów przewagi, testów istotności i twierdzeń
matched FLOPs. Nie zmieniamy profili po poznaniu kosztów.

## Definicje kosztu

Parametry: wszystkie tensorowe parametry, także affine BN/bias, bez buforów.
MAC na obraz: Conv2d Hout*Wout*Cout*(Cin/groups)*Kh*Kw; Linear Nin*Nout.
Jedno multiply-add=1MAC. Bias, BN, ReLU, pooling, residual add i normalizacja
nie należą do MAC; BN i ReLU raportowane osobno jako liczba elementów
(nie FLOPs). MAC sprawdzany dwiema drogami: ręczna propagacja geometrii
grafu ResNet/PreAct z kształtów wag oraz hooki rzeczywistego forward CPU.
Każda warstwa conv/linear musi zgadzać się dokładnie. Parametry także
sprawdzane ze frozen E035. Runtime zawiera pełny forward, w tym BN/ReLU.

## Runtime i pamięć

Conda attention,4GPU, seed42/43/44/45→fizyczne GPU0/1/2/3; brak konkurencji.
Jeden proces/seed/GPU, modele kolejno. FP32, TF32 off (matmul i cuDNN),
autocast off; eval/inference_mode, NCHW contiguous, bez compile/fuzji BN.
torch deterministic algorithms=True; cudnn enabled=True, benchmark=False,
deterministic=True. OMP/MKL/OPENBLAS=2; CUBLAS_WORKSPACE_CONFIG=:4096:8.

Wejścia CPU randn, osobny generator seed=360000+rozmiar+batch; identyczne
na seedach i metodach, hash zapisany; bez host-device transfer w pomiarze.
Batch1 latency i batch128 throughput; OOM kończy run, bez zmiany batch.
50 warmup i200 forwardów na serię,3 serie dla każdego modelu/batch.
Każdy forward osobną parą CUDA events, end.synchronize(); synchronizacja
także przed serią. Raport mediana/p10/p90 z600 zdarzeń per seed/model/batch;
samples/s=1000*batch/średni czas zdarzenia. To steady-state device-forward
throughput z synchronizacją po każdym batchu, bez loadera/transferów,
nie end-to-end throughput aplikacji. Surowe200 czasów/seria zachowane.

Kolejność dla każdej pary/batch/serii: teacher, potem budżety30/50/70,
U,T gdy (seed−42+seria) parzyste, inaczej T,U. Kolejność batch1,128.
Teacher osobno w każdej serii. Żadnych równoczesnych kopii modeli na GPU.
Modele przygotowane na CPU; przed każdym modelem gc+empty_cache i
memory_allocated==0; identyczna polityka dla teacher/U/T. Po przeniesieniu
modelu/wejścia:50 warmup, synchronizacja, usunięcie output, dopiero wtedy
reset_peak_memory_stats. Baseline allocated=model+bufory+input po warmup;
peak allocated podczas200 forwardów, delta=peak−baseline (przejściowe
aktywacje/workspace/output widziane przez allocator). Reserved baseline/
peak także zapisane. CUDA context, biblioteki i zewnętrzne alokacje nie
wchodzą do allocated. Output usuwany po każdej iteracji, nie utrzymywany
z poprzedniej. Nie czyścimy cache między iteracjami.

Agregacja: per-seed median/p10/p90 z600 zdarzeń; w tabeli głównej średnia
czterech per-seed median, ilorazy T/U najpierw sparowane per seed, potem
średnia. Rozrzut seedów/serii opisowy. Cztery przypisania seed/GPU na jednym
serwerze nie są niezależną replikacją sprzętową ani podstawą wnioskowania
statystycznego o populacji GPU. Temperatury/zegary/power i UUID/PCI zapisane;
zegary nieblokowane, pozostały dryf termiczny jawne ograniczenie.

## Kontrola i artefakty

Audit hook blokuje otwarcie czegokolwiek w data/; helpery danych/oceny,
DataLoader wyłączone w procesie. Źródła E035 pozostają niezmienione.
Freeze zawiera kopię planów, hashe checkpointów/kodu i istniejących metryk,
stan PRE i klastry. Hash przed i po runie, niezmienność stanu po pomiarach.
Bez zapisywania wag/aktywacji/danych: JSON czasów, kosztów, diagnostyki,
manifest/status/logi i CSV analizy. Cel <100MB nowych artefaktów.

Launcher monitoruje konkretne PID4workerów, manifest i obce procesy GPU
co10s; obcy proces/OOM/błąd/limit5400s zatrzymuje całość i zachowuje wyniki.
Limit obejmuje przygotowanie modeli w workerach i pomiary. Brak retry
wybierającego korzystniejsze wyniki. Oddzielne analyze/audit po pomiarach,
WYNIKI/PODSUMOWANIE, STATUS/NOTATKA/README; lokalne commity, bez push.

## Odtworzenie z katalogu projektu

```bash
export CONDA_DEFAULT_ENV=attention
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/users/s224574/miniconda3/envs/attention/bin/python
E=experiments/2026-11-07_E036_audyt_kosztu
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/benchmark.py" freeze
# Commit protokołu/kodu/frozen przed run.
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$E/benchmark.py" run
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/analyze.py" runs/E036/<run>
```
