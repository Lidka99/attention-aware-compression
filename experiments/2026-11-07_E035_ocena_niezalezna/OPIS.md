# 2026-11-07_E035_ocena_niezalezna — protokół zamrożony

Zatwierdzony przez użytkowniczkę zakres; protokół utworzony 2026-11-07 UTC. Commit tego protokołu
poprzedza otwarcie danych niezależnych i wszelką ocenę E035. Bez nowego
treningu, search, selekcji checkpointów, profili lub BN według testu.

## Zakres i źródła

Pięć par: ResNet18 CIFAR10/CIFAR100/Tiny oraz PreActResNet18 CIFAR10/CIFAR100.
Seedy42–45, redukcje parametrów30/50/70%, U oraz TILT+0.30,
PRE/CMA/reset/original repair_bn (trzy stany: PRE, CMA, UP_REPAIR).
360 ocen i60 kontroli teacher dla niezależnego zbioru; taka sama pełna siatka
dev stanowi wcześniejszą bramkę. Teacher oceniany raz na parę/seed/BN.

Dokładne checkpointy, profile i kalibracja źródłowa:
- ResNet30/70: `runs/E031/20261104T212951Z_6c3d7356`;
- ResNet CIFAR50: `runs/E029/20261102T005941Z_064da5c2`;
- ResNet Tiny50: `runs/E030/20261102T013740Z_609b7737`;
- PreAct wszystkie: `runs/E034/fold_20261106T003625Z_9ec4f2ea`, trening
  `runs/E034/train_20261105T014209Z_21b44f52`, last.pt/epoka200.
ResNet checkpointy E011/E015/E019 jak w powyższych źródłach.
Zapisujemy źródłowe plany i ich SHA256, szerokości i liczby parametrów.
Nie uruchamiamy solvera do doboru nowych profili.

Operator upstream `694691556e8314a835178a741938dd3d4d80c171`, bez modyfikacji.
Bias klasyfikatora zachowany i requires_grad=False. Wszystkie wagi zamrożone;
BN wyłącznie na źródłowym train A4096 i dokładnie tych samych crop/flip.
Tiny normalizacja stałymi train-only E019/E030. Reuse helperów tego repo
dozwolony; brak importów attention_v2/v3. Bez optymalizatora/gradientowych
aktualizacji. Każdy stan BN startuje z osobnej kopii modelu PRE.

## Dane i bramka

Oficjalne testy CIFAR po10000; Tiny oficjalny val10000, oznaczony reserve.
Tiny `cache_val.npz` ma images/labels, indeksy reserve0..9999 dotyczą tego
osobnego cache, NIE cache_train. Rezerwa CIFAR100 pozostaje nieużyta.
Tiny val był otwierany w E005/E007 do kontroli kształtu, lecz nie oceniany;
nie twierdzimy, że był nigdy nieotwarty. Dane niezależne tylko do oceny.
CIFAR10: sprawdzić dostępne archiwum, ewentualnie pobrać oficjalne
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz;
MD5 archiwum c58f30108f718f92721af3b95e74349a, test_batch
40351d587109b95175f43aff81a1287e. Rozpakować tylko test_batch, bez nadpisania
train. Hashować SHA256 danych, indeksów, kodu, checkpointów i upstream.

Przed oceną testu CPU check (geometria, forward, bias, plany, provenance),
następnie pełne odtworzenie dev na4GPU. Bramka: każda top1 względem źródła
różni się najwyżej1e-10 w skali0..1, NLL najwyżej1e-5 absolutnie;
identyczne widoki A, profile, liczby parametrów i niepuste klastry.
Wymagane360+60, cztery exit0. Niespełnienie zatrzymuje test; poprawki
infrastruktury jawnie dokumentowane, zachowujemy failed runy.

## Kryteria zamrożone przed testem

Primary E035: średnia sparowana T−U>0 oraz co najmniej3/4 dodatnie seedy
w KAŻDEJ z15 komórek para×budżet po CMA. Osobno wielkość efektu przy50/70:
średnia>=2pp i>=3/4 (10 komórek). Nie jest to ten sam próg co primary E035.
Historyczny E034 zachowuje próg>=2pp przy50/70 i>0 przy30; jego werdykt
nie jest reinterpretowany. E031 zachowuje swój mieszany wynik.
Nie wybieramy BN według wyników. Secondary: repair_bn, NLL (T−U, niżej
lepiej), absolute top1 i strata top1 względem teacher o tej samej BN.
PRE diagnostycznie. Raportujemy per-seed, średnią i próbne SD (ddof=1)
sparowanych różnic, dev obok testu, każdą porażkę oddzielnie.
Progi opisowe, nie testy istotności; n=4 powtórzenia treningu.

## Wykonanie i audyt

Conda attention `/home/users/s224574/miniconda3/envs/attention/bin/python`,
CONDA_DEFAULT_ENV=attention, CUDA_VISIBLE_DEVICES=0,1,2,3; jeden seed/GPU,
konfiguracje kolejno. Przed startem4 wolne GPU; brak konkurencyjnych zadań.
OMP/MKL/OPENBLAS_NUM_THREADS=2, CUBLAS_WORKSPACE_CONFIG=:4096:8,
deterministic algorithms, TF32 off. Batch oceny i BN128 jak źródła.
Budżet orientacyjny kilkanaście minut; twardy limit90min na pełny run
dev+test. Monitorowane konkretne PID, manifest, statusy i4 exit0 na etap.
Manifest: GPU/wersje/pip freeze/przypisania/UTC/hashes/commity/limity.
Test startuje dopiero po globalnej bramce wszystkich4 workerów dev.
Nie ma automatycznego nowego treningu ani wyboru korzystniejszego runu.
Analiza i oddzielny audyt kompletności, skończoności metryk, hashy,
niezmienności parametrów, biasu i danych. WYNIKI/PODSUMOWANIE i aktualizacja
STATUS/NOTATKA/README, lokalne commity bez push.

Po ocenie test przestaje być niewidzianą walidacją nowych wariantów.
Eksperyment sprawdza transfer przewagi TILT nad U, nie optymalność,
mechanizm preaktywacji ani pełną reprodukcję CLFM1:1.
