# 2026-11-01_E029_oficjalny_operator

Status: **frozen**, 2026-11-01, przed oceną nowych modeli na dev.

## Pytanie i zakres
Czy nasz operator foldingu i przewaga alokacji TILT są porównywalne z kodem
autorów przy identycznych wejściach, szerokościach, biasie FC i korekcie BN?
To **audyt operatora na naszych checkpointach**, nie odtworzenie pełnej
tabeli artykułu ani jego treningu. Dwa zbiory:CIFAR-10/100, seedy42–45,
jeden budżet nominalnie50% parametrów. Tiny/inna architektura poza zakresem.

Źródło CLFM: https://github.com/osaukh/folding_as_projection,
commit **694691556e8314a835178a741938dd3d4d80c171**, pobrany jako sparse clone
do `runs/E029/upstream_6946915`. Pliki upstream pozostają niezmienione,
hashe i git status sprawdzane. Import ResNet z `compression/fold.py` i
`models/resnet.py`, bez uruchamiania CLI (domyślnie pobiera test i trenuje).
Brak opcjonalnego open_clip obsługiwany stubem wyłącznie jego importu:
wywołanie konstruktora CLIP ma rzucać wyjątek. Kod CNN nie jest zastępowany.
HKMeans pozostaje n_jobs=-1,n_init=1 (ta ścieżka biblioteki jest sekwencyjna).

## Wspólne wejścia
Checkpointy E011/E015, source state_dict ładowany strict do oryginalnej
architektury CIFAR ResNet18 autorów oraz torchvision; bez trenowania.
Kontrola: zgodność parametrów i predykcji przed foldingiem.
Indeksy A4096/dev5000 z E028 (CIFAR-10) i E024 (CIFAR-100); tylko A służy
BN. Crop32/padding4+flip identyczne z E025/E028, bez rotation15 występującego
w oficjalnym CLI. Ta różnica jest jawna i wspólna dla porównywanych ramion.
Oficjalny test, rezerwa, dane do dawnego search nie są używane do BN.

Geometria: solver bez etykiet z samodzielnego E024, UNIFORM pod50%
parametrów, TILT+.30 do **faktycznego** targetu UNIFORM. Szerokości zapisać
przed oceną. To kontrolowane bieżące profile, nie ponowne liczenie dawnych
tabel E014/E015. Każdy operator dla danego profilu ma identyczną liczbę
parametrów. TILT≤target i różnica≤0,1%. Nie interpretować ratio kanałów
jako procentu globalnych parametrów.

## Ramiona — 7 na checkpoint
1. **UP_RAW_U:** niezmieniony `ResNet18_ModelFolding.apply`, uniform;
   losowany bias FC, seed42, zgodnie z kodem autorów.
2. **UP_FIXED_U:** kopia UP_RAW_U z odtworzonym biasem FC checkpointu.
3. **LEGACY_U:** implementacja referencyjna E012/E014, preserve_fc_bias=True,
   dokładnie te same szerokości U.
4. **LOCAL_U:** samodzielny operator E024, te same szerokości U.
5. **UP_FIXED_T:** rozszerzenie upstream WYŁĄCZNIE o keep_ratio zmieniane
   przed `compress_function` dla każdej grupy; bias FC odtworzony.
   To nasza alokacja na operatorze autorów, nie oficjalny wariant TILT.
6. **LEGACY_T:** referencyjny operator E012/E014 przy szerokościach T.
7. **LOCAL_T:** samodzielny E024 przy szerokościach T.

Dla każdego: **pre-BN**, a ze świeżych kopii po foldingu:
- **CMA:** procedura E025 A/aug/reset+CMA (nasza obecna korekta);
- **UP_REPAIR:** niezmienione upstream `repair_bn` na tym samym A/aug,
  zachowuje statystyki, momentum0.1 (EMA), bez gradientów i optymalizatora.
Teacher ma kontrolę pre/CMA/UP_REPAIR. Parametry muszą być niezmienione.
8checkpointów×7ramion=56 modeli,3 stany/model=168 ocen, dodatkowo24 ocen
kontrolnych teacher. Oceny kontrolowane dla zgodności operatorów/logitów,
nie repliki poprzednich prób optymalizacji. Brak nowego search.

## Kontrole i kryteria
Przed runem CPU: strict load, logity dwóch architektur na tych samych
syntetycznych wejściach max_abs≤1e-4; zgodność osi grup i liczby parametrów,
no-compression jako bypass (oficjalny CLI też omija folding ratio0).
Runtime: skończone parametry, niepuste klastry, dokładne budżety, fc bias,
hashe source/protokołu/danych, brak gradientów i zmian wag przez BN.

Primary dla każdego zbioru: LEGACY i LOCAL kontra UP_FIXED, osobno U/T,
po wspólnej CMA. Zgodność jakości: średni bezwzględny rozdźwięk top1≤0,5pp
i maksymalny≤1pp po4 seedach. Nie utożsamiać tego z równoważnością modeli.
Raportować również max_abs/RMSE logitów, zgodność predykcji, różnice
parametrów/buforów i klastry (zgodność partycji, nie tylko numerów etykiet).
Raportować wszystkie zbiory/ramiona, bez wyboru korzystnej BN.
Alokacja: UP_FIXED_T−UP_FIXED_U średnia>0 i≥3/4 dodatnich seedów osobno
dla CMA i UP_REPAIR; wsparcie kierunku, nie dowód optymalności.
Ablacja biasu: UP_FIXED_U−UP_RAW_U we wszystkich stanach.
Różnica UP_REPAIR−CMA oddziela procedurę BN od operatora.
Nie zmieniać progów po wynikach. Jeśli operator nie przejdzie kontroli,
udokumentować przyczynę zamiast nazywać poprawioną wersję „oryginalną”.

## Wykonanie i ograniczenia
4GPU0–3: seed42+i/GPUi, oba zbiory na każdym workerze sekwencyjnie.
Folding CPU (jawny wariant urządzenia oficjalnego kodu); ocena/BN GPU.
Wspólny seed42 dla kompresji, batch128/GPU/przebieg. Conda attention,
limit90min, nowe wyniki≤0,5GiB. Brak checkpointów wyjściowych, pobierania
danych/wag,96×96, testu, rezerwy, treningu i konkurencyjnych zadań GPU.
Biblioteki z bieżącego środowiska, nie oryginalny torch2.2.2: raportować.

Kontrola dotyczy dwóch zbiorów/jednego budżetu i naszych wag, a nie
pełnego pipeline autorów. Dev wielokrotnie używany. Oficjalny operator
z losowaniem biasu i adapter TILT mają osobne etykiety. Tylko dodatni
wynik audytu pozwala mówić o porównaniu na operatorze upstream w tym zakresie.

## Odtworzenie
Conda attention, z katalogu projektu:
```
CUDA_VISIBLE_DEVICES='' python -B experiments/2026-11-01_E029_oficjalny_operator/check.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B experiments/2026-11-01_E029_oficjalny_operator/run.py
CUDA_VISIBLE_DEVICES='' python -B experiments/2026-11-01_E029_oficjalny_operator/analyze.py --run-dir runs/E029/<run-ID>
```
