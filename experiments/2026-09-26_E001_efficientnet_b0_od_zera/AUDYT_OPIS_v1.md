# Audyt E001 — protokół v1

Eksperyment: `2026-09-26_E001_efficientnet_b0_od_zera`.
Audyt i wykresy są częścią tego samego eksperymentu. Nowy numer otrzyma dopiero
nowa próba badawcza. Ten dokument uzupełnia [OPIS.md](OPIS.md), nie zmieniając
ustawień ani danych zakończonego treningu.

Wersja: **1.0**, ustalona przed oceną. Data: 2026-09-29 UTC.
Status protokołu: frozen; status wykonania zapisuje manifest konkretnego uruchomienia.

## Pytanie, cel i kryterium

Czy zapisany `best.pt` E001 daje na tych samych 5000 obrazów dev dokładnie
3342 poprawne odpowiedzi i te same predykcje? Celem jest kontrola artefaktów
i procedury oceny oraz opis przebiegu uczenia, bez zmiany wag.

Sukces wymaga: zgodnych hashy wejść, ścisłego wczytania modelu i zgodności
metadanych checkpointu, dokładnie tych samych 5000 unikalnych ID i etykiet,
**zero różnic predykcji**, **3342/5000**, epoki **177** oraz różnicy dev NLL
nie większej niż **1e-6** względem zapisu tej epoki. Tolerancja dotyczy wyłącznie
redukcji zmiennoprzecinkowej, nie accuracy. Każdą rozbieżność raportujemy;
nie zmieniamy progu po wyniku. NaN/Inf, zmiana wag, brak czterech GPU lub błąd
integralności przerywa audyt. Negatywny wynik jest zachowywany.

## Wejścia

- Bieg: `runs/E001/20260926T013203Z_7cc3f46_seed42_qHKULg/`.
- Kod treningu: `7cc3f46667990ff45686f7499c5dfd00d6d57649`.
- SHA-256 `best.pt`: `f2ac9c3fddd140c0d45f2c244c78b432cbdd164b46dff32d00be956202ec5be8`.
- `artifacts.sha256` sprawdzany w całości; konfiguracja, manifest danych,
  środowisko, metryki i predykcje pochodzą z tego biegu.
- `data/cifar100/splits.json`, `cifar-100-python/train` i `meta`:
  hashe muszą zgadzać się z manifestem E001. Listy splitów sprawdzamy według
  ID i rozłączności. Rezerwa nie jest oceniana ani analizowana obrazowo.
- Oficjalny test nie jest otwierany: własny minimalny loader czyta wyłącznie
  plik treningowy CIFAR-100 (unikamy także automatycznej kontroli pliku test
  przez konstruktor torchvision). Z kontenera treningowego wybieramy tylko dev.
  Czytanie wspólnego kontenera i hash całego pliku nie oznaczają oceny rezerwy.

## Model, dane i cztery GPU

Model torchvision `efficientnet_b0(weights=None, num_classes=100)` z SyncBN,
`strict=True`, `eval()`, `inference_mode()`, FP32, bez autocast i TF32.
Preprocessing PIL: resize bilinear z antialias do 96×96, ToTensor,
Normalize(mean=std=[0.5,0.5,0.5]), dokładnie jak w E001.

Cztery procesy; rank r ocenia pozycje `r, r+4, ...` listy dev, po **1250**
obrazów. Batch 32/GPU, dwa workery, końcówka po 2 obrazy nie jest odrzucana.
Nie używamy dopełniającego DistributedSampler. Zbieramy rekordy ID/etykieta/predykcja
i niezależnie sprawdzamy ich pokrycie i unikalność; redukujemy sumy NLL/poprawnych/liczby.
To jeden model i jedna ocena podzielona na cztery GPU, nie cztery powtórzenia.

Conda `attention`; seed 42+rank, seed generatora loadera 42+rank+100000.
Algorytmy deterministyczne; CUBLAS_WORKSPACE_CONFIG=:4096:8,
PYTHONHASHSEED=42, OMP_NUM_THREADS=1, CUDA_VISIBLE_DEVICES=0,1,2,3.
Stan RNG przed i po ocenie jest zapisany. Hash tensorów przed i po inferencji
musi być taki sam na wszystkich rankach. Wersje bibliotek i sprzętu porównujemy
z E001; zmiany raportujemy. Nie przywracamy optymalizatora ani treningu.

## Wykresy i interpretacja

Wszystkie 200 epok, bez wygładzania: train/dev NLL i accuracy; train accuracy
= train_correct/train_count. Zaznaczamy epokę najlepszego dev top-1.
Train to pomiar online na augmentowanych obrazach w train(), z AMP;
dev to ocena zamrożonego po epoce modelu w eval(), FP32. Nie są to identyczne
warunki pomiaru. Duża utrzymująca się różnica oraz rozbieżne trendy strat
mogą wskazywać na przeuczenie; sam audyt nie ustala jego przyczyny.
Najlepszy dev top-1 nie musi odpowiadać minimum dev NLL.

## Odtworzenie i artefakty

Z katalogu głównego repozytorium, po zapisaniu kodu i protokołu w lokalnym
commicie i przy czystym drzewie:

```bash
conda run --no-capture-output -n attention python experiments/2026-09-26_E001_efficientnet_b0_od_zera/launch_audit.py
```

Launcher tworzy nowy katalog `runs/E001/audits/<UTC>_<git>_seed42_<unikalny>/`,
zapisuje komendę torchrun, commit, kopię kodu/protokołu, hashe, konfigurację,
środowisko, eksport Condy/pip, logi, predykcje i wynik oraz
manifest i sumy kontrolne po zamknięciu logów. Wykresy PNG/PDF i ich opis trafiają
do `wykresy/<run-ID>/` **w tym folderze E001**, ich hashe do manifestu audytu.
Nie nadpisuje katalogów.
Raport i odnośniki do lokalnego biegu zapisujemy również w Git.
Brak zewnętrznego magazynu artefaktów pozostaje ograniczeniem archiwizacji.

Samo odtworzenie wykresów (CPU, wejście to istniejące metryki, nowy katalog wyjścia):

```bash
conda run -n attention python experiments/2026-09-26_E001_efficientnet_b0_od_zera/plot_metrics.py --metrics runs/E001/20260926T013203Z_7cc3f46_seed42_qHKULg/metrics.jsonl --output-dir experiments/2026-09-26_E001_efficientnet_b0_od_zera/wykresy/odtworzenie_01
```

To audyt jednego checkpointu, nie ponowny trening 200 epok, niezależny test
generalizacji ani dowód korzyści atencji lub kompresji. Nie rozpoczynamy etapu 3.

## Przegląd przed uruchomieniem

Statyczny przegląd E001 potwierdził podział dev bez dopełniania i ocenę FP32.
Audyt wzmacnia kontrolę liczby przykładów o kontrolę dokładnych ID i porównanie
predykcji. Train liczy 39 936 obrazów/epokę ze względu na drop_last, co uwzględniamy
w wykresach. Historyczne opisy „trening nieuruchomiony” nie zastępują manifestu
i wyników ukończonego E001; nie zmieniamy jego zamrożonego protokołu.
