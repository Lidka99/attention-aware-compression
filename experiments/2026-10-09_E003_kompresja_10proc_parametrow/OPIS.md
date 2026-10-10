# E003 — rzeczywista kompresja całego EfficientNet-B0 o 10% parametrów

> Zachowany szkic v0.1. Działający protokół wykonania:
> [PROTOKOL_v1.md](PROTOKOL_v1.md), [config.json](config.json).
> Stwierdzenia o brakującym kodzie poniżej opisują historyczny stan szkicu.

Data: 2026-10-09 UTC. Wersja protokołu: **0.1-draft**.
Eksperyment: `2026-10-09_E003_kompresja_10proc_parametrow`.
Status: **projekt rzeczywistego eksperymentu, kompresja jeszcze nieuruchomiona**.
Status wykonania zapisze manifest biegu. Kod, wejścia i kontrole muszą
być zapisane w czystym commicie przed obliczeniem wyników kompresji.
Konfiguracja szkicu: [config.draft.json](config.draft.json).

## Pytanie badawcze i warunki

**Czy przy identycznej fizycznej strukturze modelu mniejszego o co najmniej
10% parametrów wybór par po SE zachowuje jakość lepiej niż wybór przed SE,
przy wspólnym operatorze i bez dostrajania?** Wybór losowy jest kontrolą.

Cztery warunki: BASE, PRE, POST, RANDOM. BASE to checkpoint E001 bez zmian.
PRE/POST/RANDOM mają identyczne szerokości, liczbę usuniętych kanałów,
dostęp do danych i procedurę kompensacji. Nie opracowujemy nowego
kryterium „attention-aware”; osobna informacja o wagach a wymaga później
wykazania korzyści ponad aktywacje z po SE. E003 mierzy rzeczywisty
skutek kompresji, nie korelacje samych map.

Główny wynik: różnica poprawnych odpowiedzi POST−PRE na tych samych
5000 obrazów dev (licznik i punkty procentowe). Dodatkowo spadek top-1
i zmiana NLL względem BASE, pary rozbieżnych odpowiedzi, rozmiar modeli,
liczba parametrów oraz czas forwardu. Wynik zerowy lub gorszy jest
pełnoprawnym wynikiem. Nie deklarujemy przewagi metody na podstawie
jednego checkpointu. Nie ustalamy po wyniku progu sukcesu praktycznego.

## Model i dokładny budżet, niezależny od jakości

best.pt E001, epoka 177, wytrenowany od losowych wag:
`runs/E001/20260926T013203Z_7cc3f46_seed42_qHKULg/best.pt`.
SHA-256 `f2ac9c3fddd140c0d45f2c244c78b432cbdd164b46dff32d00be956202ec5be8`.
Źródłowy commit treningu: `7cc3f46667990ff45686f7499c5dfd00d6d57649`.

BASE: **4 135 648 parametrów**. Planowana liczba każdego modelu skompresowanego:
**3 721 424 parametrów**, o **414 224 mniej = 10,0159% całej sieci**.
Usuwamy łącznie 1016 kanałów wewnętrznych w 15 blokach. Zewnętrzne
wymiary bloków, residuale, stem, ostatnia projekcja i klasyfikator bez zmian.
Pierwszy MBConv bez ekspansji pomijamy.

| Blok | Szerokość bazowa | Nowa | Usunięte |
|---|---:|---:|---:|
| features.2.0 | 96 | 88 | 8 |
| features.2.1 | 144 | 128 | 16 |
| features.3.0 | 144 | 128 | 16 |
| features.3.1 | 240 | 216 | 24 |
| features.4.0 | 240 | 216 | 24 |
| features.4.1 | 480 | 424 | 56 |
| features.4.2 | 480 | 424 | 56 |
| features.5.0 | 480 | 424 | 56 |
| features.5.1 | 672 | 600 | 72 |
| features.5.2 | 672 | 600 | 72 |
| features.6.0 | 672 | 600 | 72 |
| features.6.1 | 1152 | 1016 | 136 |
| features.6.2 | 1152 | 1016 | 136 |
| features.6.3 | 1152 | 1016 | 136 |
| features.7.0 | 1152 | 1016 | 136 |

Pełny utrwalony plan: [PLAN_STRUKTURY.json](PLAN_STRUKTURY.json), SHA-256
`fdfa35242201b9544afa47838a83f1683aa35482f9164555910b90ed225cab74`.
Kod jego wyznaczenia: [plan_structure.py](plan_structure.py), niezależnie
sprawdzony statyczny budżet. Nie wykonano kompresji wytrenowanych wag.

Plan wynika wyłącznie ze struktury: od zera dodajemy po 8 usuniętych
kanałów w bloku o najmniejszym dokładnym (r+8)/C; remisy według ścieżki.
Maksimum połowa C. Kończymy po osiągnięciu ceil(P_BASE/10). Koszt usunięcia
kanału to in+out+k²+2*squeeze+5. Wszystkie pozostałe szerokości są
wielokrotnościami 8. Liczby trzeba potwierdzić na rzeczywistych grafach
po operacji, a nie tylko odjąć na papierze. Test jednego kanału służy
wyłącznie kontroli kodu, nie jest warunkiem badawczym.

## Dane i kalibracja

Te same listy A/B co E002: A=500 i B=500 obrazów train, po 5/klasę
w każdej połowie, rozłączne. SAMPLES.json E002 SHA-256:
`338eb31862ab3d35495c22c8ae58109aed280a8218f1c32e234c941c3d1bae1d`.
A jest jedynym zbiorem wyznaczania par i współczynników. B służy
diagnostyce błędu bloków; dev to dokładnie 5000 ID E001. Ani rezerwa,
ani oficjalny test nie są oceniane i nie trafiają do modelu.
Dev wcześniej wybierał checkpoint E001, więc nie jest niezależnym testem
generalizacji. A/B także były wcześniej oglądane i należą do train.

Preprocessing identyczny z E001: PIL Resize bilinear antialias do 96×96,
ToTensor, Normalize mean=std=(0.5,0.5,0.5), bez augmentacji. Środowisko
Conda attention: Python 3.11.15, torch 2.12.0+cu130, torchvision 0.27.0+cu130,
NumPy 2.4.6. FP32 forward, bez AMP/TF32, algorytmy deterministyczne.

Na A zbieramy dla każdego bloku sumy iloczynów h, z i g=avgpool(h),
gdzie g to rzeczywiste wejście fc1 SE. Miary/sumy FP64 na CPU; zapis
statystyk wystarczających zamiast wielkich zbiorów wszystkich map.
Dokładnie: S_h=sum(H_batch H_batch^T), S_z analogicznie, S_g=sum(G_batch^T G_batch),
gdzie mapy są spłaszczone po obrazach i pozycjach, G_batch ma wymiar
batch×C. Sumujemy FP64 po batchach w kolejności ID danego ranku;
agregacja ranków w stałej kolejności 0,1,2,3. Wznowienie nie zmienia
kolejności ani list. Zapisujemy statystyki każdego ranku i sumy końcowe.

## Wybór par i wspólny operator

PRE: cosinus pełnych h po wszystkich obrazach/pozycjach; POST: analogicznie z.
Rangi malejąco, remisy według indeksów. Greedy wybiera wymaganą liczbę
**rozłącznych par w każdym bloku**. RANDOM: permutacja tych samych
kwalifikowanych kanałów, seed 20261009+1000*index_bloku (index od 0,
w kolejności PLAN_STRUKTURY.json),
kolejne pary. Reprezentant to mniejszy indeks, drugi kanał usuwamy.
Kanały z normą h/z/g <=1e-12 niekwalifikowane w żadnej metodzie;
jeśli brakuje par, stop failed bez automatycznej zmiany budżetu.

Dla każdej wybranej pary (c,d), na A, bez interceptu i bez clippingu:
alpha=<g_c,g_d>/<g_c,g_c>, beta=<z_c,z_d>/<z_c,z_c>.
alpha dopasowuje wejście SE, beta pełną mapę czytaną przez projekcję.
Ten sam przepis i dwa skalary/parę dla wszystkich trzech metod.
Skalary są wchłaniane do macierzy wag podczas operacji; nie pozostają
oddzielnymi parametrami ani dodatkowymi operacjami w forwardzie.

Fizyczne folding reprezentantowe:
- usuwamy producenta d: wiersz expansion, oba BN kanału d i depthwise grupę;
- SE fc1: kolumna c += alpha*kolumna d, potem usunięcie kolumn d;
- SE fc2: usunięcie wierszy/biasów d; hidden squeeze bez zmian;
- projection: kolumna c += beta*kolumna d, potem usunięcie kolumn d;
- pozostałe parametry/bufory skopiowane, projekcyjny BN i residual bez zmian.

Każdy blok zachowuje oryginalny hidden squeeze, eps/momentum BN i wszystkie
inne hiperparametry poza szerokością oraz liczbą grup depthwise. Nie wolno
regenerować squeeze jako C_new//4. Numery kanałów w manifestach są
oryginalne; zapisujemy też pełną listę zachowanych indeksów i mapowanie.

Warunkowa kontrola algebry: jeśli g_d=alpha*g_c, wejście nieliniowej
części SE jest zachowane, więc zachowane bramki są identyczne. Jeśli
jednocześnie z_d=beta*z_c punktowo, projekcja i wyjście bloku są zachowane.
W praktyce proporcjonalność jest przybliżona, a to nie dowodzi dokładności
operatora na obrazach spoza A. Test proporcjonalnego h wykonujemy na
podsystemie SE/projekcji; SiLU nie pozwala uzyskać go przez samo skalowanie
wag producenta. Test pełnego MBConv używa duplikowanych ścieżek i równych
wierszy/biasów fc2 (alpha=beta=1).

Pary są rozłączne; wszystkie bloki zmieniamy jednocześnie, ze statystyk
nauczyciela. Nie dopasowujemy późniejszych bloków do już zmienionego
modelu. To jawne przybliżenie; błąd może się kumulować. SE po operacji
działa normalnie i może zmienić **wszystkie** swoje bramki. Oceniany jest
rzeczywisty zmieniony forward, nie zamrożone atencje nauczyciela.
Nie uśredniamy konwolucji nieliniowych ścieżek.

Budżet naprawy: **zero kroków optymalizacji, zero rekalibracji BN**.
Analityczna kompensacja alpha/beta jest wspólną częścią operatora, z tym
samym dostępem do A. Oryginalny checkpoint pozostaje niezmieniony.
Manifest grup, reprezentantów i współczynników utrwalamy **przed oceną B/dev**.

## Ocena i cztery GPU

Cztery procesy, po jednej GPU, jeden model bazowy i te same trzy warunki.
Nie są to cztery powtórzenia treningu. A/B: rank r dostaje positions[r::4]
(125 obrazów/rank), dev 1250/rank. Batch 32, końcówki bez odrzucania
i dopełniania. Dokładne ID/etykiety i unikalność kontrolowane po agregacji.
Modele oceniane kolejno na każdej karcie, bez konkurencyjnych zadań GPU.

Na B: błąd gałęzi przed dodaniem residual oraz wyjścia każdego bloku
na tym samym wejściu nauczyciela (suma kwadratów błędu / suma kwadratów
celu). Dla zerowego celu (norma <=1e-12) raportujemy wartość nieokreśloną.
Hook gałęzi musi skopiować tensor przed dodaniem residual: MBConv używa
inplace `result += input`, więc sama referencja dawałaby błędny pomiar.
Dev: top-1, NLL, predykcje/logity z ID dla BASE/PRE/POST/RANDOM.
BASE musi odtworzyć 3342/5000 i predykcje E001, NLL atol=1e-6.

POST−PRE: sparowana różnica accuracy, liczby rozbieżnych trafień oraz
95% przedział bootstrap po obrazach, stratyfikowany po klasach (2000
powtórzeń, seed 20261009). Resampling bootstrap nie jest ponowną ewaluacją;
we właściwym forwardzie każdy obraz liczony raz. To zmienność obrazów przy jednym modelu,
nie zmienność między treningami ani niezależna generalizacja.

Zapisujemy liczbę parametrów każdego modelu i rozmiar porównywalnych
checkpointów inference-only (bez optymalizatora). Pomiar forwardu na
każdej GPU: batch 32 syntetycznych tensorów 3×96×96, 10 warmupów i
50 pomiarów CUDA events, eval/inference_mode/FP32. Raport per GPU,
mediany/kwartyle, bez utożsamiania oszczędności parametrów z przyspieszeniem.
Kolejność BASE, PRE, POST, RANDOM na każdym ranku. Zapisujemy sterownik,
GPU, batch, precision i pomiary każdej karty; nie są to niezależne seedy modelu.

## Kontrole i archiwizacja

Przed uruchomieniem kontrole CPU: kształty i pełna liczba parametrów,
niezmienność BASE, rozłączne pary, poprawność mapowania indeksów,
identyczne/proporcjonalne sztuczne kanały (float64 atol=rtol=1e-10).
Zmniejszenie warstw rzeczywiste, nie maskowanie kanałów.
Checkpoint skompresowany musi mieć metadane szerokości i dawać zgodny
wynik po odtworzeniu nowego grafu oraz ścisłym wczytaniu tensorów.
Nie stosujemy foldingu drugi raz podczas odczytu już skompresowanych wag.
Operator odmawia działania, jeśli wejście nie ma bazowej liczby parametrów
i wszystkich bazowych szerokości. Artefakt zawiera flagę compression_applied,
wariant, plan szerokości i hash grup. Loader materializuje same kształty,
potem wczytuje state_dict strict=True; nie przelicza ponownie alpha/beta.

Seedy procesów 42+rank, loaderów 42+rank+100000, jeden wątek obliczeniowy
CPU na proces oraz **dwa procesy workerów DataLoadera/GPU** (to różne
ustawienia). Workery otrzymują seed z torch.initial_seed() dla NumPy/Pythona,
loader zachowuje kolejność danych (in_order=True),
CUDA_VISIBLE_DEVICES=0,1,2,3, CUBLAS_WORKSPACE_CONFIG=:4096:8,
PYTHONHASHSEED=42, OMP/OPENBLAS/MKL_NUM_THREADS=1. Hash BASE i wszystkich
kopii przed/po oceną musi pozostać taki sam i zgadzać się między rankami.
NaN/Inf, brak 4 GPU, błąd danych/kształtów/parametrów lub baseline
przerywa bieg; wynik niekorzystny nie jest błędem technicznym.

Nowy `runs/E003/<UTC>_<git>_<unikalny>/`: manifest, źródła, konfiguracja,
hashe danych/modelu/list, środowisko/Conda/pip, dokładna komenda, RNG,
logi, statystyki A, candidate_manifest, predykcje, pomiary czasu,
checkpointy inference-only, wynik i artifacts.sha256. Wykresy i małe
raporty w tym folderze eksperymentu, bez nadpisywania innych biegów.
Duże artefakty poza Git, zewnętrzna kopia nadal nie jest skonfigurowana.

Do zamrożenia brakuje jeszcze implementacji operatora/launchera, dowodu
kontroli i instrukcji odtworzenia działającego programu. Wersja frozen
powstanie osobno; zachowamy ten szkic. Komenda wykonania: **jeszcze nieustalona**.
Komendę launchera i hashe kodu dopiszemy przed zamrożonym uruchomieniem;
samo istnienie tego opisu nie oznacza wykonania. Nie deklarujemy nowości
algorytmicznej, szerokiej przewagi ani odtworzenia treningu E001.

## Przegląd przygotowania

2026-10-09: general-flash niezależnie odtworzył statyczny budżet i sprawdził
kształty MBConv oraz wzory foldingu. Dwa workery loadera i jeden wątek
obliczeniowy CPU nie są sprzeczne; nie zastępują też wymagania 4 GPU.
Przegląd i plan architektury nie są dowodem wykonania kompresji lub
pomiaru jakości. Implementacja i kontrole pozostają warunkiem uruchomienia.
