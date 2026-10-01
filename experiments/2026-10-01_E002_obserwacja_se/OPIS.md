# 2026-10-01_E002_obserwacja_se

> Zachowany szkic v0.1. Bieżący protokół wykonania: [PROTOKOL_v1.md](PROTOKOL_v1.md),
> konfiguracja: [config.json](config.json). Poniższa sekcja 8 opisuje stan
> sprzed implementacji, nie aktualne braki.

Status: **draft — pomiar nieuruchomiony**. Wersja: **0.1-draft**.
Data utworzenia: 2026-10-01 UTC. Etap 3: obserwacja, bez zmiany modelu.
E001 zachowuje swój audyt i wykresy. E002 jest nową próbą obserwacyjną,
z innym pytaniem, nie kolejnym audytem treningu.

## 1. Pytanie i cel

Czy kanały podobne przed SE otrzymują różne wagi zależnie od obrazu i czy
wpływa to na podobieństwo ich aktywacji po SE?

Celem jest opis tego zjawiska w **jednym bloku jednego modelu**. Nie
łączymy kanałów, nie mierzymy jeszcze skutków kompresji i nie uczymy modelu.
Hipoteza robocza: wysoki poziom podobieństwa przed SE nie musi oznaczać
podobnego zachowania wag SE ani identycznego podobieństwa po SE.

## 2. Model i wybór bloku przed wynikami

- Zweryfikowany `best.pt` E001, epoka 177, dev 3342/5000.
- Źródło: `runs/E001/20260926T013203Z_7cc3f46_seed42_qHKULg/`.
- SHA-256 checkpointu:
  `f2ac9c3fddd140c0d45f2c244c78b432cbdd164b46dff32d00be956202ec5be8`.
- Kod treningu: `7cc3f46667990ff45686f7499c5dfd00d6d57649`.
- EfficientNet-B0 torchvision, `weights=None`, 100 klas, SyncBatchNorm,
  ścisłe wczytanie checkpointu, eval(), inference_mode(), FP32.
- **Blok `features.4.1`, SE `features.4.1.block.2`.** Wybór na podstawie
  struktury sieci, nie wyników podobieństwa. Blok pośredni, stride 1,
  gałąź residualna 80→80, rozszerzenie 80→480, depthwise 3×3,
  SE 480→20→480, projekcja 480→80.
- Dla wejścia 96×96: h i z mają kształt B×480×6×6, a B×480×1×1.
  Rozmiary wynikają ze struktury; podczas pomiaru trzeba je potwierdzić.

Odczyt przez hooki: pre-hook SE daje **h**, hook `SE.scale_activation`
daje **a**, hook wyjścia SE daje **z**. **z nie oznacza wyjścia fc1**.
Każdy hook musi wykonać się raz na batch. Sprawdzamy z=a*h; niczego
nie zastępujemy ani nie nadpisujemy w forwardzie.

## 3. Dane: 1000 obrazów train, dwa rozłączne podzbiory

Korzystamy wyłącznie z listy **train E001**. Obrazy zostały użyte podczas
treningu modelu; jest to diagnostyka, nie niezależny test generalizacji.
Nie używamy dev, rezerwy ani oficjalnego testu.

- Dokładnie 10 obrazów każdej ze 100 klas, łącznie 1000.
- NumPy `default_rng(20261001)`; klasy 0..99 po kolei. W każdej klasie
  permutujemy posortowane ID należące do train. Pierwszych 5 trafia do
  **A** (wybór par), następnych 5 do **B** (sprawdzenie na innych obrazach).
- Listy A i B sortujemy i zapisujemy przed inferencją, każda po 500 ID.
  Potwierdzamy ich rozłączność i przynależność do train. Zapisujemy klasę
  i ID każdego obrazu oraz hash manifestu wyboru.
- Istniejący `splits.json` SHA-256:
  `ecb8804e41fbc4bee7c61fd0585bf1c7dcfc94ac8472333fc283d51dc6460e86`.
- Plik `cifar-100-python/train` SHA-256:
  `735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b`.
- Preprocessing jak przy ocenie E001: PIL Resize bilinear, antialias,
  96×96, ToTensor, Normalize mean=std=(0.5,0.5,0.5), bez augmentacji.
  Kontener treningowy jest czytany, ale tylko wybrane obrazy trafiają do
  modelu. Loader nie otwiera pliku test.

Po utworzeniu list nie zmieniamy ich pod obserwowany efekt. Podział A/B
sprawdza stabilność na obrazach, nie zastępuje niezależnych treningów.

## 4. Pomiary ustalone przed inferencją

Dla każdego kanału c tworzymy wektor H_c: wszystkie jego liczby z map
6×6 na wszystkich obrazach danego podzbioru, w kolejności ID, potem wiersz
i kolumna. Analogicznie Z_c po SE. Nie centrujemy tych wektorów i nie
normalizujemy osobno każdej mapy obrazu. Dzięki temu zachowujemy różnice
skali między obrazami. Redukcje i obliczenia par: FP64 na CPU, po odczycie
aktywacji FP32. Obliczamy osobno A i B, nie łącząc ich do wyboru par.

Dla każdej pary c<d spośród 480 kanałów (**114960 par**):

1. **Podobieństwo przed SE:**
   `s_h = dot(H_c,H_d)/(norm(H_c)*norm(H_d))`.
2. **Kontrola po SE:** analogiczny `s_z` dla Z; zapis `delta_s=s_z-s_h`.
   Wspólny wektor wielu obrazów pozwala zauważyć różne skalowanie zależne
   od obrazu. Stałe dodatnie przeskalowanie całego kanału nadal nie zmienia
   cosinusa — dlatego obliczamy także następny pomiar.
3. **Różnica uwzględniająca siłę aktywacji:**
   `d_h=norm(H_c-H_d)/(norm(H_c)+norm(H_d))`; analogicznie d_z.
   To nie jest błąd scalenia ani optymalnie dopasowana rekonstrukcja.
   Celowo pokazuje też stałe różnice skali, których cosinus nie widzi.
4. **Różnice wag SE:** dla obrazu x, `b(x)=a_c(x)-a_d(x)`:
   średnia b, RMS b i `std(b, ddof=0)`. Pierwsze dwie opisują całkowitą
   różnicę, a std różnicę zmieniającą się między obrazami.
5. **Kontrola względnych wag:** `l(x)=log(a_c(x))-log(a_d(x))`,
   średnia l i `std(l, ddof=0)`. Stały stosunek wag daje zerową zmienność l,
   nawet jeśli różnica bezwzględna b zmienia się. Do log używamy
   `clip(a,1e-12,1)`, raportując liczbę przycięć; duże przycinanie jest
   ograniczeniem, nie ukrywaną normalizacją.

Dodatkowo na kanał: normy H/Z, liczba dokładnie zerowych map, rozkład wag
SE (średnia, std, min, max). h może zawierać wartości ujemne po SiLU;
nie stosujemy dodatkowego ReLU. Wartości nieskończone lub NaN przerywają
pomiar. Wektory o normie ≤1e-12 oznaczamy jako numerycznie zerowe; cosinus
i porównania wymagające norm pozostają dla nich nieokreślone (nie wpisujemy 0).
Raportujemy liczbę wykluczeń; dokładnie zerowe mapy liczymy niezależnie.
Wyniki cosinusów przycinamy do [-1,1] tylko z powodu błędów zaokrąglenia,
zapisując liczbę takich korekt.

## 5. Wybór par i prezentacja bez wybierania korzystnego wyniku

- Podstawowy wynik: pełna tabela wszystkich par osobno na A i B.
- Do szczegółowej diagnostyki wybieramy **100 par o największym s_h na A**,
  nie po różnicach atencji. Remisy: kolejność (c,d). Pary numerycznie zerowe
  na A są niekwalifikowane; jeśli poprawnych jest mniej niż 100, raportujemy
  wszystkie. Nie nazywamy arbitralnie tych par „wystarczająco podobnymi”.
- Te same pary oceniamy na B. Jeśli stają się nieokreślone, raportujemy to
  i nie zastępujemy innymi. Zapisujemy tabelę indywidualną, mediany, kwartyle
  i zakres s_h, s_z, d_h, d_z, std b i std l.
- Kontrola zakresu: taki sam opis dla wszystkich poprawnych par na B.
- Wykresy: s_h vs s_z; s_h vs std l; d_h vs d_z na B, z zaznaczeniem
  par wybranych na A. Rozkłady std l oraz wag kanałów. Bez wygładzania.
- Zapisujemy także pokrycie kanałów w 100 parach: pary mogą współdzielić
  kanały, nie są niezależnymi replikami. Nie stosujemy testów istotności
  traktujących 114960 par jak niezależne obserwacje.

## 6. Kryteria techniczne i interpretacja

**Technicznie poprawny pomiar:** zgodne hashe i model, dokładnie 1000
unikalnych wybranych ID z poprawnymi klasami, komplet h/a/z, poprawne kształty,
jedno wywołanie hooka/batch, z zgodne z a*h (`atol=1e-7, rtol=1e-6`),
wagi SE w [0,1], brak NaN/Inf, identyczny hash parametrów i buforów modelu
przed/po na wszystkich GPU. Każde odstępstwo zapisujemy i przerywamy;
nie zmniejszamy batcha ani liczby GPU automatycznie.

**Interpretacja naukowa jest opisowa**, nie ustalamy po wyniku progu
„sukcesu metody”. Sprawdzamy, czy pary o wysokim s_h na A zachowują wysokie
s_h na B, jaka jest ich zmienność względnych wag i czy różni się s_z/d_z.
Jeżeli pary nie zachowują podobieństwa, uczciwie raportujemy niestabilność.
Jeśli std l jest bliskie zeru lub podobieństwo po SE nie różni się wyraźnie,
jest to równie ważny wynik. Nie przeszukujemy kolejnych bloków w tym biegu.
Nie traktujemy każdego niezerowego wyniku jako istotnego praktycznie.

Ten eksperyment nie mierzy strat jakości po połączeniu, więc nie może
potwierdzić, że wagi SE pomagają wybierać bezpieczne scalenia. Aktywacje po
SE już zawierają efekt atencji; osobne użycie a musi później wykazać korzyść
ponad to odniesienie. Dalsze próby wymagają nowego protokołu.

## 7. Cztery GPU i reprodukowalność

Conda attention jak E001: Python 3.11.15, torch 2.12.0+cu130,
torchvision 0.27.0+cu130, NumPy 2.4.6. Ścieżki z configu są względem
katalogu głównego repozytorium. Cztery procesy na czterech RTX 2080 Ti.
Jeden checkpoint, rozłączne części posortowanej po ID listy 1000 obrazów:
rank r otrzymuje pozycje r,r+4,... (250 obrazów/rank). Batch 32/GPU,
końcówka 26, bez dopełniania i drop_last. Dwa workery/proces, jeden wątek
CPU, seed 42+rank i loader 42+rank+100000. Seedy nie oznaczają nowych treningów.
FP32, bez AMP/TF32, algorytmy deterministyczne, CUDA_VISIBLE_DEVICES=0,1,2,3,
CUBLAS_WORKSPACE_CONFIG=:4096:8, PYTHONHASHSEED=42, OMP_NUM_THREADS=1.

Każdy proces zapisze ID, etykiety i h/a/z, a agregacja sprawdzi dokładne
pokrycie przed analizą CPU. Surowe aktywacje, około 140 MB dla h+z oraz
około 2 MB dla a bez narzutu, pozostaną poza Git. Cztery karty nie stanowią
czterech niezależnych powtórzeń.

## 8. Co trzeba uzupełnić przed frozen i uruchomieniem

Ten dokument ustala zakres, ale **nie jest jeszcze instrukcją działającego
programu**. Kolejny mały krok to implementacja zbierania h/a/z i kontrole:

1. Kod z opisem wejść/wyjść i linkiem do tego protokołu, konfiguracja,
   przegląd hooków i miar oraz mały test algebraiczny na CPU.
2. Dokładne listy A/B i ich hashe, zapisane przed inferencją; sprawdzenie
   plików danych i manifestów E001 bez nadpisywania.
3. Dokładna komenda launchera na 4 GPU, wersja kodu w commicie, czyste
   drzewo i wersje środowiska. **Komenda: jeszcze nieustalona — brak launchera.**
4. Wersja frozen protokołu jako nowy plik (v1.0), zachowanie v0.1-draft.

Każdy bieg: nowy `runs/E002/<UTC>_<git>_<unikalny>/`, manifest, kopia kodu
i protokołu, konfiguracja, środowisko/eksport Condy, seedy/RNG, dokładna
komenda, hashe wejść, logi, dane ranków, tabele par, status i artifacts.sha256.
Wykresy i raporty tego eksperymentu: `wykresy/<run-ID>/` w tym folderze,
kod/dokumenty/małe wykresy w Git. Nie nadpisujemy wcześniejszych wyników.
Duże artefakty lokalne; brak zewnętrznej kopii zapasowej jest znanym ograniczeniem.

Ograniczenia: jeden wybrany blok i checkpoint, dane treningowe, przeuczenie
E001, brak niezależnej oceny generalizacji i brak próby scalania. Podział
A/B nie eliminuje zależności od wspólnego treningu modelu.

## 9. Przegląd szkicu, 2026-10-01

General-flash sprawdził strukturę torchvision na CPU i zgodność zakresu
z dokumentami E001. To inspekcja biblioteki i kontrola na sztucznych
wejściach, nie pomiar zachowania wytrenowanego checkpointu. W kodzie
odczytu trzeba jawnie rozróżnić `SE.activation` (SiLU wewnątrz SE,
20 kanałów) od `SE.scale_activation` (Sigmoid, wagi a dla 480 kanałów).

Algebraiczna kontrola NumPy, bez danych CIFAR i bez GPU:

- H_c=[1,2,1,2], H_d=2*H_c; a_c=[0.9,0.1], a_d=[0.1,0.9]
  dla dwóch obrazów (waga powtarzana na ich dwóch pozycjach). Cosinus h
  wynosi 1, cosinus z po połączeniu obrazów około 0.219512; na każdym
  obrazie osobno nadal wynosi 1.
- Dla a_c=[0.4,0.1], a_d=2*a_c cosinus z pozostaje 1. Std różnicy wag
  to 0.15, lecz std log-stosunku wynosi około 8e-17, czyli numerycznie 0.
  Dlatego sama zmienność różnicy wag nie dowodzi zmiennego względnego ważenia.

Wysokie podobieństwo i stały stosunek wag **nie gwarantują bezpiecznego
scalenia**: zmiana kanałów zmienia też wejście i działanie modułu SE.
Wybór najlepszych par na A może dać mniej skrajne wyniki na B bez błędu
implementacji. Nie traktujemy tego automatycznie jako efektu atencji.

## Odnośniki

- [Proste wyjaśnienie](WYJASNIENIE.md)
- [Konfiguracja szkicu](config.draft.json)
- [Raport audytu E001](../2026-09-26_E001_efficientnet_b0_od_zera/AUDYT_WYNIKI.md)
- [Aktualny plan](../../AKTUALNY_PLAN.md)
