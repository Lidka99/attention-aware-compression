# 2026-11-22_E046_l06_b10_tiny — jeden zamrożony profil przy b10

**Pytanie:** czy na Tiny dev L06 po projekcji do b10 zmniejszy stratę top‑1
względem pełnego modelu przy obu rzeczywistych pułapach UNIFORM z E045?
Priorytetem jest zachowanie trafności. Ten etap obejmuje wyłącznie bramkę
geometrii na CPU. **Jeden profil kandydacki, jeden budżet; bez oceny trafności.**

## Zamrożenie profilu i bramka CPU

1. Odniesienia z [audytowanego E045](../2026-11-22_E045_lagodny_uniform_tiny/WYNIKI.md),
   `runs/E045/20261122T031350Z_a58b56a3`: pełny model ma **11 279 112 P**
   i **148 148 224 MAC/obraz**; pułapy b10 to **10 111 430 P** i **132 336 196
   MAC/obraz**, nie nominalne 90% parametrów. Używamy dokładnego wektora `U`
   z `frozen.json` tego runu.
2. L06 oznacza zapisane w E043 przesunięcia profilu wybranego na CIFAR-100
   dev w E042: `[+2, −15, −14, +13, −1, −5, −6, +9, −12, −13, +17, −6]`
   w pp udziału zachowanych kanałów, w kolejności grup z `frozen.json` E043.
   Nie przenosimy końcowych szerokości z b50 ani b30. Propozycja jest liczona
   jak w E044: `clip(rint(U + n·offsets_pp/100), LOW, HIGH)`.
3. **Jawna zmiana reguły projekcji** względem E043/E044: `HIGH = n` zamiast
   `floor(0,9n)` — dopuszczamy także grupy bez redukcji kanałów. Dolna granica
   `LOW = floor(n/10)` pozostaje. Stary górny wektor maksymalizuje P i MAC w
   starej domenie (rachunki kosztów są rosnące w szerokościach); na Tiny
   wymusza co najmniej 19,21% redukcji P i nie dosięga kosztu b10 od dołu.
   Samo spełnienie górnych pułapów przez znacznie mniejszy model nie jest
   dopasowaniem b10. Zamrożona kopia solvera E041 z E042, z rachunkiem kosztu
   Tiny jak w E044, wykonuje jedną projekcję SLSQP od `U`: minimalizuje
   kwadrat odległości w `k/n` od propozycji przy `P, MAC = 0,9995 × pułapy`,
   następnie stosuje dotychczasowe zaokrąglenie i naprawę do dwóch współrzędnych.
4. Bramka wymaga całkowitych legalnych szerokości różnych od `U` oraz
   `99,9% ≤ koszt / koszt(U) ≤ 100%` **dla każdego z dwóch kosztów**. Na
   CPU odtwarzamy pełny model, `U/b10` oraz jednego kandydata z każdego
   istniejącego `last.pt` E019, seedy 42–45. Sprawdzamy SHA‑256, mapę grup,
   niepuste klastry, bias FC, niezmieniony model źródłowy i brak gradientów.
   Rachunek musi zgadzać się z liczbą parametrów modelu i MAC Conv/Linear
   z hooków dla syntetycznego zera `1×3×64×64` w trybie `eval`. Używamy
   tego samego operatora CLFM (`694691556e8314a835178a741938dd3d4d80c171`),
   dla L06 z adapterem szerokości per grupa jak w E044. CPU nie otwiera
   obrazów ani etykiet, nie kalibruje BN i nie zapisuje wag.
5. Przy niespełnieniu bramki zapisujemy powód i kończymy: nie zmieniamy
   okna, algorytmu ani puli. Nie wznawiamy E024–E028 i nie rozbudowujemy
   E042. PASS zamraża wektor do późniejszej oceny; sam nie mówi nic o top‑1.

## Dane i późniejszy pomiar

- **Tiny dev**: te same 10 000 indeksów w `cache_train.npz` co E035 dev i
  E045. Jest to używany zbiór rozwojowy. A4096 to ta sama zamrożona lista
  indeksów z części train, rozłączna z dev. Dokładne listy, normalizacja,
  hash cache i oczekiwany hash widoków A pochodzą z `frozen.json` E045.
- **Oficjalny Tiny val** to osobny `cache_val.npz`, użyty w E035 i E044.
  Tiny nie uczestniczył w wyborze L06, ale pomiar E044 nie był na świeżym
  zbiorze projektu. **Rezerwa CIFAR-100** (5000 obrazów z części train,
  odłożonych od dev) została użyta w E043 i jest innym zbiorem niż Tiny.
  E046 nie używa żadnego z tych dwóch zbiorów. Nowy trening lub ocena na
  nieużytych danych testowych wymagają odrębnej zgody.
- Późniejsza faza (po decyzji o uruchomieniu): Conda `attention`,
  `CUDA_VISIBLE_DEVICES=0,1,2,3`, po jednym procesie na kartę:
  GPU 0/1/2/3 → seed 42/43/44/45. `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, FP32, TF32 wyłączone, deterministycznie
  jak w E045. Sprawdzić PID → UUID, wersje środowiska i kody zakończenia;
  przy mniej niż czterech kartach nie uruchamiać. Te same końcowe `last.pt`
  E019. Dla każdego modelu ten sam `e30.augment` (crop 64, skala 0,4–1,0,
  poziomy flip, seed `20261103 + indeks`) i `e30.normalize`; sprawdzić
  hash widoków A. BN: `CMA_reset`, 32 partie po 128, bez zmiany parametrów.
  Dev bez augmentacji, z tą samą normalizacją i batch 128. Przed oceną
  kandydata odtworzyć na wszystkich seedach stany po CMA i trafienia
  pełnego modelu oraz U/b10 z plików `seed_<s>/metrics.json` E045 o hashach
  z manifestu; sprawdzić rzeczywiste P i MAC. Limit:
  cztery oceny kandydata i osiem odtworzeń kontroli. Brak fine-tuningu.
- Raport: per seed liczba trafień i bezwzględny top‑1 (%), sparowana strata
  `top‑1(pełny) − top‑1(profil)` (pp), oraz `strata(U) − strata(L06)` (pp);
  średnia, próbne SD i zakres po czterech seedach. Podajemy rzeczywiste P/MAC,
  redukcje względem pełnego i marginesy względem obu pułapów. Kierunkowy
  wynik pilota na dev: dodatnia średnia poprawa straty i dodatnia poprawa
  na co najmniej 3/4 seedów; wielkość poprawy i pozostałą stratę oceniamy
  oddzielnie. Kontekst E045: pełny **51,2200%**, U/b10 **46,3625%**, strata
  **4,8575 pp** (4,38–5,60 pp po seedach).

[Odtworzenie kontroli CPU](URUCHOMIENIE.md).
