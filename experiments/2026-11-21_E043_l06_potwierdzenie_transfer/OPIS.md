# 2026-11-21_E043_l06_potwierdzenie_transfer — zamrożony L06 na rezerwie CIFAR-100

**Protokół v1.0, zamrożony przed oceną jakości.** Teraz wykonujemy wyłącznie
audyt metadanych i kontrolę modelu na CPU. Osobna, wyraźna zgoda użytkownika
jest warunkiem jakiegokolwiek pomiaru jakości na rezerwie. Nowy trening nie
należy do protokołu i wymaga osobnej zgody. [Odtworzenie](URUCHOMIENIE.md).

## Pytania i plan porównania

CIFAR-100, ResNet18 32×32, cztery końcowe `last.pt` E015 (seedy 42–45).
**b50 (główne):** czy ten sam profil i budżet z pilota E042 dają praktyczny
zysk również na niewykorzystanych do wyboru obrazach? **b30 (wtórne, ustalone
z góry):** czy ta sama reguła daje zysk przy ~30% zamiast ~50% **redukcji
parametrów**? Oba porównania na tej samej rezerwie, ale z oddzielnymi werdyktami.
Wykorzystujemy te same checkpointy co E042; nie badamy nowych seedów treningu,
architektur ani zbiorów. E042 wybrał L06 na dev z 24 profili; traktujemy ten
wybór jako dany i nie uruchamiamy selektora ani dalszego search.

## Geometria i bramka techniczna

- Źródło L06: zamrożony `runs/E042/20261121T212257Z_207ae03f/frozen.json`,
  `bank.pairs.resnet_cifar100.proposals[profile=L06]`; zgodność z bankiem CPU
  E042. Przenosimy **`offsets_pp / 100` w przestrzeni `k/n`** w tej samej
  kolejności 12 grup. Propozycja przy każdym budżecie to
  `clip(np.rint(k_U + offsets_pp * n / 100), n//10, (9*n)//10)`.
  `np.rint` zaokrągla do najbliższej liczby całkowitej (remisy do parzystej).
  Nie skalujemy przesunięć przez poziom kompresji i nie używamy jako
  przesunięć różnic między końcowymi szerokościami L06 i U z b50.
- U = historyczny UNIFORM danego budżetu z E035, bez zmian szerokości.
  T-proj = historyczny TILT **tego samego budżetu** z E035, po projekcji
  na P/MAC jego UNIFORM. Pułapy to rzeczywiste P i MAC profilu U, z E035/E037:
  **b50 5 581 542 / 275 335 732; b30 7 846 668 / 386 292 784**.
  P obejmuje wszystkie parametry bez buforów; MAC to Conv/Linear na obraz.
- Dla L06 i T-proj używamy niezmienionej funkcji `project` z kopii
  `geometry_E041.py` zapisanej przez E042: SLSQP od `k_U/n`, minimalizacja
  kwadratu odległości od propozycji w `k/n`, cele `0,9995 × (P_U, MAC_U)`.
  Po zbieżności i sprawdzeniu residuum (`max |residual| ≤ 10⁻⁷`): `rint`,
  ograniczenie zakresu, gdy potrzeba naprawa najwyżej dwóch współrzędnych
  (najmniejsza odległość od rozwiązania ciągłego w `k/n`, remis leksykograficzny).
  U, T-proj i L06 muszą mieć całkowite legalne szerokości i trafić w przedział
  **[99,9%, 100%] obu** pułapów, bez przekroczeń. b50 musi odtworzyć dokładnie
  propozycję L06 oraz końcowe szerokości i koszty wszystkich trzech profili
  E042. W b30 sprawdzamy, że profile są odrębne, i raportujemy RMS; próg
  różnorodności całej puli E042 nie jest regułą ponownej selekcji L06.
- CPU: niezależne sprawdzenie mapy osi grup i rzeczywistej geometrii z
  operatorem upstream CLFM (`694691556e8314a835178a741938dd3d4d80c171`,
  adaptery E029/E042). Liczymy P z parametrów i MAC hookami na syntetycznym
  zerowym wejściu 32×32, sprawdzamy 12 szerokości, niepuste klastry, zachowany
  bias FC, brak gradientów i niezmienność stanu modelu źródłowego. Bez obrazów,
  etykiet, BN na danych ani trafności. P/MAC i szerokości sprawdzamy też przy
  każdej późniejszej konstrukcji. Niezgodność wejść, geometrii lub modelu
  blokuje pomiar; żadnego zastępczego profilu.

## Dane, CMA i ocena po zgodzie

Jednorazowy pomiar na `splits.reserve` (5 000 wierszy **oficjalnego pliku
train**, w kolejności z `data/cifar100/splits.json`). Przed pomiarem wymagany
audyt rozłączności indeksów z `splits.train`, `A4096` i `splits.dev` oraz
zgodności hashy metadanych i checkpointów E015/E042. Oficjalny test CIFAR-100
z E035 był już użyty i jest osobnym plikiem; `dev` służył do wyboru L06.
Żaden z nich nie jest źródłem wyniku E043.

Dla wszystkich modeli, także nieskompresowanego **teachera** poza pułapami,
jedna procedura jak E042: reset buforów BN, CMA (`momentum=None`, tylko BN
w trybie train w czasie przebiegu bez gradientów), 32 partie po 128 z tego
samego **A4096 z train**, w tej samej kolejności i z tymi samymi widokami
crop/flip jak E042. Użyć `augment_uint8` E025 (padding 4 zerami, seed per
indeks `20261101 + i`, flip 0,5) i zachować normalizację E015 z 40k train.
Hash widoków A musi zgadzać się z E042; A bez etykiet. Rezerwa: te same
stałe normalizacji, bez augmentacji; wszystkie modele w `eval()`, żaden
bufor nie jest aktualizowany na rezerwie. Weryfikować, że parametry po
foldingu nie zmieniają się w CMA ani w inferencji. Teacher po CMA jest
mierzony raz na seed i odniesieniem dla obu budżetów.

Conda `attention`, dokładnie cztery GPU, seed **42/43/44/45 → GPU 0/1/2/3**,
jeden proces na GPU, siedem modeli na seed (teacher, potem U/T-proj/L06 w
b50 i b30), **28 ocen**. Batch 128 na GPU (ostatni batch oceny: 8), FP32,
deterministyczne algorytmy, bez TF32/AMP, seed foldingu 42 jak E029/E042.
Modele i bufory po CMA pozostają w RAM/GPU; bez nowych plików wag. Nie
udostępniać częściowych trafności do decyzji; brak pełnych czterech seedów
lub niespełniona bramka techniczna oznacza **brak rozstrzygnięcia**.

## Zamrożony kontrast i decyzja

Główny kontrast = **L06 − UNIFORM po CMA, top‑1, b50**. T-proj pozostaje
kontrolą: raportujemy L06−T-proj i T-proj−U dla obu budżetów, ale nie wybieramy
referencji według wyniku na rezerwie. Dla każdego budżetu osobno: jeżeli
`d_s = trafienia(L06,s) − trafienia(U,s)` na tych samych 5 000 obrazach, to
`Δ_s = 100·d_s/5000` pp. Próg praktyczny jak w E042: **średnia Δ ≥ +1,00 pp
i ≥3 z 4 seedów ściśle dodatnich**. Na liczbach całkowitych: `sum(d_s) >= 200`
i `sum(d_s > 0) >= 3`. Remis nie jest poprawą. Stosujemy ten sam próg do
ustalonego z góry wtórnego kontrastu L06−U w b30.

| b50 | b30 | Werdykt |
|---|---|---|
| Spełnia | Spełnia | Potwierdzenie na rezerwie i transfer do b30 na tym samym zbiorze/modelach. |
| Spełnia | Nie spełnia | Potwierdzenie tylko przy b50; brak wykazanego transferu do b30. |
| Nie spełnia | Spełnia | Sygnał korzyści przy b30; brak potwierdzenia wyniku E042 przy b50. |
| Nie spełnia | Nie spełnia | Brak potwierdzenia praktycznej korzyści w obu warunkach. |

Zapisujemy trafienia/N i top‑1 [%] dla każdego seeda i modelu; dla kontrastów
i **straty względem teachera** (`top1_teacher,CMA − top1_profil,CMA`, w pp)
podajemy cztery sparowane różnice, średnią, odchylenie próbki (`n−1`) i
liczbę dodatnich/remisów/ujemnych. Pokazujemy P, MAC, ich stosunek do U,
oraz rzeczywistą redukcję P względem teachera. Bramka jest progiem praktycznym,
nie testem istotności; cztery modele mają wspólną rezerwę. Wynik b30 nie
może zastąpić głównego b50. Rezerwa po pomiarze staje się użyta dla obu
budżetów. Dopiero potwierdzony L06 stanowi tanią referencję do ewentualnego
badania selektora w odrębnym protokole i na niezależnych danych.
