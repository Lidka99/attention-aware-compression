# E029 — oficjalny operator: audyt zakończony pozytywnie

Status: **completed, kryteria zgodności jakości i kierunku alokacji spełnione**,
2026-11-02. Protokół utworzony wcześniej w folderze2026-11-01.

**Najważniejszy wniosek:** przy identycznych szerokościach, biasie FC i BN
nasza referencja E012/E014 odtwarza parametry oficjalnego operatora, a obie
nasze implementacje odtwarzają jego trafność. TILT poprawia wynik również
na operatorze autorów i przy ich funkcji `repair_bn`. W tym sprawdzonym
zakresie przewaga alokacji nie pochodzi ze słabszego operatora baseline.

## Co faktycznie odtworzono

- Kod: `osaukh/folding_as_projection`, commit
  **694691556e8314a835178a741938dd3d4d80c171**; klon
  `runs/E029/upstream_6946915`, repo pozostaje czyste/niezmodyfikowane.
- Bezpośrednie użycie `ResNet18_ModelFolding.apply` i `compress_function`
  oraz `utils.tune_utils.repair_bn`; nie kopiowanie ich implementacji.
- Checkpointy **nasze**: E011 CIFAR-10 i E015 CIFAR-100, seedy42–45.
  Oryginalna architektura autorów przyjmuje state_dict strict. Predykcje
  przed kompresją zgodne z torchvision z błędem numerycznym≤1e-4.
- Budżet nominalnie50% parametrów; dokładnie te same szerokości per profil
  dla trzech operatorów. TILT realizowane adapterem zmieniającym wyłącznie
  `keep_ratio` grupy, z zachowaniem oficjalnego scalania/przebudowy.
  **TILT nie jest wariantem opublikowanym przez autorów.**
- Rozdzielone warianty: natywny upstream z losowanym FC biasem oraz ten sam
  operator z odtworzonym biasem checkpointu; LEGACY E012/E014 i LOCAL E024.
- Oceny przed BN oraz dwie procedury: wspólna CMA/reset i oryginalne
  `repair_bn` (retained EMA0.1), obie na identycznym A4096 crop/flip.

**To nie jest reprodukcja pełnego artykułu1:1.** Inne checkpointy,
dev zamiast oficjalnego testu, obecne wersje bibliotek; kalibracja na
wydzielonym A4096, bez RandomRotation15 z oficjalnego CLI. Folding CPU,
ocena/BN GPU. Nie uruchamiano CLI z domyślnym fine-tuningiem ani downloaderów.
Brak opcjonalnego `open_clip` obsłużono stubem importu, którego konstruktor
rzuca wyjątek przy wywołaniu. Żaden kod CLIP nie był używany; CNN nie zmieniono.

## Wykonanie i artefakty

- Zamrożony [OPIS.md](OPIS.md), kontrole CPU w `check.json`.
- Run: **`runs/E029/20261102T005941Z_064da5c2`**.
- 4/4 workery completed, GPU0–3/seedy42–45, oba zbiory na każdym GPU,
  exit codes `[0,0,0,0]`. **56 modeli×3 stany=168 ocen +24 kontrole teacher**.
- Czas187,91s (3,13min). Run około2,9MiB, upstream732KiB. Limity90min
  i0,5GiB spełnione. Conda attention,4×RTX2080Ti,batch128/GPU/przebieg.
- Źródła, commit, hashe checkpointów/danych/kodu/protokołu, PID, pakiety
  i modele GPU zapisane w manifest/source/pip-freeze/logach.
- Bez gradientów, zmian wag przez BN, treningu, nowych danych,
  publicznych checkpointów,96×96, oficjalnego testu i rezerwy.
- Analiza: `analysis_20261102T010349Z/`: `analysis.json`, `metrics.csv`,
  `comparisons.csv`, `controls.csv`, `TABLES.md`. Per seed/zbiór także
  etykiety klastrów i pełne diagnostyki BN.

## 1. Zgodność naszych operatorów z upstream

Porównania do upstream po odtworzeniu biasu FC, przy identycznym profilu:

| Implementacja | Wagi po foldingu | Top1 pre/CMA/UP_REPAIR | Logity |
|---|---|---|---|
| LEGACY E012/E014 | identyczne dla wszystkich16 konfiguracji seed×zbiór×profil | różnica0pp we wszystkich przypadkach | max_abs≤1,91×10⁻⁶ |
| LOCAL E024 | różnice numeryczne; w kilku konfiguracjach także numery/partycje klastrów | różnica0pp we wszystkich przypadkach | max_abs≤4,58×10⁻⁵ |

LEGACY: klastry zgodne (ARI1), parametry max_abs0. Różnice logitów
wynikają z wykonania dwóch implementacji architektury, nie z różnic wag.

LOCAL: po CMA **100% zgodności predykcji** z upstream na wszystkich
badanych przykładach. Przy UP_REPAIR jeden przypadek (CIFAR-100 seed45
UNIFORM) ma99,98% zgodności, przy tej samej trafności. Nie twierdzimy,
że wszystkie implementacje są bitowo lub algebraicznie identyczne.

Jawny wyjątek w reprezentacji: LOCAL/TILT CIFAR-10 seed43 ma w jednej
grupie ARI0,94068; w kilku konfiguracjach identyczne partycje mają inne
numery klastrów. Bez dopasowania permutacji różnica pojedynczych wag
dochodzi do0,2963, mimo logitów różniących się najwyżej o kilka10⁻⁵.
Raportujemy to bez utożsamiania podobnych predykcji z dowodem identyczności
partycji. **Zgodność jakości jest empiryczna na sprawdzonych wejściach.**

Wszystkie **8/8 bramek jakości** (2 zbiory×2 profile×2 implementacje)
przechodzą: średnia i maksymalna bezwzględna różnica top1 po CMA=0pp,
przy progach≤0,5pp średnio i≤1pp na seed.

## 2. Czy TILT pomaga także na operatorze autorów?

Porównanie upstream U/T z zachowanym biasem FC, średnia±SD po4 seedach:

| Zbiór | BN | UNIFORM [%] | TILT [%] | TILT−UNIFORM [pp] | Dodatnie |
|---|---|---:|---:|---:|---:|
| CIFAR-10 | CMA/reset | 89,435±0,630 | 93,430±0,322 | **+3,995±0,408** | **4/4** |
| CIFAR-10 | oryginalne repair_bn | 89,735±0,574 | 93,480±0,404 | **+3,745±0,315** | **4/4** |
| CIFAR-100 | CMA/reset | 53,855±1,509 | 55,550±1,105 | **+1,695±1,115** | **4/4** |
| CIFAR-100 | oryginalne repair_bn | 54,510±1,484 | 55,970±1,059 | **+1,460±1,142** | **4/4** |

Kierunek utrzymuje się w obu procedurach i na obu zbiorach. Wcześniejszy
słabszy CIFAR-100 nadal jest słabszy. Nie zmieniamy progu ani werdyktu E015;
to inne profile/warunki kalibracji i nowy audyt, nie reinterpretacja historii.

## 3. Czy losowanie biasu tłumaczyło przewagę?

Natywny `nn.Linear` w upstream dostaje nowe losowe biasy, bo mapa grup
nie przekazuje biasu FC do odbudowy. Osobno zmierzono efekt przywrócenia
biasu checkpointu: `UP_FIXED_U−UP_RAW_U` [pp], średnie po4 seedach:

| Zbiór | pre-BN | CMA | repair_bn |
|---|---:|---:|---:|
| CIFAR-10 | +0,075 | −0,020 | +0,030 |
| CIFAR-100 | +0,095 | −0,040 | −0,015 |

Efekt jest mały i ma zmienny znak, nie tłumaczy przewagi TILT rzędu
1,46–4,00pp. Nie nazywamy zachowania upstream „naprawionym” bez wskazania
tej zmiany: natywny wynik z losowanym biasem również pozostaje w tabelach.

Natywny UNIFORM+oryginalne repair_bn: **89,705% CIFAR-10 /54,525% CIFAR-100**.

## 4. Co wnosi oficjalna korekta BN?

Przy tych samych4096 augmentowanych obrazach:
- CIFAR-10: repair_bn−CMA=+0,300pp U,+0,050pp T;
- CIFAR-100: +0,655pp U,+0,420pp T.

Zmiana BN nie usuwa efektu alokacji. Jest to porównanie retained EMA
z CMA/reset na tym A; nie dowód ogólnej wyższości EMA. Nie dotyczy EMA
po resecie, która przy32 batchach zawodziła w E025.

## Kryteria i decyzja

| Warunek | Wynik |
|---|---|
| Strict load, zgodność architektur/osi, niepuste klastry, skończone modele | TAK |
| Dokładne parametry per profil/implementacja, TILT pod targetem | TAK |
| Zgodność jakości LEGACY/LOCAL z upstream po CMA | TAK,8/8 |
| Dodatni efekt TILT na upstream,≥3/4 seedów | TAK,4/4 dla każdego zbioru/BN |
| 4GPU/exit0, kompletność i limity | TAK |

**Możemy teraz napisać:** „W kontrolowanym porównaniu na naszych ośmiu
checkpointach CIFAR-10/100 przy około50% redukcji parametrów zweryfikowaliśmy
zgodność adaptacji z operatorem autorów; nierówna alokacja poprawia wynik
również po zastosowaniu ich funkcji korekty BN”.

Nie można jeszcze napisać „odtworzyliśmy cały CLFM1:1” ani „działa na
wszystkich architekturach/budżetach”. Brakuje Tiny/pozostałych budżetów
w audycie upstream i drugiej architektury. Oficjalna baza operatora jest
sprawdzona w ograniczonym zakresie — ważny brak został zmniejszony.

**Następna rekomendacja:** przygotować osobny protokół drugiej architektury
(np. PreActResNet18). Jeżeli wymaga nowych checkpointów, trening dopiero
po wyraźnej zgodzie użytkowniczki. Dalszy audyt Tiny/30/70% może korzystać
z istniejących wag, bez treningu. Nic z tego nie zostało uruchomione w E029.
