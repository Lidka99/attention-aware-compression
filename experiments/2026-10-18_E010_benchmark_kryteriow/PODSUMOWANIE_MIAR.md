# E010 — jakie miary liczyliśmy i po co

Dokument pomocniczy do eksperymentu `2026-10-18_E010_benchmark_kryteriow`.
Stan: po zakończeniu biegu `runs/E010/20261018T235741Z_007be2d_seed42_824m539e`
(19.10.2026, status `completed`). Szczegóły protokołu:
[PROTOKOL_v1.md](PROTOKOL_v1.md); wyniki liczbowe: `summary.json` w katalogu biegu.

## 1. Po co ten eksperyment

Chcieliśmy odpowiedzieć na jedno pytanie:

> Czy któreś ze standardowych kryteriów wyboru kanałów do scalenia daje
> po **krótkiej naprawie** (1–20 epok) istotnie lepszą jakość niż losowy
> wybór — i czy efekt utrzymuje się na drugim datasecie?

Cel praktyczny: **wskazać kandydata do metody**, zanim zainwestujemy w E011
(3 seedy, wierne baseline'y z literatury, porównanie head-to-head). E010
jest celowo tanim przesiewem: jeden seed, dwa datasety, wspólny operator.
Nie jest dowodem publikacyjnym i nie miał nim być.

Dlaczego teraz: E008/E009 pokazały, że po dostatecznie długiej naprawie
różnice między PRE/POST/RANDOM znikają; literatura zwykle porównuje
kryteria zero-shot albo po „wystarczającym" fine-tuningu. Chcieliśmy
zobaczyć, co dzieje się w **krótkim oknie naprawy**, gdzie wybór kanałów
jeszcze może mieć znaczenie.

## 2. Co dokładnie porównujemy

- **Operator:** fizyczny folding par kanałów (E008/E003). Kanał `d` jest
  usuwany, a jego wkład wchłaniany przez reprezentanta `c`: α do kolumny
  `fc1` bloku SE, β do kolumny projekcji. W E010 dopuszczono też, by jeden
  reprezentant wchłonął wiele kanałów (potrzebne dla k-means).
- **α/β:** dopasowanie least-squares z momentów Grama:
  `α = S_g[c,d]/S_g[c,c]`, `β = S_z[c,d]/S_z[c,c]`.
- **Datasety i ratio:**
  - Tiny ImageNet 224, 30% redukcji — **primary** (9 miar × T={1,5,10,20}),
  - Tiny ImageNet 224, 10% — walidacja (6 miar × T=5),
  - CIFAR-100 96, 10% — **secondary** (9 miar × T={1,5,10,20}).
- **Budżet naprawy T:** osobny przebieg dla każdego T; start z tego samego
  modelu po scaleniu; AdamW 3e-4, cosine na T epok, batch 32/GPU, 4 GPU,
  seed 42. T=0 to ocena zero-shot (bez treningu).
- **Wspólne dla wszystkich miar:** te same dane treningowe, ten sam seed,
  ta sama augmentacja i optimizer. Różni się tylko sposób wyboru kanałów
  (i budżet T).

## 3. Dziewięć miar — co mierzy każda

| Miara | Co mierzy | Skąd dane | Jak wybiera kanały | Rola |
|---|---|---|---|---|
| **PRE** | podobieństwo aktywacji **przed SE** | kalibracja A (Tiny 1000, CIFAR 500) | zachłannie rozłączne pary o największym cosine; reprezentant = niższy indeks | baseline „podobieństwo przed atencją" |
| **POST** | podobieństwo aktywacji **po SE** | kalibracja A | jak PRE, ale na mapach po SE | baseline „podobieństwo po atencją" |
| **RANDOM** | brak sygnału — kontrola | losowanie | losowe rozłączne pary | punkt odniesienia: czy kryterium bije losowanie |
| **L1** | ważność kanału = norma L1 filtra ekspansji | wagi modelu | usuwa najniższe L1; przypisuje do najpodobniejszego zachowanego po cosine filtrów po BN | prosty baseline „magnitude" |
| **BN_GAMMA** | ważność = \|γ\| BN po ekspansji | wagi modelu | usuwa najniższe \|γ\|; przypisanie jak L1 | tani baseline „skala BN" |
| **TAYLOR** | wrażliwość loss = \|∂L/∂a · a\| na aktywacji ekspansji | gradient na kalibracji A | usuwa najniższą wrażliwość; przypisanie jak L1 | kryterium funkcjonalne (pierwszy rząd) |
| **APOZ** | „martwość" kanału = częstość zer aktywacji | forward na kalibracji A | usuwa najwyższy odsetek zer (najniższe 1−APoZ); przypisanie jak L1 | tani baseline „aktywność" |
| **REC** | błąd rekonstrukcji po scaleniu | momenty Grama (h, z, g) | score pary minimalizuje względną resztę LS; zachłannie rozłączne pary | kryterium rekonstrukcyjne (bez nowego przebiegu) |
| **KMEANS** | grupowanie podobnych filtrów | wagi modelu po BN | k-means++ na wektorach filtrów, `k` = docelowa liczba kanałów; medoid = reprezentant; pozostali scalani do niego | baseline „grupowanie" (nie średnia klastra z literatury) |

### Wspólne reguły wyboru

- Liczba usuwanych kanałów w bloku = `remove_count` z planu struktury
  (30% Tiny: 3168 kanałów; 10% Tiny: 1048; 10% CIFAR: 1016).
- Kanał „niekwalifikujący się" (którąkolwiek norma Grama ≤ 1e-12) zostaje
  w modelu i nie jest reprezentantem.
- Miary per-kanał (L1, BN-γ, Taylor, APoZ): najpierw wybierany jest zbiór
  usuwanych, potem każdy usuwany dostaje **unikalnego** reprezentanta
  z zachłannego dopasowania po cosine filtrów ekspansji po BN.
- Pary są rozłączne po stronie usuwanych; reprezentant może się powtarzać
  tylko w KMEANS (wchłania cały klaster).
- Wszystkie modele mają dokładnie założony budżet parametrów:
  2 981 828 (Tiny 30%), 3 837 364 (Tiny 10%), 3 721 424 (CIFAR 10%).

## 4. Co mierzymy i jak oceniamy

- **Dev top-1** (główna metryka) i **NLL** na zbiorze dev:
  Tiny 10 000 obrazów (50/klasę), CIFAR 5 000 (50/klasę).
- **Zero-shot (T=0):** jakość zaraz po scaleniu, bez naprawy.
- **Krzywe naprawy:** dev top-1 po T=1, 5, 10, 20 epokach.
- **Porównania sparowane:** te same obrazy, bootstrap 10 000 powtórzeń,
  seed 20261018, stratyfikacja po klasach; raportowane jako różnica
  w punktach procentowych z 95% CI.
- **Kontrola wielokrotności:** Holm w rodzinie 6 nowych miar przy
  primary (Tiny 30%, T=5).
- **Bramki techniczne:** odtworzenie modeli E008/E003 bit w bit, pokrycie
  dev, parametry, α/β, determinizm grup; modele reużyte i BASE musiały
  odtworzyć top-1 dokładnie i NLL z tolerancją 1e-5.

## 5. Kryterium decyzyjne (ustalone przed biegiem)

- **Primary:** Tiny 30%, T=5. Nowa miara X „przechodzi", jeśli
  `top1(X) − top1(RANDOM) ≥ 1,0 pp` **oraz** jednostronne p z bootstrapu
  po korekcie Holma < 0,05.
- **Replikacja:** wynik dodatni musi powtórzyć się na CIFAR-100 10%, T=5
  (dolna granica 95% CI > 0 i próg ≥ 1,0 pp). Tiny 10% tylko dla miar
  obecnych w tej siatce.
- Pozostałe T i pary są **eksploracyjne** — bez etykiety „survivor".

## 6. Czego ten eksperyment nie mierzy

- **Jednego seeda** — brak oszacowania wariancji międzyseedowej. To
  screening; „survivor" wymaga powtórki w E011.
- **Wiernych metod z artykułów** — L1/BN/Taylor/APoZ to nasze warianty
  pod wspólny operator; KMEANS nie jest średnią klastra z Cut Less Fold
  More. Do twierdzenia „lepsi od literatury" trzeba je w E011 odtworzyć.
- **Szerokiego reżimu** — jedna architektura (EfficientNet-B0+SE),
  ratio 10/30, T≤20, brak 50%+.
- **Niezależnego testu** — dev wybierał epokę baseline'u; rezerwa i test
  są nietknięte.

## 7. Co wyszło (skrót, bez nadinterpretacji)

- **Primary:** jedyną miarą, która przeszła próg, jest **TAYLOR**:
  +1,03 pp vs RANDOM, CI [+0,26; +1,79], Holm p=0,0047 (Tiny 30%, T=5).
- **Replikacja:** na CIFAR-100 10%, T=5 TAYLOR dał +0,36 pp,
  CI [−0,42; +1,12] → **brak replikacji**. Decyzja protokołu:
  `survivor_not_replicated`.
- **Warto zauważyć:** na Tiny 30% TAYLOR jest ponad RANDOM na każdym
  budżecie (T=1: +2,47; T=5: +1,03; T=10: +1,40; T=20: +1,05 pp).
  Na CIFAR-100 zero-shot TAYLOR ma 65,02% (vs RANDOM 49,48%), ale
  po naprawie przewaga znika. Tiny 10%, T=5: wszystkie miary ~56%.

## 8. Co to znaczy dla E011

E011 ma sens jako **wąskie, 3-seedowe potwierdzenie** TAYLORA (z kontrolami
PRE/POST/RANDOM i ewentualnie REC/L1), z wiernymi baseline'ami z literatury
i porównaniem head-to-head. Jednym seedem nie ogłaszamy nowości — obecny
wynik jest kandydatem, nie dowodem.
