# 2026-10-11_E006_kryterium_bramkowe_predykcja

Wersja wykonania: **1.1** (zamrożona; config `protocol_version` 1.1).
Historia: 0.1 → 0.2-draft → 1.0 (pierwszy bieg nieudany) → **1.1**;
szczegóły w sekcji „Historia wersji". Status wykonania podaje manifest biegu.

Badanie **predykcyjne** (bez pełnej kompresji): czy zachowanie bramek SE
przewiduje rzeczywistą szkodę scalenia pary kanałów **ponad** podobieństwo
aktywacji po SE (`s_post`). To etap „wykrycie zjawiska" — sukces nie
dowodzi jeszcze przewagi pełnej metody kompresji.

## Pytanie i hipoteza

Dla par o podobnym `s_post`, ale różnym zachowaniu bramek, rzeczywista
szkoda scalenia ma się różnić, a sygnał z bramek ma to przewidywać.
Kontrola główna: `s_post`. Wynik negatywny zamyka wątek bramek jako
sygnału ponad z i zostanie zaraportowany.

## Baseline i dane

- Model: `last.pt` z E005, SHA-256
  `eb03bb9b9709265134abddceec724243f6a96d2426a30683e2939e0a4531f415`
  (Tiny ImageNet, 96×96, 200 klas, EfficientNet-B0 z SE).
- Blok: `features.4.1` (480 kanałów, mapy 6×6).
- Kalibracja z train E005: **A = 1000 obrazów i B = 1000 obrazów**
  (po 5/klasę), rozłączne, seed 20261012 ustalony przed uruchomieniem.
  A: dopasowanie alpha/beta i wszystkie sygnały; B: pomiar szkody.
  Dev, rezerwa i test nietknięte.

## Pary kandydackie (selekcja bez wiedzy o szkodzie)

- **K = 150 par rozłącznych** (300 z 480 kanałów; reszta nieużywana) —
  rozłączność usuwa główną zależność między parami, więc bootstrap po
  parach jest uprawniony; pokrycie kanałów raportujemy.
- Losowanie deterministyczne (seed 20261012) z **kubełków kwantylowych
  `s_post`** (5 kubełków równej liczności × 30 par) — podział po randze
  `s_post` liczonego na A dla wszystkich par; ma dawać spektrum od par
  mało do bardzo podobnych. Deklaracja: kubełki po `s_post` (kontrola
  główna), nie po h, żeby wewnątrz kubełka zmienność bramek nie była
  tylko proxy `s_post`. **Korekta 1.1:** wersja 1.0 używała kubełków
  równych szerokości; rozkład `s_post` jest skrajnie skupiony (mediana
  0,0076; 110 216 z 114 960 par w jednym kubełku) i w najwyższym kubełku
  było 276 par na 24 kanałach (maks. 12 par rozłącznych przy wymaganych
  30) — bieg 1.0 nie mógł się udać. Zmiana jest wymuszona wykonalnością,
  nie wynikiem: w biegu 1.0 żadna szkoda nie została policzona.

## Szkoda rzeczywista (ground truth, na B)

Dla pary (c,d), reprezentant c, usuwane d — operator **zgodny z E003**:
- alpha = ⟨g_c,g_d⟩/⟨g_c,g_c⟩ (g = avgpool(h), wejście fc1), beta =
  ⟨z_c,z_d⟩/⟨z_c,z_c⟩ na pełnych mapach; dopasowanie na A;
- usunięcie producenta d (expansion, BN, depthwise), scalenie kolumny
  fc1 dla c, usunięcie wiersza/biasu fc2 dla d, scalenie kolumny
  projekcji dla c — jak [folding.py E003](../2026-10-09_E003_kompresja_10proc_parametrow/folding.py);
- pomiar na B **bez dostrajania**, na wejściach nauczyciela:
  `branch_relative_mse` i `block_relative_mse` (suma (y_s−y_t)²/suma y_t²).
Szkoda na B, sygnały na A — brak przecieku dopasowania.

## Sygnały (na A)

Pierwszorzędny (jeden, pre-registrowany):

- **`g_pert`** — przewidywane zaburzenie bramek przez **skompensowany**
  operator: dla obrazu x zmiana wejścia fc1 wynosi dokładnie
  `delta(x) = W1[:,d] · (alpha·g_c(x) − g_d(x))`; sygnał to średnia
  ‖delta(x)‖/‖s_full(x)‖ po A (s_full = pełna preaktywacja fc1).
  Liczony wprost z wag po kompensacji — spójny z ground truth, nie z
  „surowym" usunięciem kanału.

Wtórne (eksploracja, oznaczone jawnie, bez wybierania po wyniku):
`s_post` (kontrola), `s_pre`, `g_corr` (cosinus profili a_c,a_d),
`g_lstd` (std log-stosunku wag), `g_flip` (częstość zmiany znaku
a_c−a_d), `g_amp` (cosinus h ważony min(mean a_c, mean a_d) — stała
reguła), `random` (seed stały).

## Kryterium oceny (ustalone przed uruchomieniem)

1. Kontrola dodatnia: Spearman(`s_post`, szkoda) ujemny (raport).
2. **Test pierwszorzędny (jeden):** różnica Spearmanów
   Δρ = ρ(`g_pert`) − ρ(`s_post`) na B. Bootstrap po 150 parach,
   10 000 powtórzeń, seed 20261013; przedział 95%. Sukces zjawiska:
   dolna granica przedziału **> 0**.
3. Diagnostyka przyrostowa: korelacja cząstkowa (rangi)
   `g_pert`–szkoda po usunięciu wpływu `s_post`.
4. Sygnały wtórne: raport deskryptywny + test permutacyjny maksimum
   (rodzina wtórna, kontrola FWER w przybliżeniu); bez wpływu na decyzję.

Moc: przy K=150 par dolna granica wykrywalnej różnicy korelacji to
~0,15–0,20. Minimalny efekt wykrywalny raportujemy po biegu z szerokości
przedziału bootstrapu; nie dobieramy progu po wynikach.

## Cztery GPU

Obliczenia lekkie: rank r liczy A/B na swojej 1/4 obrazów (teacher),
pary szkód r::4 na swoim shardzie B; sumy po parach redukowane
all_reduce. Kolejność i agregacja deterministyczne. To jeden pomiar
podzielony na 4 GPU, nie cztery niezależne powtórzenia.

## Ograniczenia z góry

Jeden model i blok; szkoda lokalna bloku na wejściach nauczyciela
(bez wpływu zmienionych wejść dalszych bloków); pary rozłączne, ale
współdzielą model i obrazy; brak niezależnego testu. Sukces upoważnia
do E007 (pełna kompresja: kryterium bramkowe vs POST/PRE/RANDOM, równe
budżety, dostrajanie jak w E004). Porażka zamyka kierunek bramek.

## Kolejne kroki

1. Implementacja `e006_metrics.py` (fold pojedynczego bloku, sygnały,
   statystyki) + kontrole CPU na danych syntetycznych.
2. `measure_e006.py` (4 GPU) + `launch_e006.py`, konfiguracja i freeze
   (hashe baseline'u i danych wpisane przed uruchomieniem).
3. Uruchomienie, weryfikacja artefaktów, raport WYNIKI.md z jedną
   decyzją o dalszym kierunku.

## Historia wersji

- **1.0** (config `protocol_version` 1.0, commit dc7b47f): pierwsze
  uruchomienie `runs/E006/20261011T051120Z_dc7b47f_seed42_tl2bv200`
  **nieudane** — `sample_disjoint_pairs` zgłosił niedobór par w kubełkach
  równych szerokości `[3, 4]` (osiągnięto `[30, 30, 30, 10, 4]`); żadna
  szkoda nie została policzona; run zachowany jako failed.
- **1.1**: kubełki kwantylowe (równej liczności) zamiast równoszerokich,
  K=150 i kryterium oceny bez zmian; `pairs.binning = "quantile"` w
  configu; skrócony timeout NCCL (30→10 min, tylko szybsze sprzątanie po
  błędzie). Weryfikacja na zapisanych aktywacjach nieudanego biegu: każdy
  kubełek ma ~22 992 kandydatów i pojemność ~230 par rozłącznych;
  symulacja algorytmu daje 5×30 = 150 par i 300 kanałów. Decyzja
  zapisana w [docs/DECISIONS.md](../../docs/DECISIONS.md).
