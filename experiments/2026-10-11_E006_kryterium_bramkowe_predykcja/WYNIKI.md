# E006 — wynik badania predykcyjnego kryterium z bramek SE

Data: 2026-10-11 UTC. **Status wykonania: completed.**
Protokół: [OPIS.md](OPIS.md) v1.1 (zamrożony jako config `protocol_version` 1.1).
Bieg: **`runs/E006/20261011T053842Z_eb6b660_seed42_c5gen8pp/`**,
commit **`eb6b660`**, dirty=false.

## Wynik główny

Pytanie: czy zachowanie bramek SE przewiduje rzeczywistą szkodę scalenia
pary kanałów **ponad** podobieństwo aktywacji po SE (`s_post`)?

| Miara | Wartość |
|---|---:|
| Kontrola dodatnia: Spearman(`s_post`, szkoda) | **−0,155** |
| Sygnał pierwszorzędny: Spearman(`g_pert`, szkoda) | **+0,044** |
| Test główny: Δρ = ρ(`g_pert`) − ρ(`s_post`) | **+0,199** |
| 95% przedział bootstrap (150 par, 10 000 powtórzeń) | **[−0,020; +0,411]** |
| Kryterium sukcesu: dolna granica przedziału > 0 | **NIE** (`primary_passed = false`) |
| Korelacja cząstkowa rang `g_pert` po usunięciu `s_post` | **+0,047** |
| Pary / użyte kanały | 150 / 300 (z 480) |

Pomiar bez dostrajania, na zbiorze B (1000 obrazów, wejścia nauczyciela);
szkoda pierwszorzędna `block_relative_mse` (mediana 7,5×10⁻⁶, maks.
1,1×10⁻²). Wariant `branch_relative_mse` ma identyczne rangi i identyczne
wartości ρ, więc wniosek nie zależy od wyboru z tych dwóch miar.

## Interpretacja

- **Kontrola dodatnia działa kierunkowo:** podobniejsze kanały po SE
  (wyższe `s_post`) mają mniejszą szkodę scalenia (ρ = −0,155); sygnał
  słaby, ale zgodny z oczekiwaniem.
- **Sygnał bramkowy `g_pert` nie przewiduje szkody:** ρ = +0,044 to
  praktycznie zero, a korelacja cząstkowa po usunięciu `s_post` wynosi
  +0,047. Dodatnie Δρ = +0,199 bierze się w całości z ujemnego
  ρ(`s_post`), nie z siły sygnału.
- Szerokość przedziału bootstrap to ±0,21: efektów słabszych niż ~0,2 nie
  odróżnilibyśmy od zera przy K=150. Widać jednak nie tylko „brak
  istotności", ale wartość bliską zeru.
- Zgodnie z protokołem **wynik negatywny zamyka wątek bramek SE jako
  sygnału ponad `s_post`**. To etap „wykrycie zjawiska"; brak podstaw do
  pełnej kompresji E007 z kryterium bramkowym.

## Sygnały wtórne (deskryptywnie, bez wpływu na decyzję)

| Sygnał | Spearman | p permutacyjne (2-str., 10 000) |
|---|---:|---:|
| `s_pre` | −0,169 | 0,038 |
| `g_corr` | +0,160 | 0,046 |
| `g_lstd` | −0,187 | 0,022 |
| `g_flip` | +0,021 | 0,802 |
| `g_amp` | −0,168 | 0,039 |

Kontrola FWER rodziny wtórnej (statystyka maksimum, 10 000 permutacji,
seed 20261014): **p = 0,081** — rodzina nie przechodzi progu 0,05.
Szczegóły: [analiza FWER](analizy/2026-10-11_kontrola_FWER_wtornych/WYNIKI.md).
Wszystkie te wartości są na poziomie szumu; zapisane wyłącznie dla
kompletności.

## Dane i wykonanie

- Baseline: `last.pt` z E005 (SHA-256
  `eb03bb9b9709265134abddceec724243f6a96d2426a30683e2939e0a4531f415`),
  EfficientNet-B0 z SE, Tiny ImageNet, 96×96, 200 klas, od losowych wag.
- Blok: `features.4.1` (ekspansja 480 kanałów, mapy 6×6, wejście 80×6×6).
- Kalibracja A: 1000 obrazów train (5/klasę), pomiar szkody B: 1000
  obrazów (5/klasę), rozłączne, seed 20261012. Dev, rezerwa i oficjalny
  test: nietknięte.
- Pary: 150 rozłącznych (300/480 kanałów), kubełki **kwantylowe**
  `s_post` (5×30, `pairs.binning = "quantile"`), wybór bez wiedzy
  o szkodzie; `s_post` wybranych par: [−0,104; +0,295].
- 4 GPU (RTX 2080 Ti): jeden pomiar podzielony na 4 rangi (nie cztery
  niezależne powtórzenia), deterministycznie, TF32 wyłączone. Czas biegu:
  **5 min 52 s**.
- Self-test CPU (w tym nowa regresja skośnego rozkładu `s_post`) przeszedł
  przed biegiem. Nieudane próby: wersja 1.0 `...051120Z` (failed —
  niedobór par; poprawka protokołu 1.1, [DECISIONS](../../docs/DECISIONS.md)),
  start techniczny `...053110Z` (przerwany SIGTERM, bez wyniku,
  `NOTE_ABORTED.md` w katalogu runu).

## Artefakty i hashe

- `artifacts.sha256`: **32/32 plików OK** (weryfikacja `sha256sum -c`).
- Kluczowe: `result.json` `a9fa8bbccb1c2a6adb1c4b01b9cd6f8270699898ca1ff2b81ef2f6e91ace2ff6`,
  `pairs.csv` `475044a089f2e6e28567d27b31069246610504838cd0287e2c83795655103d2b`,
  `manifest.json` `9b242142fdfe99d79f421782204ae875f25632dbb5ec23b64d6a71f167537fd8`.

## Ograniczenia

- Jeden model i jeden blok; szkoda lokalna bloku na wejściach nauczyciela
  (bez propagacji do dalszych warstw).
- Pary są rozłączne kanałowo, ale współdzielą model i obrazy; bootstrap
  po parach zakłada niezależność, która jest tu przybliżeniem.
- Brak niezależnego testu; przy K=150 czułość na Δρ to ~0,15–0,20.
- Wnioski dotyczą wykrycia zjawiska, nie pełnej kompresji sieci.

## Następny krok

Wątek bramek zamknięty. Decyzja o dalszym kierunku należy do użytkowniczki;
na stole (z [DECISIONS](../../docs/DECISIONS.md), 2026-10-11):

1. szybki test **30% redukcji** na obecnym baseline — czy przy większym
   budżecie selekcja kanałów w ogóle zaczyna mieć znaczenie (E003/E004
   testowały tylko 10%),
2. baseline **224×224 z mocniejszą recepturą** przeciw przeuczeniu pod
   publikację.

Propozycja: najpierw (1) jako tani test rozstrzygający, potem (2).
