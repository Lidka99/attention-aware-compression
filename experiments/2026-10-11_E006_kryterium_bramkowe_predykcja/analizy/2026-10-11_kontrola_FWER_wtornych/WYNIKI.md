# E006 — kontrola FWER sygnałów wtórnych

Data: 2026-10-11 UTC. **Status: completed** (analiza CPU po biegu).
Wejście: zamrożony `pairs.csv` biegu
`runs/E006/20261011T053842Z_eb6b660_seed42_c5gen8pp` (commit eb6b660,
protokół 1.1). Kod: [analiza_fwer.py](analiza_fwer.py), wynik:
`wynik_fwer.json`.

## Co i po co

OPIS v1.1, punkt 4 kryterium oceny, deklarował dla sygnałów wtórnych
raport deskryptywny **oraz** test permutacyjny maksimum (kontrola FWER
rodziny w przybliżeniu). Bieg raportuje tylko korelacje; ta analiza
domyka zapowiedziany element na zamrożonych danych — bez nowej
inferencji modelu i bez wpływu na decyzję pierwszorzędną.

## Wynik

| Sygnał | ρ (Spearman) | p permutacyjne (2-str.) |
|---|---:|---:|
| `s_pre` | −0,169 | 0,038 |
| `g_corr` | +0,160 | 0,046 |
| `g_lstd` | −0,187 | 0,022 |
| `g_flip` | +0,021 | 0,802 |
| `g_amp` | −0,168 | 0,039 |

Statystyka maksimum |ρ| = 0,187 (`g_lstd`); **p_FWER = 0,081**
(10 000 permutacji, seed 20261014) — rodzina nie przechodzi progu 0,05.
Żadnego z tych sygnałów nie traktujemy jako odkrycia.

## Ograniczenia

Analiza wykonana po biegu (na zamrożonym `pairs.csv`); wynik permutacji
zależy od jawnego ziarna; rodzina zdefiniowana przed biegiem w OPIS;
brak niezależnego testu. Bez wpływu na wynik pierwszorzędny E006
(negatywny, [WYNIKI](../../WYNIKI.md)).
