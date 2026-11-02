# E014 — krótki raport z analizy międzyseedowej

Data analizy: 2026-10-27. Status: **analiza CPU ukończona**; protokół i wyniki
źródłowe bez zmian. Run: `runs/E014/20261027T094707Z_267904d_6a0e93eb`.

Kod: [analyze_e014.py](analyze_e014.py). Pełne tabele i wykresy:
[wykresy/20261027T101955Z_cb9ffaf_analiza_miedzyseedowa/analysis.md](wykresy/20261027T101955Z_cb9ffaf_analiza_miedzyseedowa/analysis.md).
Audyt z komendą, logiem i hashami: `runs/E014/audits/20261027T101955Z_cb9ffaf_analiza_miedzyseedowa/`.
Protokół: [OPIS.md](OPIS.md).

## Co sprawdziliśmy i na czym

Czy przewaga **TILT+0.30 nad UNIFORM** powtarza się na czterech checkpointach
E011 (seedy 42–45), przy trzech budżetach redukcji parametrów: 30/50/70%.
Analiza czyta 48 surowych `summary.json` (4 seedy × 3 budżety × 4 warianty),
sprawdza manifest, baseline, liczbę przykładów, spójność parametrów i
rzeczywiste budżety, a następnie **paruje seedy** i liczy różnice top-1 po BN.
Nie ma treningu, GPU ani nowej inferencji.

## Jak czytać wynik

Jednostką jest **seed (n=4)**, nie pojedyncza para seed×budżet. Trzy budżety
korzystają z tych samych czterech checkpointów, więc **12 różnic nie jest 12
niezależnymi replikacjami**. `seed42` był użyty selekcyjnie w E013; jego wynik
w E014 odtwarza E013, więc nie jest niezależną replikacją. Dlatego obok
pełnego zbioru 42–45 raportujemy osobno **seedy 43–45**.

Przedziały t-95% przy n=4 (df=3) i n=3 (df=2) są **opisowe i ostrożne**;
przy tak małym n nie szacują stabilnie wariancji populacji.

## Wynik: TILT+0.30 − UNIFORM [pp]

| Budżet | seed42 | seed43 | seed44 | seed45 | średnia 42–45 | SD | zakres |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30% | +0,58 | +0,94 | +0,88 | +1,06 | **+0,865** | 0,204 | +0,58…+1,06 |
| 50% | +4,34 | +3,52 | +3,94 | +3,28 | **+3,770** | 0,468 | +3,28…+4,34 |
| 70% | +10,70 | +9,56 | +11,16 | +13,68 | **+11,275** | 1,739 | +9,56…+13,68 |

| Budżet | Podzbiór | n | średnia | SD | zakres | 95% CI (t) |
|---:|---|---:|---:|---:|---:|---:|
| 30% | 42–45 | 4 | +0,865 | 0,204 | 0,48 | [+0,54; +1,19] |
| 30% | **43–45** | 3 | **+0,960** | 0,092 | 0,18 | [+0,73; +1,19] |
| 50% | 42–45 | 4 | +3,770 | 0,468 | 1,06 | [+3,03; +4,51] |
| 50% | **43–45** | 3 | **+3,580** | 0,334 | 0,66 | [+2,75; +4,41] |
| 70% | 42–45 | 4 | +11,275 | 1,739 | 4,12 | [+8,51; +14,04] |
| 70% | **43–45** | 3 | **+11,467** | 2,077 | 4,12 | [+6,31; +16,63] |

Wszystkie **12/12** różnic TILT+0.30 − UNIFORM jest dodatnich. Po wyłączeniu
seed42 kierunek i wielkość efektu się utrzymują; przy 70% przedział dla
seedów 43–45 jest szeroki, co pokazuje, że trzy seedy nie ustalają precyzyjnie
wielkości efektu.

Pozostałe warianty (średnia różnica vs UNIFORM, 42–45 / 43–45):

- **TILT+0.15:** +1,015 / +1,180 pp przy 30%, +2,415 / +2,240 pp przy 50%,
  +8,320 / +8,607 pp przy 70%. Przy 30% jest średnio o 0,15 pp lepszy od
  TILT+0.30, ale przy 50/70 wyraźnie gorszy.
- **MAG2:** −1,140 / −0,860 pp, −4,425 / −4,067 pp, −14,905 / −13,660 pp.
  TILT+0.30 bije MAG2 średnio o +2,0 / +8,2 / +26,2 pp; wszystkie 12 par
  seed×budżet są dodatnie.

Metryki wtórne TILT+0.30 vs UNIFORM są zgodne kierunkowo na wszystkich 12
parach: NLL, MAE, RMSE i KL niższe, agreement wyższy (np. przy 70%:
ΔNLL −0,295 ± 0,062; Δagreement +11,65 ± 2,05 pp).

## Rzeczywiste budżety

TILT trafia bliżej celu niż UNIFORM/MAG2 i zachowuje nieco więcej parametrów:

| Budżet | UNIFORM/MAG2 | TILT+0.30 | Dodatkowe parametry TILT | Różnica redukcji |
|---:|---:|---:|---:|---:|
| 30% | 7 808 058 (30,123%) | 7 821 572 (30,002%) | +13 514 (+0,121% oryginału) | −0,121 pp |
| 50% | 5 584 580 (50,021%) | 5 586 898 (50,001%) | +2 318 (+0,021%) | −0,021 pp |
| 70% | 3 345 905 (70,056%) | 3 352 160 (70,000%) | +6 255 (+0,056%) | −0,056 pp |

Ta niewielka różnica budżetu działa **na korzyść TILT**, więc nie wyjaśnia
efektu, ale nie jest zerowa i nie została tu renormalizowana. MAG2 ma
dokładnie ten sam budżet co UNIFORM.

## Co z tego wynika

Kierunek z E013 utrzymuje się: nierówna alokacja (TILT+0.30) poprawia top-1
po BN względem jednolitego foldingu na każdym seedzie i budżecie, także na
seedach 43–45, których E013 nie użył do wyboru kierunku. To nadal **jeden
dataset (CIFAR-10), dev, cztery checkpointy z
jednego przepisu i adaptacja operatora E012**; brak niezależnego testu i
drugiego datasetu. Nie traktujemy tego jako dowodu istotności populacyjnej.

## Następny krok

Powtórzyć alokację na CIFAR-100 i Tiny ImageNet oraz opisać mechanizm
(depth-aware vs predykcja wrażliwości), zanim zapadnie decyzja o publikacji.
