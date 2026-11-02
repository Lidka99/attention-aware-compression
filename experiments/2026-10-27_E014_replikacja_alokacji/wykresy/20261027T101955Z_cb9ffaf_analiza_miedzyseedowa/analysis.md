# E014 — analiza międzyseedowa TILT+0.30 vs UNIFORM

Wygenerowano: 2026-10-27T10:26:50.282209+00:00  
Run: `20261027T094707Z_267904d_6a0e93eb` (git `267904d632d70b69446ee3b4fa4239d455e66695`)  
Opis eksperymentu: [OPIS.md](../../OPIS.md)  
Skrypt: `analyze_e014.py` (SHA-256 w `META.json`)

## Jak czytać

Jednostką jest **seed** (n=4), nie pojedyncza para seed×budżet. Trzy budżety korzystają z tych samych czterech checkpointów, więc 12 różnic seed×budżet **nie jest** 12 niezależnymi replikacjami.

seed42 był użyty selekcyjnie w E013; jego wynik w E014 odtwarza E013 i nie jest niezależną replikacją. Dlatego obok pełnego zbioru 42–45 raportujemy osobno seedy **43–45**.

Przedziały t-95% przy n=4 (df=3) i n=3 (df=2) są **opisowe i ostrożne**; przy tak małym n nie szacują stabilnie wariancji populacji. Wszystkie różnice liczone są jako `TILT − UNIFORM` w punktach procentowych (pp).

## 1. Rzeczywiste budżety

| Budżet | Wariant | Parametry | Redukcja | Odch. od celu | Δ redukcji vs UNIFORM | Dodatkowe param. vs UNIFORM |
|---:|---|---:|---:|---:|---:|---:|
| 30% | UNIFORM | 7 808 058 | 30,123% | +0,123 pp | — | — |
| 30% | TILT+0.30 | 7 821 572 | 30,002% | +0,002 pp | -0,121 pp | 13 514 |
| 30% | TILT+0.15 | 7 821 534 | 30,002% | +0,002 pp | -0,121 pp | 13 476 |
| 30% | MAG2 | 7 808 058 | 30,123% | +0,123 pp | +0,000 pp | 0 |
| 50% | UNIFORM | 5 584 580 | 50,021% | +0,021 pp | — | — |
| 50% | TILT+0.30 | 5 586 898 | 50,001% | +0,001 pp | -0,021 pp | 2 318 |
| 50% | TILT+0.15 | 5 586 963 | 50,000% | +0,000 pp | -0,021 pp | 2 383 |
| 50% | MAG2 | 5 584 580 | 50,021% | +0,021 pp | +0,000 pp | 0 |
| 70% | UNIFORM | 3 345 905 | 70,056% | +0,056 pp | — | — |
| 70% | TILT+0.30 | 3 352 160 | 70,000% | +0,000 pp | -0,056 pp | 6 255 |
| 70% | TILT+0.15 | 3 352 087 | 70,001% | +0,001 pp | -0,055 pp | 6 182 |
| 70% | MAG2 | 3 345 905 | 70,056% | +0,056 pp | +0,000 pp | 0 |

Cel budżetu jest liczony na całym modelu. UNIFORM i MAG2 mają identyczną liczbę parametrów; TILT trafia bliżej celu i zachowuje nieco więcej parametrów. To niewielka różnica na korzyść TILT, znacznie mniejsza niż efekt top-1, ale nie jest zerowa i nie została tu renormalizowana.

## 2. TILT+0.30 vs UNIFORM — różnice na seedach

| Budżet | seed42 | seed43 | seed44 | seed45 | średnia | SD | zakres |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30% | +0,58 | +0,94 | +0,88 | +1,06 | +0,865 | 0,204 | 0,48 |
| 50% | +4,34 | +3,52 | +3,94 | +3,28 | +3,770 | 0,468 | 1,06 |
| 70% | +10,70 | +9,56 | +11,16 | +13,68 | +11,275 | 1,739 | 4,12 |

Wszystkie **12/12** różnic jest dodatnich.

## 3. Podsumowanie statystyk

| Budżet | Podzbiór | n | średnia | SD | min | max | zakres | 95% CI (t) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 30% | 42–45 | 4 | +0,865 | 0,204 | +0,58 | +1,06 | 0,48 | [+0,54; +1,19] |
| 30% | 43–45 | 3 | +0,960 | 0,092 | +0,88 | +1,06 | 0,18 | [+0,73; +1,19] |
| 50% | 42–45 | 4 | +3,770 | 0,468 | +3,28 | +4,34 | 1,06 | [+3,03; +4,51] |
| 50% | 43–45 | 3 | +3,580 | 0,334 | +3,28 | +3,94 | 0,66 | [+2,75; +4,41] |
| 70% | 42–45 | 4 | +11,275 | 1,739 | +9,56 | +13,68 | 4,12 | [+8,51; +14,04] |
| 70% | 43–45 | 3 | +11,467 | 2,077 | +9,56 | +13,68 | 4,12 | [+6,31; +16,63] |

## 4. Pozostałe warianty vs UNIFORM

| Wariant | Budżet | średnia 42–45 | SD | zakres | średnia 43–45 | SD 43–45 |
|---|---:|---:|---:|---:|---:|---:|
| TILT+0.15 | 30% | +1,015 | 0,456 | 1,10 | +1,180 | 0,386 |
| TILT+0.15 | 50% | +2,415 | 0,528 | 1,22 | +2,240 | 0,485 |
| TILT+0.15 | 70% | +8,320 | 1,532 | 3,48 | +8,607 | 1,741 |
| MAG2 | 30% | -1,140 | 0,636 | 1,40 | -0,860 | 0,370 |
| MAG2 | 50% | -4,425 | 1,094 | 2,48 | -4,067 | 1,012 |
| MAG2 | 70% | -14,905 | 5,090 | 11,18 | -13,660 | 5,438 |

TILT+0.30 vs MAG2 (średnia ± SD, 42–45): 30%: +2,005 ± 0,441; 50%: +8,195 ± 1,466; 70%: +26,180 ± 3,520 pp. Dodatnie pary: 12/12.

## 5. Metryki wtórne TILT+0.30 vs UNIFORM

Wartości to średnia ± SD różnicy na czterech seedach. Dla NLL, MAE, RMSE i KL wartość ujemna oznacza lepszy TILT; dla agreement dodatnia oznacza lepszy TILT. Agreement podano w pp.

| Budżet | ΔNLL | Δagreement | ΔMAE | ΔRMSE | ΔKL |
|---:|---:|---:|---:|---:|---:|
| 30% | -0,0238 ± 0,0032 | +1,20 ± 0,08 | -0,1030 ± 0,0204 | -0,1738 ± 0,0265 | -0,0368 ± 0,0045 |
| 50% | -0,1191 ± 0,0199 | +4,26 ± 0,67 | -0,0346 ± 0,0188 | -0,0647 ± 0,0309 | -0,1245 ± 0,0220 |
| 70% | -0,2952 ± 0,0623 | +11,65 ± 2,05 | -0,1101 ± 0,0221 | -0,1687 ± 0,0373 | -0,3050 ± 0,0666 |

Dla każdej z tych metryk znak różnicy był zgodny na wszystkich 12 parach seed×budżet (NLL/MAE/RMSE/KL ujemne, agreement dodatni).

## 6. Weryfikacja i pliki

- manifest: `completed`, official_test_used=`False`;
- sprawdzono 48 plików summary.json (4 seedy × 12 wariantów);
- baseline na seedach: top-1 95,28–95,60%, poprawnych 4764–4780 z 5000; baseline różni się między seedami, więc porównania są parowane w obrębie seeda;
- parametry i target_params spójne między seedami; maks. odchylenie od celu 0,123 pp;
- brak GPU, treningu i zmiany istniejących wyników.

Pliki: `analysis.json`, `budgets.csv`, `seed_level.csv`, `summary_stats.csv`, `variant_means.csv`, `META.json` oraz:

- `roznice_tilt030_wg_budzetu.png` / `.pdf`
- `top1_wg_budzetu.png` / `.pdf`
- `zrealizowany_budzet.png` / `.pdf`

## Ograniczenia i wniosek

Kierunek TILT+0.30 jest spójny na wszystkich czterech seedach i trzech budżetach, także po wyłączeniu selekcyjnego seed42. Nadal jest to jeden dataset (CIFAR-10), dev, cztery checkpointy z jednego przepisu i adaptacja operatora E012; brak niezależnego testu. Małe n i współdzielone seedy ograniczają wnioskowanie statystyczne — raportujemy kierunek i rozrzut, nie istotność populacyjną.
