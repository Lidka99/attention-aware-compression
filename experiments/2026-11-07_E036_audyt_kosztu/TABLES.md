# E036 — tabele kosztu

Run `runs/E036/20261107T231027Z_27bac186`. Wszystkie liczby poniżej są przeliczone wprost z surowych `costs.json` i plików czasów przez `tables_from_raw.py` (tylko CPU). Wartości to średnie czterech seedów; każdy model/batch ma 600 zdarzeń (3 serie×200), mediana liczona per seed. Przepustowość =128000/średnia z600 zdarzeń; peak pamięci = mediana 3 serii per seed, potem średnia4, `peak_allocated`/2^20. Jakość T−U pochodzi wyłącznie z zamrożonych referencji E036 `quality_CMA` (reuse E035), nie z nowej oceny. MAC: 1 multiply-add=1; bias/BN/ReLU poza MAC.

**Korekta:** wcześniejsze SD batch1 oraz część wartości parametrów wpisano ręcznie i nie odpowiadały temu runowi. Ta wersja liczy parametry, MAC, czasy, pamięć i SD wprost z surowych plików; starych SD nie używamy. SD to odchylenie z czterech sparowanych per-seed ilorazów (ddof=1), nie CI. Wykresy i opis: [figures/README.md](figures/README.md).

## Koszt bezwzględny

| Para | Red. | Param. U/T [M] | MAC U/T [M] | Lat. b1 U/T [ms] | Przepust. b128 U/T [obr./s] | Peak b128 U/T [MB] | Jakość T−U [pp] |
|---|---:|---:|---:|---:|---:|---:|---:|
| R/C10 | 30% | 7,808/7,804 | 386,3/426,8 | 2,82/2,87 | 12476/11825 | 195,9/205,8 | +0,81 |
| R/C10 | 50% | 5,585/5,583 | 276,4/377,0 | 2,90/2,88 | 16047/12552 | 168,2/197,9 | +3,41 |
| R/C10 | 70% | 3,346/3,346 | 166,6/298,0 | 2,87/3,01 | 15963/14128 | 207,4/188,8 | +10,99 |
| R/C100 | 30% | 7,847/7,842 | 386,3/426,8 | 2,90/2,96 | 12341/11695 | 196,0/206,0 | +0,83 |
| R/C100 | 50% | 5,582/5,581 | 275,3/376,1 | 2,81/2,84 | 15875/12387 | 167,4/198,0 | +1,30 |
| R/C100 | 70% | 3,335/3,335 | 161,9/295,1 | 2,91/2,85 | 15835/13861 | 204,4/188,8 | +2,06 |
| R/Tiny | 30% | 7,845/7,841 | 103,5/114,8 | 2,97/3,02 | 22076/20885 | 174,5/184,5 | +3,54 |
| R/Tiny | 50% | 5,623/5,623 | 75,4/102,2 | 3,06/3,09 | 32739/22975 | 104,5/176,0 | +9,36 |
| R/Tiny | 70% | 3,367/3,367 | 45,4/82,0 | 2,96/3,07 | 34665/25707 | 92,0/167,4 | +7,48 |
| P/C10 | 30% | 7,807/7,806 | 386,3/426,9 | 3,02/2,90 | 12285/11628 | 195,8/205,8 | +0,96 |
| P/C10 | 50% | 5,583/5,583 | 276,4/375,3 | 2,94/2,89 | 15782/12458 | 168,2/197,9 | +4,95 |
| P/C10 | 70% | 3,345/3,345 | 166,6/291,2 | 3,01/3,06 | 15681/13839 | 207,4/187,3 | +13,58 |
| P/C100 | 30% | 7,845/7,843 | 386,3/426,9 | 3,04/3,01 | 12283/11629 | 196,0/206,0 | +4,49 |
| P/C100 | 50% | 5,580/5,579 | 275,3/374,7 | 2,98/2,99 | 15803/12429 | 167,3/198,0 | +7,64 |
| P/C100 | 70% | 3,334/3,334 | 161,9/289,8 | 2,98/3,02 | 15690/13884 | 204,4/187,3 | +22,74 |

## Ilorazy T/U

| Para | Red. | Param. | MAC | Przepust. b128 ±SD | Latencja b1 ±SD |
|---|---:|---:|---:|---:|---:|
| R/C10 | 30% | 0,9995 | 1,105 | 0,948 ±0,001 | 1,018 ±0,023 |
| R/C10 | 50% | 0,9996 | 1,364 | 0,782 ±0,001 | 0,997 ±0,100 |
| R/C10 | 70% | 0,9999 | 1,789 | 0,885 ±0,003 | 1,046 ±0,075 |
| R/C100 | 30% | 0,9994 | 1,105 | 0,948 ±0,002 | 1,027 ±0,072 |
| R/C100 | 50% | 0,9998 | 1,366 | 0,780 ±0,000 | 1,012 ±0,020 |
| R/C100 | 70% | 0,9999 | 1,823 | 0,875 ±0,001 | 0,981 ±0,066 |
| R/Tiny | 30% | 0,9994 | 1,109 | 0,946 ±0,000 | 1,018 ±0,024 |
| R/Tiny | 50% | 1,0000 | 1,357 | 0,702 ±0,023 | 1,009 ±0,042 |
| R/Tiny | 70% | 0,9999 | 1,806 | 0,742 ±0,023 | 1,038 ±0,019 |
| P/C10 | 30% | 0,9999 | 1,105 | 0,947 ±0,003 | 0,965 ±0,055 |
| P/C10 | 50% | 0,9999 | 1,358 | 0,789 ±0,001 | 0,985 ±0,022 |
| P/C10 | 70% | 0,9999 | 1,748 | 0,883 ±0,002 | 1,015 ±0,024 |
| P/C100 | 30% | 0,9998 | 1,105 | 0,947 ±0,001 | 0,994 ±0,073 |
| P/C100 | 50% | 0,9998 | 1,361 | 0,787 ±0,002 | 1,006 ±0,071 |
| P/C100 | 70% | 0,9999 | 1,790 | 0,885 ±0,002 | 1,018 ±0,084 |

Uwaga: średnie ilorazy T/U bliskie1 (0,965–1,046) nie dowodzą równoważności metod ani dominacji narzutu uruchomienia; SD z czterech seedów wynosi 0,019–0,100. Przepustowość batch128 pokazuje rzeczywisty koszt: TILT jest wolniejszy o5–30% (iloraz 0,702–0,948). Parametry U i T nie są identyczne (iloraz 0,9994–1,0000), dlatego podano osobne wartości, a nie wspólną liczbę.

Kontrola: per-seed ilorazy zgodne z `analysis.json` (max |Δ| = 0.0e+00); pliki wejściowe niezmienione przed i po generowaniu.
