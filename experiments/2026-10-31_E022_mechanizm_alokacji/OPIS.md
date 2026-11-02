# 2026-10-31_E022_mechanizm_alokacji

Status: **frozen**, 2026-10-31; przed nowym uruchomieniem i obliczeniem
statystyk E022. Reanaliza istniejących wyników, nie niezależna prerejestracja.

## Zakres i pochodzenie
Pięć pytań: głębokość a szkoda pre/post BN; odzysk BN i krytyczne grupy;
słabszy CIFAR-100; ochrona grup przez TILT; rekonstrukcja przy stałym ratio.
Checkpointy E011/E015/E019, żadnego treningu. Wyłącznie train do BN i dev
do oceny. Test, rezerwa, nowe dane i 96×96 wykluczone.

Inwentaryzacja ujawniła pominięty w STATUS E020: sensitivity
`runs/E020/20261031T055920Z_cb9ffaf_sens_e650f55f` (49/49) i ablacja
`runs/E020/20261031T061615Z_cb9ffaf_abl_bbb35b29` (28/36 completed,
8 brakujących Tiny). Jego manifest ablacji pozostał running, logi puste;
przyczyna przerwania nieustalona. Nie zmieniamy tych artefaktów.
E022 importuje 98 pomiarów E016 S1 i 49 E020, alokacje E014/E015/E019
oraz 28 ablacji E020. Oblicza tylko 8 brakujących ablacji ANTI-TILT.

## Pomiary GPU
Operator i helpery E020/E012/E014 bez zmian naukowych. ANTI-TILT −0.30,
budżety i checkpointy zgodne z manifestem E020; target = faktyczny UNIFORM,
tolerancja 1%, bez przekroczenia targetu. Fold seed42, BN seed20261024,
batch128/GPU, pełny train. Cztery niezależne workery GPU0–3 po dwa zadania;
nie są to cztery powtórzenia. Limit 90 min, nowe artefakty ≤0,5 GiB.
Zapis tylko summary/diagnostics, bez nowych checkpointów i kopii danych.
Hash protokołu, kodu i źródłowych summary w manifeście. Środowisko attention.

## Analiza i kryteria zamrożone
- Szkoda pp = 100*(baseline−compressed); odzysk = pre_damage−post_damage.
  Odzysk względny = suma odzysku / suma szkody pre (nie średnia ilorazów).
  Raportować też kontrolę ratio0 po BN i szkodę post względem tej kontroli.
- Grupy: 12 w kolejności operatora; poziomy 25/50/75/90% kanałów osobno.
  Primary poziom75%, pozostałe analizy wrażliwości. Spearman głębokość–szkoda
  pre i post, trzy najbardziej krytyczne grupy. To opis, bez założenia
  monotonicznej wrażliwości. Raport szkody względnej względem baseline.
- Ochrona = ratio_UNIFORM−ratio_TILT (dodatnia oznacza mniej kompresji).
  Korelacja ochrony ze szkodą **przy wspólnym ratio75%**; nie korelować
  primary ze szkodą przy zmiennym ratio TILT (konfundowanie poziomem).
  Wsparcie silnej ochrony: rho≥0,5; porównać wszystkie trzy zbiory.
  Dodatkowo interpolacja krzywych szkody (kotwica0, bez ekstrapolacji)
  pod rzeczywiste ratios; suma to wskaźnik opisowy, nie model addytywny.
- Rekonstrukcja: rho ze szkodą post przy każdym stałym ratio, n=12.
  Wsparcie na zbiorze: rho≥0,5 w co najmniej3/4 poziomów. Mocna wspólna
  hipoteza: wsparcie na co najmniej2/3 zbiorów. Pooled rho tylko kontrast.
- Ablacja: przy globalnym budżecie50% ANTI−UNIFORM<0 i ≥3/4 ujemnych
  seedów na CIFAR-10 i Tiny wspiera znaczenie kierunku; na CIFAR-100
  osobno sprawdzić ANTI−UNIFORM≥0 lub ANTI−TILT>0. Nie przesądza przyczyn.
- Agregacja alokacji: średnia i SD po4 seedach, nie po grupach jako seedach.
  Profile grupowe mają tylko seed42; bez populacyjnych wniosków przyczynowych.
- Integralność: 147/147 grupowych z kontrolami, 36/36 ablacji (28+8),
  identyczność kontroli pre BN ≤0,05pp i agreement≥0,999999,
  wszystkie nowe workery exit0 i wykorzystane GPU0–3, zgodne hashe wejść.

E020 zawierał niejednoznaczne znaki w kryteriach H3/W2; E022 nie zmienia
tamtego protokołu, stosuje powyższe jawne definicje. Nie dobiera progów
ani nowej alokacji do wyników. Różnice zbiorów/stemów nie izolują przyczyny.

## Odtworzenie
Z katalogu projektu, `conda activate attention`:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B experiments/2026-10-31_E022_mechanizm_alokacji/run_e022.py
python -B experiments/2026-10-31_E022_mechanizm_alokacji/analyze_e022.py --run-dir runs/E022/<run-ID>
```
Launcher pomija ukończone źródłowe zadania E020; każde uruchomienie tworzy
nowy katalog. Nie uruchamiać ponownie po zakończeniu E022.
