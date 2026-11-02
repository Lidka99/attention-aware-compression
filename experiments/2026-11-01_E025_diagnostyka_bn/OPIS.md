# 2026-11-01_E025_diagnostyka_bn

Status: **frozen**, 2026-11-01, przed oceną nowych wariantów.

## Pytanie i związek z celem
Adresujemy otwarty problem budżetu per warstwa w foldingu. E024 ujawnił,
że wspólna korekta BN obniża nieskompresowany teacher o5,83pp; zanim
zmienimy cel/wyszukiwanie alokacji, badamy to źródło zniekształcenia oceny.
E025 to diagnostyka **BN na stałych modelach**, nie nowe wyszukiwanie.
Nie zmieniamy negatywnego werdyktu E024 ani historycznych wyników.

## Wejścia i implementacja
CIFAR-100, checkpointy E01542–45. Cztery modele/seed: teacher,
UNIFORM, TILT, SEARCH z zamrożonych szerokości E024
`runs/E024/20261031T141558Z_971ba86e/seed_XX/selection.json`.
Używamy samodzielnego, zweryfikowanego operatora/danych E024, bez zmian
i bez zależności od dawnych helperów E012–E020. Dane, profile i kod
E024 mają sprawdzane hashe. Nowy kod dotyczy wyłącznie wariantów BN.

## Projekt czynnikowy
2 liczebności ×2 augmentacje ×3 procedury BN =12 komórek/model/seed:

- Liczebność **A4096** (dokładne indeksy/kolejność E024) lub **L35904**
  (historyczny train40k bez B2048 i C2048). L to A jako prefiks, następnie
  pozostałe próbki w kolejności permutacji RNG20261031.
- Obrazy **plain** (normalizacja E024) lub **aug**: jeden deterministyczny
  widok/obraz, crop32 z padding4 zerami w surowej przestrzeni [0,1],
  horizontal flip p=.5, dopiero potem normalizacja. RNG20261101+indeks
  obrazu, te same widoki przy każdej procedurze/modelu/seedzie.
- **CMA_reset**: reset mean0/var1/licznik0, momentum=None.
- **EMA_reset**: ten sam reset, momentum=.1.
- **EMA_retain**: start ze statystyk źródła/foldingu, momentum=.1;
  licznik niezerowany. To test inicjalizacji, nie trzeci estymator.

Wszystkie pozostałe moduły eval, tylko BN train, inference_mode, bez
gradientów. Batch128, drop_last=False:32 lub281 batchy (ostatni64).
CMA uśrednia po batchach, nie po próbkach. EMA waży niedawne batche:
nie staje się CMA przy większej liczbie próbek. Różnica reset/retain
zanika jak .9^N, ale różnica CMA/EMA może pozostać.
Każda komórka startuje ze świeżej kopii modelu **przed** BN.

## Ponowne użycie, liczność i dane oceny
A/plain/CMA_reset już istnieje w E024:16 wyników ponownie użytych.
Pozostałe11×4modele×4seedy=**176 nowych ocen**, łącznie192 komórki BN.
Dodatkowo16 referencji bez BN z E024, bez ponownej oceny.
Nie powtarzamy search ani historycznych komórek. Teacher logity na C
liczone jako referencja dla KL; brakujące KL teacher kontroli E024=null,
nie dopisujemy ich przez nowe obliczenia.
Ocena dev5000: top1 i NLL; C: KL względem oryginalnego teacher, bez
etykiet. Train B/C wyłączone z kalibracji; test/rezerwa nietknięte.
C/dev były już oglądane: jest to **diagnostyka na dev**, nie niezależne
potwierdzenie nowej metody ani wybór alokacji.

## Zamrożone kontrasty i kryteria
Raport wszystkich komórek, mean±SD po4 seedach, różnice parowane.
Primary: teacher, odzyskanie utraty top1 względem E024.
Primary kontrasty względem A/plain/CMA_reset (po jednym czynniku):
1. większa liczebność L/plain/CMA_reset;
2. augmentacja A/aug/CMA_reset;
3. estymator A/plain/EMA_reset.
Reset: osobno EMA_retain−EMA_reset dla każdej liczebności/augmentacji.
Pozostałe komórki testują interakcje, nie wybieramy po nich progu.
Efekt istotny praktycznie: średnia poprawa≥+1pp i≥3/4 różnic dodatnich.
Odtworzenie teacher: średnia strata top1≤1pp i strata każdego seeda≤1,5pp
wobec teacher bez BN. Raportować wszystkie spełniające komórki, bez
wybierania zwycięzcy po dev.
Na skompresowanych modelach: różnica względem własnej komórki E024,
SEARCH−TILT/UNIFORM i NLL/KL. Nie przyznajemy E024 sukcesu retrospektywnie;
SEARCH był wybierany pod inną BN, a optimum mogłoby się zmienić.
Nie wnioskować o jakości search z samej poprawy BN teacher.

## Kontrole i zasoby
Ściśle rozłączne train kalibracji i B/C/dev/rezerwa. Bez etykiet w BN.
Augmentacja przed normalizacją i identyczna niezależnie od batchowania;
plain dokładnie zgodne z E024. A prefiks L. Jednostkowa kontrola formuł
CMA/EMA na syntetycznym BN z analitycznie znanym wynikiem, liczników,
niezmienności parametrów i niezależności kopii.
Każda ocena zapisuje per-BN zmianę statystyk, liczniki przed/po, liczbę
próbek/batchy, wagę inicjalizacji EMA, hash widoków/parametrów.
Folding z etykietami klastrów zgodnymi z diagnostyką E024.
4GPU0–3: seed42+i na GPUi,44 nowe oceny/worker, jeden launcher,
Conda attention, limit90min, artefakty≤0,5GiB. Bez checkpointów, nowych
danych,96×96, treningu lub automatycznych restartów.
Sprawdzamy kody wyjścia, kompletność176+16, hashe kodu/protokołu i wejść.

## Odtworzenie
Z katalogu projektu, Conda attention:
```
CUDA_VISIBLE_DEVICES='' python -B experiments/2026-11-01_E025_diagnostyka_bn/check.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B experiments/2026-11-01_E025_diagnostyka_bn/run.py
CUDA_VISIBLE_DEVICES='' python -B experiments/2026-11-01_E025_diagnostyka_bn/analyze.py --run-dir runs/E025/<run-ID>
```
Wynik stanowi podstawę decyzji o **osobnym** następnym protokole
optymalizacji budżetu. Nie uruchamiamy nowego treningu ani search w E025.
