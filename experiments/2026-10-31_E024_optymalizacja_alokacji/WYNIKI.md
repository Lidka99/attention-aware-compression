# E024 — optymalizacja budżetu per grupa: pilotaż samodzielnej implementacji

Status: **completed, kryterium sukcesu niespełnione**, 2026-10-31.
SEARCH poprawia KL i NLL oraz trafność względem UNIFORM, ale nie pokonuje
TILT w primary top1. Dodatkowo kontrola BN ujawnia istotne ograniczenie
nowej procedury kalibracji. Nie rozszerzać automatycznie wyszukiwania.

## Problem i implementacja

Adresujemy otwarty problem wskazany przez autorów foldingu: **optymalizację
budżetu kompresji per warstwa**. W ResNet18 zmiennymi jest12 szerokości
grup zgodnych z połączeniami residualnymi, przy wspólnym limicie parametrów.
To alokacja z kalibracją, bez fine-tuningu; nie calibration-free.

Zgodnie z decyzją użytkowniczki kod napisano samodzielnie, bez importów
skryptów wcześniejszych eksperymentów. Z historii użyto wyłącznie wejść:
lokalnych danych/splitu CIFAR-100 i checkpointów E01542–45. Biblioteki:
torch/torchvision, NumPy, sklearn, HKMeans. Operator foldingu jest stały;
nie szukano miar grupowania. E023 zachowany jako **anulowany przed runem**.

Hipoteza E024: wybór alokacji przez ocenę pełnej sieci po BN uniknie
ograniczeń addytywnych wskaźników szkody pojedynczych grup z E022.
Cel: KL(teacher||candidate), temperatura1, na nieetykietowanym train.
Ograniczenie: liczba parametrów pod faktycznym UNIFORM, różnica≤0,1%.
Start to lepszy wg KL z UNIFORM/TILT; dwie rundy po24 zamiany budżetu
między grupami. Żadne wagi nie są optymalizowane.

## Wykonanie i integralność

- Protokół i kryteria: [OPIS.md](OPIS.md), frozen przed search.
- Run: `runs/E024/20261031T141558Z_971ba86e`.
- **4/4 seedy completed, 50 ocen/seed, 200/200 łącznie**, GPU0–3
  odpowiednio seedy42–45, exit codes `[0,0,0,0]`.
- Wall-time **586,48s (9,77min)**, suma czasu workerów około38,4min;
  artefakty około764KiB. Limit90min/0,5GiB spełniony.
- 4×RTX2080Ti, Conda attention, PyTorch2.12.0+cu130; wersje Python,
  CUDA i pakietów w manifest/pip-freeze. Batch128/GPU i128/przebieg.
- Train rozdzielony bez etykiet: A4096 do BN, B2048 do search, C2048
  do audytu. C i dev odczytywane do oceny dopiero po zapisie/hashu wyboru.
  A/B/C/dev/rezerwa rozłączne. C jest holdoutem wobec search, nie wobec
  treningu teacher. Test i rezerwa nie były oceniane.
- CPU check: identity dokładne, niezmienność źródła, rzeczywista
  klasteryzacja, rachunek parametrów, forward residual/FC i reguły scalania
  poprawne. Każda korekta BN sprawdza niezmienność parametrów i brak gradientów.
- Analiza sprawdza źródła, identyczność wybranego profilu, limit ocen,
  deduplikację, dopasowanie parametrów i kolejność wyboru przed dev.
  Baseline teacher na dev zgodny z historycznymi checkpointami.
- Pełne dane: `analysis/{analysis.json,metrics.csv,pairs.csv,TABLES.md}`;
  per seed: `trace.json`, `selection.json`, `summary.json`, diagnostyki.
  Manifest i `source/` przechowują kod, protokół i hashe.

## Wynik: CIFAR-100, nominalnie50% parametrów

Średnia±SD po4 seedach; wszystkie metody ocenione nową wspólną procedurą,
nie kopiowane z historycznej tabeli E015.

| Metoda | Dev top1 po BN [%] | Dev NLL | KL na search B | KL na holdout C |
|---|---:|---:|---:|---:|
| UNIFORM | 50,06±1,60 | 2,2724 | 1,4208 | 1,4258 |
| TILT+0.30 | 51,47±1,01 | 2,3359 | 1,5019 | 1,4750 |
| SEARCH | 51,30±0,63 | **2,1904** | **1,2456** | **1,2677** |

| Seed | SEARCH−TILT [pp] | SEARCH−UNIFORM [pp] | KL_C(TILT)−KL_C(SEARCH) |
|---|---:|---:|---:|
| 42 | −0,64 | +0,86 | +0,2064 |
| 43 | +0,66 | +1,40 | +0,1559 |
| 44 | −0,64 | +0,18 | +0,2299 |
| 45 | −0,08 | +2,50 | +0,2370 |
| Średnia±SD | **−0,175±0,616** | **+1,235±0,980** | **+0,2073±0,0366** |

### Zamrożone kryteria

| Warunek | Wynik |
|---|---|
| SEARCH−TILT≥+1pp i≥3/4 dodatnich seedów | **NIE:** −0,175pp, 1/4 |
| SEARCH−UNIFORM średnio>0 | TAK: +1,235pp, 4/4 |
| KL na C niższe od TILT na≥3/4 | TAK: 4/4 |
| Wszystkie warunki jednocześnie | **NIE — pilotaż negatywny względem primary** |

Niższe NLL i KL są wynikami wtórnymi, nie zmieniają werdyktu primary.

## Budżety i wybrane profile

Teacher ma11 220 132 parametrów; UNIFORM5 581 542, rzeczywista redukcja
około50,25%. TILT5 580 606. SEARCH per seed:
5 581 212 / 5 581 000 / 5 580 744 / 5 581 147.
Wszystkie w granicach0,1% wspólnego targetu i bez jego przekroczenia.
Największa odchyłka w trzech metodach to936 parametrów (około0,0168%).

We wszystkich seedach start wybrany na B był **UNIFORM**, bo miał
niższe KL od TILT. SEARCH wybrał różne, niemonotoniczne profile;
nie sprowadził się do dopasowania jednego przesunięcia TILT.
Przykład seed42, kolejność grup w `selection.json`:

- UNIFORM: `[45,45,45,90,90,90,180,180,180,361,361,361]`.
- TILT: `[57,57,57,115,115,115,221,207,194,359,331,303]`.
- SEARCH: `[45,45,57,90,100,90,180,181,167,335,372,361]`.

To rzeczywiste lokalne wyszukiwanie alokacji pod ograniczeniem parametrów,
ale bez dowodu globalnej optymalności ani przewagi nad random search.

## Co się udało i gdzie jest problem?

1. **Optymalizacja funkcji celu działa poza próbkami search.** KL na C
   poprawia się na4/4 seedów wobec TILT i UNIFORM. Nie jest to wyłącznie
   poprawa na zbiorze B. Nie dowodzi jednak generalizacji do innych danych.
2. **Cel KL nie jest równoważny top1.** Już przed search UNIFORM ma lepsze
   KL niż TILT, mimo niższej trafności. SEARCH polepsza KL i NLL, ale nie
   uzyskuje przewagi top1 nad TILT. To konkretny problem zgodności celu
   optymalizacji z primary, nie dowód, że alokacji nie da się optymalizować.
3. **Kontrola BN jest niepokojąca:** teacher bez kompresji ma średnio
   **75,235%**, a po naszej korekcie BN na A bez augmentacji **69,405%**,
   czyli spadek **5,83pp**. Per seed:−5,58/−5,62/−5,78/−6,34pp.
   To zmiana samych statystyk BN, przy niezmienionych wagach. Nie zakładamy,
   że jej jedyną przyczyną jest rozmiar A; wpływ resetu, cumulative average
   i braku augmentacji nie został rozdzielony.
4. Wszystkie ramiona używają tej samej BN, więc porównanie wewnętrzne jest
   kontrolowane. **Nie oznacza to, że protokół BN jest dobry**, ani że wyniki
   można zestawiać1:1 z wcześniejszym pełnym train/augmentacją E015.
5. Primary jest negatywny. Cztery seedy na jednym zbiorze/budżecie i50 ocen
   na checkpoint to ograniczony pilotaż. Nie deklarujemy nowej najlepszej metody.

## Decyzja

Pozostajemy przy luce **optymalizacji budżetu per warstwa**, lecz nie
skalujemy obecnego search na kolejne zbiory. **Najpierw osobno zweryfikować
korektę BN** na nieskompresowanym teacher i stałych profilach, z protokołem
zamrożonym przed pomiarami. Dopiero po tej kontroli podejmować decyzję
o dalszej funkcji celu i wyszukiwaniu. Nie zmieniać kryteriów E024 po wyniku.
Żaden kolejny eksperyment ani trening nie został uruchomiony.
