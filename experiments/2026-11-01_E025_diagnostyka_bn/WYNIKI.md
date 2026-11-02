# E025 — diagnostyka korekty BN przed optymalizacją alokacji

Status: **completed**, 2026-11-01. **Brak augmentacji przy przeliczaniu BN
wyjaśnia prawie cały spadek teacher w E024 w badanym kontrastcie.** Samo
zwiększenie liczby obrazów nie rozwiązuje problemu. Dodatkowo krótka EMA
po resecie statystyk powoduje osobne załamanie jakości.

Cel pozostaje niezmieniony: **optymalizacja budżetu foldingu per warstwa**.
E025 usuwa problem procedury oceny kandydatów; nie proponuje nowej miary
grupowania ani nie optymalizuje nowych alokacji.

## Co wykonano

- Zamrożony [OPIS.md](OPIS.md): 2 liczebności ×2 augmentacje ×3 procedury
  BN ×4 modele ×4 seedy. CIFAR-100, istniejące checkpointy E01542–45.
- Modele: teacher, UNIFORM, TILT i wybrany wcześniej SEARCH z E024.
  Szerokości, parametry i operator stałe; **brak ponownego search**.
- Run: `runs/E025/20261101T121011Z_9c3d4231`.
- **176/176 nowych ocen**,16 ponownie użytych komórek E024, łącznie192.
  Dodatkowo16 referencji bez korekty BN z E024. Cztery workery po44 nowe
  oceny, GPU0–3/seedy42–45, wszystkie exit0.
- Wall-time **170,41s (2,84min)**, artefakty około3,7MiB. Cztery RTX2080Ti,
  Conda attention, batch128/GPU/przebieg. Wersje w manifest/pip-freeze.
- Kalibracja tylko na train: A4096 lub L35904=train bez B/C. A jest
  prefiksem L. B/C/dev/rezerwa poza kalibracją. Oficjalny test nietknięty.
- Augmentacja: jeden stały crop32/padding4 i flip/obraz, **przed**
  normalizacją; te same obrazy/widoki/kolejność między procedurami,
  modelami i seedami. To te same rodzaje transformacji co w treningu
  checkpointów, nie odtworzenie historycznych losowań augmentacji.
- Parametry niezmienione, brak gradientów; tylko bufory BN aktualizowane.
  Hashe wejść/kodu/protokołu, zgodność etykiet klastrów z E024, liczniki,
  liczby próbek, widoki i kompletność zweryfikowane.
- Analiza: `analysis_20261101T121334Z/` (`analysis.json`, `metrics.csv`,
  `summary.csv`, `effects.csv`, `TABLES.md`). Audyt niespodziewanego efektu
  EMA **post hoc**: `ema_audit/{audit.json,AUDYT.md}`.

## 1. Który czynnik powodował spadek teacher?

Teacher bez korekty BN: **75,235%**. E024 (A/plain/CMA_reset):69,405%.
Primary kontrasty różnią się od E024 **jednym czynnikiem**:

| Zmiana | Teacher top1 [%] | Różnica vs E024 [pp], mean±SD | Dodatnie seedy | Próg praktyczny |
|---|---:|---:|---:|---|
| Więcej obrazów:4096→35904, bez augmentacji, CMA | 69,540 | +0,135±0,118 | 4/4 | NIE |
| Crop/flip,4096 obrazów, CMA | **75,140** | **+5,735±0,350** | **4/4** | **TAK** |
| CMA→EMA0.1 po resecie,4096, bez augmentacji | 1,000 | −68,405±0,118 | 0/4 | NIE |

Próg: poprawa≥1pp i≥3/4 dodatnich seedów. Tylko augmentacja spełnia go
wśród tych trzech primary kontrastów. Przy tej samej liczbie próbek i
tym samym estymatorze odzyskuje **około98,4%** wcześniejszego spadku5,83pp.

**Wniosek:** w tym kontrolowanym porównaniu spadek nie wynika głównie
z niedostatecznej liczby próbek. Znaczenie ma rozkład wejść użytych do
wyznaczania statystyk BN. Crop i flip badano razem, nie rozdzielono ich
indywidualnych efektów; nie jest to uniwersalna reguła dla innych modeli.

## 2. Wszystkie warianty — średnia po4 seedach

A=4096 obrazów; L=35904. CMA=średnia po batchach; EMA momentum0.1.
Reset: mean0/var1/licznik0. Retain: statystyki źródła/foldingu zachowane.
Wszystkie wartości poniżej to dev top1[%] po BN.

| Dane | Procedura | Teacher | UNIFORM | TILT | SEARCH |
|---|---|---:|---:|---:|---:|
| A plain | CMA reset (E024 reuse) | 69,405 | 50,060 | 51,470 | 51,295 |
| A plain | EMA reset | 1,000 | 1,000 | 1,000 | 1,000 |
| A plain | EMA retain | 70,305 | 51,485 | 52,410 | 52,740 |
| A aug | CMA reset | **75,140** | 53,855 | 55,550 | 54,715 |
| A aug | EMA reset | 1,000 | 1,000 | 1,000 | 1,000 |
| A aug | EMA retain | 75,115 | 54,510 | 55,970 | 55,635 |
| L plain | CMA reset | 69,540 | 50,020 | 51,410 | 51,370 |
| L plain | EMA reset | 69,075 | 49,390 | 50,950 | 50,950 |
| L plain | EMA retain | 69,075 | 49,390 | 50,950 | 50,950 |
| L aug | CMA reset | 75,080 | 53,810 | 55,415 | 54,580 |
| L aug | EMA reset | 74,870 | 53,545 | 55,340 | 54,400 |
| L aug | EMA retain | 74,870 | 53,545 | 55,340 | 54,400 |

Kryterium odtworzenia teacher (średnia strata≤1pp, każdy seed≤1,5pp)
spełnia **pięć** komórek: A/aug/CMA_reset, A/aug/EMA_retain oraz wszystkie
trzy L/aug. Żadna bez augmentacji. Nie wybieramy zwycięzcy na podstawie
najwyższego dev w tej tabeli.

## 3. Dlaczego krótka EMA po resecie daje1%?

Po32 batchach EMA0.1 zachowuje **0,9^32=0,03434** wpływu inicjalizacji.
Reset wariancji do1 pozostawia więc składnik około0,0343. To nie jest
mało względem rzeczywistych wariancji niektórych warstw, rzędu0,0003.

Audyt zapisanych statystyk (bez nowej oceny GPU) sprawdził zależność:

`var_reset(N) − var_retain(N) = 0.9^N · (1 − var_źródła)`.

- **1280 porównań warstw**, maksymalny błąd równania **4,26×10⁻⁸**.
- Teacher seed42, `layer4.1.bn2`, A/plain:
  źródłowa średnia wariancja0,000326; po EMA reset0,034593;
  po EMA retain0,000268 — **129,3×** większa po resecie.
- Dla tej warstwy na czterech seedach iloraz wynosi122–132×.
- Przy281 batchach w L waga inicjalizacji to1,39×10⁻¹³; reset i retain
  dają te same metryki. **Nie oznacza to zrównania EMA z CMA** — EMA nadal
  waży przede wszystkim końcowe batche.

To potwierdza silne zanieczyszczenie wariancji przez inicjalizację w
krótkiej EMA i jest zgodne z załamaniem jakości. Nie wykonano dodatkowej
interwencji wyłącznie na wariancji, więc nie rozdzielamy całkowicie udziału
błędu mean i var. **EMA0.1 po resecie,32 batche nie nadaje się tutaj do oceny
alokacji**, mimo poprawnej implementacji wzoru.

## 4. Co dzieje się z porównaniem alokacji po naprawie BN?

Najprostszy kontrast do E024: zmieniamy wyłącznie plain→aug, pozostawiając
A4096/CMA_reset. Nie wybieramy najlepszej z12 komórek po dev.

| Model | E024 top1 | A/aug/CMA top1 mean±SD | Poprawa [pp] |
|---|---:|---:|---:|
| Teacher | 69,405 | 75,140±0,400 | +5,735 |
| UNIFORM | 50,060 | 53,855±1,509 | +3,795 |
| TILT | 51,470 | 55,550±1,105 | +4,080 |
| SEARCH (profil ustalony w E024) | 51,295 | 54,715±0,879 | +3,420 |

W tym wariancie:
- SEARCH−UNIFORM **+0,860±1,356pp**,3/4 dodatnich seedów.
- SEARCH−TILT **−0,835±1,474pp**,1/4 dodatnich seedów.
- KL na C: UNIFORM1,2011; TILT1,2503; SEARCH**1,0639**.
- Dev NLL: UNIFORM2,1501; TILT2,2198; SEARCH**2,0843**.

**Naprawienie BN nie zmienia starego SEARCH w zwycięzcę top1.**
Rozbieżność celu KL i trafności nadal występuje dla tych stałych profili.
Nie jest to test nowego search pod poprawioną BN: optimum alokacji może
być inne, a profil E024 został wybrany pod procedurą bez augmentacji.
E024 pozostaje negatywny względem swojego zamrożonego kryterium.

## 5. Decyzja dla dalszej optymalizacji budżetu

Zidentyfikowaliśmy konkretną poprawkę infrastruktury oceny: **do BN używać
widoków crop/flip zgodnych rodzajem z treningiem**, z nieetykietowanej,
wydzielonej części train. Dla następnego protokołu sensownym kandydatem jest
**A4096+aug+CMA_reset**: minimalna zmiana wobec E024, odtwarza teacher,
nie zależy od początkowych statystyk foldingu i nie wymaga większego A.
To rekomendacja oparta na diagnostyce dev, nie niezależnie zatwierdzony standard.

**Następny krok do osobnego zamrożenia:** sprawdzić wyszukiwanie alokacji
pod tą poprawioną procedurą, utrzymując ten sam cel KL i limit ocen, aby
odizolować wpływ BN. Dopiero potem rozstrzygać, czy trzeba zmienić cel.
Nie zwiększać równocześnie przestrzeni szukania i liczby prób. Żaden nowy
search ani trening nie został uruchomiony w E025.

## Ograniczenia
Jedna rodzina architektury, jeden zbiór/budżet,4 seedy, C/dev wcześniej
używane. Kalibracja deterministyczna, jeden widok/obraz i jeden seed
augmentacji, bez badania zmienności losowań. CMA jest średnią po batchach;
L ma ostatni batch64, A same128. EMA retain startuje od różnych statystyk
w teacher i modelach folded. Nie wyciągamy wniosku o optymalnym BN ogólnie
ani o optymalności alokacji. Nie użyto testu, rezerwy, nowych danych ani treningu.
