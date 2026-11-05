# 2026-11-05_E033_interakcja_pozne_grupy

Status: **frozen przed runem**, 2026-11-05 UTC.

## Pytanie i hipoteza

E031 dał słaby i niestabilny zysk TILT na CIFAR-100, a E032 pokazał, że
po korekcie BN późne grupy (`layer4`) są tam wyraźnie bardziej wrażliwe niż
wczesne. TILT właśnie późne grupy kompresuje mocniej. **Hipoteza E033:**
znaczna część kosztu CIFAR-100 pochodzi z dodatkowej kompresji późnych grup,
a nie z samej ochrony wczesnych.

E033 testuje tę hipotezę dwoma zamrożonymi profilami hybrydowymi przy
**tym samym globalnym budżecie parametrów**. To ablacja diagnostyczna,
nie search, nie nowy optymizator i nie nowa metoda.

## Zamrożony zakres

- CIFAR-10, CIFAR-100 i Tiny64, kompresja30% i70% (**pozostaje70/30%
  parametrów**), seedy42–45, istniejące checkpointy E011/E015/E019.
- Profile `U` i `T` są **dokładnie** profilami E031 z
  `runs/E031/20261104T212951Z_6c3d7356` (te same szerokości i target).
- `EP` (early-protect): grupy9–11 (`layer4`) mają szerokości UNIFORM;
  grupy0–8 dostają kształt TILT+0.30, a ich `base` jest dobierany do
  wspólnego targetu. `LC` (late-compress): grupy0–8 mają szerokości
  UNIFORM; grupy9–11 dostają kształt TILT+0.30 i `base` dobierany do
  tego samego targetu. W obu przypadkach po solverze uzupełniamy tylko
  grupy nietrwałe (`fill`), nie zmieniamy grup odniesienia.
- Wymagania:12grup, dokładne szerokości w `check.json` przed runem,
  każdy profil≤target, różnica≤0,1%, brak pustych klastrów.
  `EP`/`LC` to **nasze** profile na niezmienionym operatorze upstream;
  nie są wariantami opublikowanymi przez autorów.
- Operator: `osaukh/folding_as_projection`, commit
  `694691556e8314a835178a741938dd3d4d80c171`, klon E029 niezmieniony.
  CIFAR: klasa autorów ze strict load naszych wag. Tiny64: operator autorów
  na naszym torchvision ResNet18 (upstream nie ma klasy Tiny64), jak E030.
  TILT/hybrydy to nasz adapter `keep_ratio` per grupa.
- Dane/BN: A4096 crop/flip jak E031, dev5000/5000/10000, CMA/reset oraz
  niezmienione `repair_bn` na tym samym A. Teacher PRE/CMA/repair_bn.
  Oficjalny test i rezerwa nietknięte; bez nowych danych i treningu.

## Ramiona i koszt

Na każdy zbiór×budżet×seed:
1. `UP_U` — UNIFORM na operatorze autorów.
2. `UP_T` — TILT+0.30 na operatorze autorów.
3. `UP_EP` — profil EP na operatorze autorów.
4. `UP_LC` — profil LC na operatorze autorów.
5. `LEGACY_EP` — referencyjny operator E012/E014 z szerokościami EP.
6. `LEGACY_LC` — referencyjny operator E012/E014 z szerokościami LC.

Każde ramię w `PRE`, `CMA`, `UP_REPAIR`:6×3×4×3×2=**432 oceny**
+36 kontroli teacher. `LEGACY` dla EP/LC jest kontrolą operatora na nowych
profilach; U/T nie są powtarzane w LEGACY, bo E031 już je sprawdził.
Bez gradientów i zmian wag przez BN.

## Kryteria zamrożone

1. **Operator EP/LC:** `LEGACY_EP` vs `UP_EP` i `LEGACY_LC` vs `UP_LC`
   po CMA i `UP_REPAIR`: średnia |Δtop1|≤0,5pp i maksimum≤1pp po4 seedach,
   dla każdego zbioru×budżetu. Razem24 bramki.
2. **Mechanizm, primary CIFAR-100/b70/CMA:** `EP−T` średnia>0 i≥3/4
   dodatnich. To bezpośredni test hipotezy: usunięcie dodatkowej kompresji
   późnych grup ma pomóc. `LC−T` jest **eksploracyjne** i raportowane bez
   progu, ponieważ przy dopasowanym budżecie region nietknięty absorbuje
   różnicę parametrów i LC nie jest czystym „tylko późna kompresja”.
3. **Mechanizm, secondary:** ten sam kierunek EP−T dla `UP_REPAIR` na
   CIFAR-100/b70 oraz raport wszystkich zbiorów/budżetów/BN. Nie wybieramy
   korzystnej procedury BN.
4. **Integralność:** strict load,12grup, dokładne parametry, niepuste
   klastry, skończone logity,4GPU/exit0, limity90min/0,5GiB, brak
   testu/rezerwy, hashe kodu/protokołu/danych/checkpointów.

Jeśli primary przejdzie, mówimy o **wsparciu hipotezy konfliktu późnych
grup** w tym zakresie. Jeśli nie, hipoteza nie jest potwierdzona; nie
zmieniamy progów. Jeśli EP poprawi także CIFAR-10/Tiny, raportujemy to
jako sygnał hipotezy-generujący, nie jako nową metodę. Nie wracamy do
search, AUTO50 ani miar grupowania na podstawie tego wyniku.

## Ograniczenia

Profile hybrydowe zmieniają kilka grup równocześnie; nie izolują
pojedynczej grupy ani pojedynczego kanału. Dopasowany budżet wymusza
przesunięcia w grupach nietrwałych, więc EP/LC nie są czystymi
„jednoczynnikowymi” interwencjami. Dev był używany wcześniej, są4seedy,
jedna rodzina architektury, a Tiny ma inny stem/rozdzielczość. Brak
pełnej reprodukcji paperu1:1 i niezależnego testu. Wynik nie jest
dowodem optymalności ani uniwersalności alokacji.

## Wykonanie

Conda attention, GPU0–3: seed42+i/GPUi, trzy zbiory i dwa budżety
sekwencyjnie na workerze; folding CPU, BN/ocena GPU, batch128/GPU.
Jeden launcher, konkretne PID, limit90min. Każdy run:
`runs/E033/<UTC>_<token>/` z manifestem, hashami, `source/`, logami.
Bez treningu,96×96, publicznych wag, testu i rezerwy.

```
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -B experiments/2026-11-05_E033_interakcja_pozne_grupy/experiment.py check
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -B experiments/2026-11-05_E033_interakcja_pozne_grupy/experiment.py run
CUDA_VISIBLE_DEVICES='' python -B experiments/2026-11-05_E033_interakcja_pozne_grupy/experiment.py analyze --run-dir runs/E033/<ID>
```
