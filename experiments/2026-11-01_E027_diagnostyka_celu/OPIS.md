# 2026-11-01_E027_diagnostyka_celu

Status: **frozen**, 2026-11-01, przed wyszukiwaniem i oceną dev.

## Pytanie
Czy ograniczeniem optymalizacji alokacji jest **funkcja celu**, czy **przestrzeń
i budżet wyszukiwania**? E026 dał bezpośredni sygnał: TILT ma gorsze KL od
UNIFORM na4/4 seedów, a lepszą trafność na4/4. Optymalizacja KL może więc
wskazywać w przeciwną stronę niż top1.

Nadal adresujemy otwarty problem budżetu kompresji per warstwa w foldingu.
E027 jest diagnostyką kryterium, nie miary grupowania i nie nowym treningiem.

## Wszystko, co pozostaje stałe względem E026
CIFAR-100, checkpointy E01542–45, globalny budżet≈50%, target=UNIFORM
5 581 542. Te same12 grup, operator E024, PCA/HKMeans, solver, pary donor→receiver,
RNG20261031+runda, start z lepszego UNIFORM/TILT,2 rundy po24, limit50 ocen/seed,
cache i stop przy poprawie≤1e-6 **funkcji wybranej w tym eksperymencie**.
BN: A4096 z crop/flip, CMA reset,32×batch128, hash widoków E025.
A/B/C/dev bez zmian i rozłączne; test/rezerwa nietknięte.

## Nowe: wiele kryteriów oceny tej samej puli kandydatów
W każdej rundzie, dla każdego kandydata i seed, na B2048 (bez etykiet) liczymy:
1. `kl`: mean KL(teacher‖candidate), jak E024/E026;
2. `disagreement`: frakcja próbek, gdzie argmax(candidate)≠argmax(teacher);
3. `teacher_ce`: mean cross-entropy candidate względem **twardej** etykiety
   argmax(teacher).
Wszystkie trzy z tego samego forwardu; brak dodatkowego kosztu modelu.

**Primary, zamrożone przed runem: wybór minimalizujący `disagreement`.**
Uzasadnienie: ta miara jest bezpośrednio zgodna z top1 teacher, w odróżnieniu
od KL. `kl` i `teacher_ce` służą jako porównanie na **identycznej puli**.
Nie wybieramy zwycięzcy po dev i nie stroimy progu.

## Diagnostyka po zamrożeniu wyboru
Po zapisaniu i zahashowaniu `selection.json` oceniamy **każdego z≤50
kandydatów** na dev top1/NLL. Dev nie wpływa na search ani wybór; służy tylko
do analizy post hoc:
- `selection regret`: dev top1 wybranej alokacji minus maksimum w puli,
  osobno dla wyboru po disagreement, KL i teacher_ce;
- **oracle pułapu**: maksymalny dev top1 w puli (górna granica selektora
  działającego na tym dev, nie metoda i nie wynik do publikacji);
- Spearman w obrębie seeda między każdą miarą B a dev top1 kandydatów (n=50).
Pula zależy od wybranej funkcji, bo search jest adaptacyjny; dlatego
porównania trzech selektorów wykonujemy **na tej samej, faktycznie odwiedzonej
puli**, a nie na różnych przebiegach.

## Kryteria (zamrożone)
- **Primary:** wybór po `disagreement` daje NEW−TILT≥+1pp średnio i≥3/4
  dodatnich seedów. TILT/UNIFORM/stary SEARCH z poprawioną BN: reuse E025/E026.
- **Diagnostyka celu:** średni Spearman(ρ) z dev top1 dla `disagreement`
  większy niż dla `kl` i dla `teacher_ce` (po4 seedach).
- **Diagnostyka przestrzeni:** oracle pułapu dev top1 w odwiedzonej puli
  minus TILT≥+1pp średnio. Jeśli TAK, a primary NIE → ograniczeniem jest
  selektor/cel, nie brak lepszych alokacji w puli. Jeśli NIE → wąska jest
  pula/propozycje, a nie tylko cel.
- Raportujemy mean±SD, wszystkie4 seedy, koszt i pełne trace. Wynik negatywny
  zachowujemy. Nie zmieniamy limitu, progu, przestrzeni ani funkcji po dev.

## Kontrole i wykonanie
CPU: formuły trzech miar na syntetycznych logitach, zgodność KL kierunku,
disagreement=1-agreement, teacher_ce przy zerowej dywergencji, hash widoków BN,
deterministyczność propozycji, brak etykiet B w measure. Hashe kodu/protokołu,
checkpointów, splitów i referencji. 4GPU0–3, seed42+i/GPUi, max200 ocen,
jeden launcher, limit120min, artefakty≤0,5GiB, Conda attention.
Zapis trace i per-kandydat dev dopiero po wyborze. Bez treningu, nowych danych,
checkpointów i konkurencyjnych zadań.

## Ograniczenia
Jeden zbiór/budżet,4checkpointy; dev i C już wielokrotnie używane, więc oracle
jest optymistyczny i nie stanowi metody. Wybór `disagreement` jest motywowany
wynikami E026, ale kryterium i próg zamrożono przed tym runem. Wnioski dotyczą
tej heurystyki, przestrzeni i kosztu; nie dowodzą optymalności alokacji ani
przewagi nad innymi optymalizatorami.

## Odtworzenie
Z katalogu projektu, Conda attention:
```
CUDA_VISIBLE_DEVICES='' python -B experiments/2026-11-01_E027_diagnostyka_celu/check.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B experiments/2026-11-01_E027_diagnostyka_celu/run.py
CUDA_VISIBLE_DEVICES='' python -B experiments/2026-11-01_E027_diagnostyka_celu/analyze.py --run-dir runs/E027/<run-ID>
```
