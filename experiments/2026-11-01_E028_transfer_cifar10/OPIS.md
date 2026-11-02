# 2026-11-01_E028_transfer_cifar10

Status: **frozen**, 2026-11-01, przed uruchomieniem.

## Cel
Test **przeniesienia** reguły wybranej w E027 na zbiór, którego nie użyto do
projektowania kryterium. Pytanie: czy bezetykietowy wybór alokacji po
`disagreement` (argmax≠teacher) bije TILT i UNIFORM również na CIFAR-10
przy dopasowanym budżecie? Adresujemy otwarty problem budżetu per warstwa
w foldingu; nie zmieniamy metody po wyniku E027.

E027 na CIFAR-100 dał +0,935pp vs TILT (4/4), poniżej progu+1,0pp.
Reguła, cel, operator, BN i koszt pozostają **identyczne**. Zmienia się
wyłącznie zbiór/checkpointy/klasy. To test transferu, nie strojenie.

## Stałe elementy metody
Samodzielny operator E024 i reguła E027, użyte bez zmian:
12 grup residualnych, PCA/HKMeans, solver, pary donor→receiver,
RNG20261031+runda, start z lepszego UNIFORM/TILT,2 rundy po24,
limit50 ocen/seed, cache, stop≤1e-6.
Cel: minimalizacja `disagreement` na B2048 (bez etykiet).
BN: A4096, crop32/padding4+flip (RNG20261101+indeks), normalizacja CIFAR-10,
CMA reset,32×batch128, bez gradientów.
Budżet: globalna redukcja≈50%, target=faktyczny UNIFORM, każdy profil
≤target i≥99,9% targetu, szerokości w[10%,90%] kanałów grupy.

## Dane i architektura
CIFAR-10, checkpointy E01142–45 (ResNet18: conv1 3×3 stride1, maxpool
Identity, fc512→10;11 173 962 parametrów). Lokalne batche train, split E011
train45k/dev5k. Nowa permutacja RNG20261102 na train: A4096/B2048/C2048,
dev5000 z pliku. Zbiory rozłączne; A do BN, B do search, C do audytu KL
po wyborze. Etykiety wyłącznie w końcowej ocenie dev.
Oficjalny test CIFAR-10 i żadna rezerwa nie są używane.

## Pomiar i rozdzielenie dev
Dla każdego z≤50 kandydatów: fold, BN na A, trzy miary na B
(`disagreement`, `kl`, `teacher_ce`). Wybór wyłącznie po `disagreement`.
Po zapisaniu i hashu `selection.json` oceniamy dev **każdego** kandydata
(tylko diagnostyka selektora/pułapu) oraz finalnie teacher, teacher+BN,
UNIFORM, TILT, NEW_SEARCH: dev top1/NLL i KL na C.
Dev nie uczestniczy w search ani wyborze; jest używany jak w E027.
UNIFORM/TILT są liczone **od nowa** pod tą samą procedurą BN — nie kopiujemy
historycznych tabel E014, bo miały inną kalibrację.

## Kryteria zamrożone
- **Primary:** NEW_SEARCH−TILT dev top1 średnio≥+1,0pp i≥3/4 dodatnich seedów.
- **Secondary:** NEW−UNIFORM średnio>0.
- Diagnostyka: alignment(disagreement vs dev top1) najwyższy z trzech miar;
  oracle pułapu puli−TILT≥+1pp (górna granica selektora, nie metoda).
- Raport mean±SD po4 seedach, wszystkie pary, parametry, oceny, czas.
- Wynik negatywny zachowany. Nie zmieniamy progu, limitu ani funkcji po dev.

## Kontrole i wykonanie
CPU: hashe danych/splitów/checkpointów/E027, identity fold, liczba parametrów,
forward fc10, determinizm augmentacji, rozłączność A/B/C/dev, replay50
propozycji z atrapą miary. GPU: 4×GPU0–3, seed42+i/GPUi, max200 ocen,
jeden launcher, limit120min, artefakty≤0,5GiB, Conda attention.
Bez treningu, nowych danych, checkpointów wyjściowych i konkurencyjnych zadań.
Test/rezerwa nietknięte.

## Ograniczenia
Jeden budżet/architektura. CIFAR-10 dev był używany wcześniej w projekcie,
lecz nie do projektowania tej reguły; przeniesienie jest więc częściowe.
Oracle na dev jest optymistyczny. Wynik dotyczy tej heurystyki i kosztu.

## Odtworzenie
Z katalogu projektu, Conda attention:
```
CUDA_VISIBLE_DEVICES='' python -B experiments/2026-11-01_E028_transfer_cifar10/check.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B experiments/2026-11-01_E028_transfer_cifar10/run.py
CUDA_VISIBLE_DEVICES='' python -B experiments/2026-11-01_E028_transfer_cifar10/analyze.py --run-dir runs/E028/<run-ID>
```
