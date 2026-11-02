# 2026-11-01_E026_alokacja_poprawione_bn

Status: **frozen**, 2026-11-01, przed nowym wyszukiwaniem i oceną dev.

## Cel
Adresujemy otwarty problem optymalizacji budżetu per warstwa w foldingu.
Sprawdzamy, czy ocena pełnych sieci z poprawioną BN pozwala znaleźć lepszą
alokację. **Jedyna zmiana naukowa względem E024: augmentacja A do BN.**
Nie zmieniamy celu KL, limitu, przestrzeni ani reguły wyboru kandydatów.
Metoda calibration-based, bez fine-tuningu; nie dowód globalnej optymalności.

## Stałe elementy
- CIFAR-100, istniejące checkpointy E01542–45, cztery GPU.
- Około50% redukcji parametrów, target=faktyczny UNIFORM:5 581 542.
  Każdy profil≤target i≥99,9% targetu. Te same12 grup residualnych.
- Samodzielny operator E024, niezmienione PCA/HKMeans i solver/proposals.
  Nie importujemy dawnych helperów E012–E020, nie zmieniamy starego kodu.
- Cel mean KL(teacher||candidate) na B2048, temperatura1, bez etykiet.
- Start=lepszy z UNIFORM i TILT według KL_B. Maksymalnie2 rundy po24
  nowych kandydatów, do50 ocen/seed. Donor oddaje round(.05*n) kanałów,
  receiver dostaje szerokość do budżetu, dyskretne fill. RNG par20261031+runda.
  Cache profili, stop przy poprawie≤1e-6. Stary SEARCH nie jest dodatkowym
  punktem startowym ani kandydatem dopisanym poza limitem.

## Poprawione BN i rozdzielenie danych
Te same indeksy A4096/B2048/C2048/dev5000 co w E024. A do BN, B do search,
C do audytu po wyborze. Test i rezerwa nietknięte. Etykiety tylko w ocenie dev.
A: dokładne widoki E025 crop32/padding4+flip, RNG20261101+indeks, następnie
normalizacja. Hash widoków identyczny z E025 dla każdego workera.
BN reset mean0/var1/licznik0, momentum=None (CMA),32×batch128. Tylko BN
w train, bez gradientów/zmiany parametrów. Każda ocena zaczyna od świeżego
foldingu checkpointu. B/C/dev bez augmentacji, teacher oryginalny eval.
Wybrany profil zapisany i zahashowany przed odczytem C/dev oraz wyników
referencyjnych do porównania. Te same deterministyczne ustawienia E024/E025.

## Ponowne użycie i ograniczenie obliczeń
Referencje UNIFORM/TILT/OLD_SEARCH oraz teacher+BN to istniejące
`runs/E025/20261101T121011Z_9c3d4231/seed_XX/items/*_A_aug_CMA_reset.json`.
Nie powtarzamy ich dev/KL_C. KL_B dla startowych UNIFORM/TILT nie był
zapisany w E025, więc jest nowym pomiarem wymaganym przez search.
Jeśli wybrana szerokość pokrywa się z jedną z tych referencji, finalne
dev/KL_C również bierzemy ze źródła. W innym przypadku jedna finalna
ocena wybranego modelu/seed (pre/post BN, KL_B jako kontrola, KL_C).
Nie oceniamy na dev wszystkich kandydatów i nie wybieramy wg dev.

## Kryteria zamrożone
Primary NEW_SEARCH−TILT post-BN top1: średnia≥+1pp i≥3/4 dodatnich seedów.
Dodatkowo NEW_SEARCH−UNIFORM średnia>0 i KL_C mniejsze niż TILT na≥3/4.
Wszystkie trzy = udany pilotaż; ten sam próg co w E024.
Wtórnie: NEW−OLD_SEARCH pod identycznym BN (top1,NLL,KL_C), liczba ocen,
profile, koszty i odsetek wybranych profili różniących się od E024.
Raport mean±SD po4 seedach i wszystkie pary. Wynik negatywny zachować;
nie zwiększać limitu ani nie stroić celu po dev.
Hipoteza: zmiana BN podczas wyboru może poprawić alokację, nie tylko
końcową ocenę. Jeśli KL się poprawi bez top1, przesłanka do osobnej analizy
zgodności celu; nie dowód, że żadna alokacja nie może pobić TILT.

## Kontrole i wykonanie
CPU: replay zapisanych50 ocen E024 na każdym seedzie, zgodność sekwencji
kandydatów i końcowego wyboru nowej funkcji sterującej; kontrola KL;
zgodność augmentacji, formuły BN i rozdzielenia danych. Hashe kodu,
protokołu, checkpointów, splitów, referencji, parametrów i widoków.
4GPU0–3, seed42+i/GPUi, max200 ocen search, jeden launcher, limit90min,
artefakty≤0,5GiB. Conda attention, batch128/GPU/przebieg. Bez treningu,
nowych datasetów,96×96, checkpointów wyjściowych, konkurencyjnych zadań.
Kody wyjścia/PID/statusy monitorowane; żadnych automatycznych restartów.

## Ograniczenia
Jeden zbiór/budżet; C/dev wielokrotnie użyte wcześniej. Wybór poprawki BN
oparto na E025, więc nie jest to niezależna walidacja. Te same4checkpointy,
jeden seed augmentacji. Nie porównujemy do equal-budget random search
ani oficjalnego kodu1:1. Wynik dotyczy tej heurystyki i funkcji celu.

## Odtworzenie
Z katalogu projektu, Conda attention:
```
CUDA_VISIBLE_DEVICES='' python -B experiments/2026-11-01_E026_alokacja_poprawione_bn/check.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B experiments/2026-11-01_E026_alokacja_poprawione_bn/run.py
CUDA_VISIBLE_DEVICES='' python -B experiments/2026-11-01_E026_alokacja_poprawione_bn/analyze.py --run-dir runs/E026/<run-ID>
```
