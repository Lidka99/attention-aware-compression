# 2026-10-31_E024_optymalizacja_alokacji

Status: **frozen**, 2026-10-31, przed wyszukiwaniem i oceną kandydatów.

## Cel i luka
Adresujemy otwarty problem wskazany przez autorów foldingu: optymalizację
budżetu kompresji per warstwa. Zmienne to12 strukturalnie związanych grup
ResNet18 (wspólne szerokości przy dodawaniu residualnym), a nie wybór miary
grupowania. Metoda używa nieetykietowanego train: **calibration-based,
bez fine-tuningu**, nie calibration-free. Nie deklarujemy pierwszeństwa
w całej literaturze ani globalnej optymalności.

Nowa samodzielna implementacja: standardowe torch/torchvision/sklearn/hkmeans,
bez importu lub kopiowania implementacji wcześniejszych eksperymentów.
Korzystamy wyłącznie z istniejących checkpointów i danych jako wejść.
Operator ustalony: cechy wag grupy, PCA pełne, HKMeans(seed42,n_init1,
max_iter10,n_jobs1); wyjścia Conv sumowane, wejścia Conv/FC uśredniane,
parametry/statystyki BN uśredniane, bias FC zachowany. Grupy realizowane
kolejno, cechy z aktualnego stanu modelu. Brak przeszukiwania operatora.

## Pilotaż i dane
CIFAR-100, ResNet18 CIFAR stem, cztery istniejące checkpointy E01542–45.
Globalna redukcja parametrów nominalnie50%, target=faktyczny UNIFORM.
Każdy kandydat≤target i≥99.9% targetu. Szerokość każdej grupy między
ceil(0.1*n) a floor(0.9*n), czyli kompresja około10–90%.
Wspólne dla seedów rozłączne części wyłącznie historycznego train40k:
permutacja RNG20261031, A=pierwsze4096 do BN, B=kolejne2048 do search,
C=kolejne2048 do audytu bez wyboru. Nie używamy etykiet A/B/C.
Dev5000 tylko raz na końcu dla ustalonego zestawu modeli; test i rezerwa
nieoceniane. Normalizacja CIFAR-100 jak w checkpointach, bez augmentacji.
BN reset running_mean/var i cumulative average na A, batch128, ta sama
procedura dla wszystkich kandydatów i końcowych modeli. Bez gradientów.

## Funkcja celu i wyszukiwanie (ustalone)
Minimalizujemy średnie KL(p_teacher || p_folded) na B, temperatura1,
teacher=oryginalny checkpoint eval, candidate po BN na A.
UNIFORM i TILT+0.30 liczone nowym operatorem i tą samą korektą BN.
TILT: clip(base+.30*(2g/11-1),.1,.9), base rozwiązany na CPU dla targetu;
końcowe dyskretne uzupełnienie szerokości do budżetu. Te baseline'y nie
są kopiowane z historycznych wyników (nowa procedura BN).
Start SEARCH=lepszy UNIFORM/TILT według KL na B, nie dev.
Maksymalnie2 rundy po24 kandydatów (łącznie≤50 ocen/seed):
- losowo uporządkowane132 pary donor→receiver, RNG20261031+numer rundy;
- donor oddaje round(0.05*n_donor) kanałów, receiver otrzymuje maksymalną
  możliwą liczbę pod targetem; para musi rzeczywiście zwiększyć receiver;
- ewentualne wolne parametry uzupełniamy po jednym kanale, wybierając
  największy przyrost, który nie przekracza targetu (remis po indeksie);
- duplikaty pomijamy, bierzemy pierwsze24 wykonalne unikalne profile;
- wybieramy najlepszą pełną sieć z rundy; przy braku poprawy KL>1e-6 stop.
Nie używamy sum szkód pojedynczych grup. Nie optymalizujemy wag.
Wybrany profil zapisany i hashowany PRZED odczytem dev.
Na końcu audyt KL na C i dev top1/NLL dla teacher, teacher+BN,
UNIFORM, TILT i SEARCH (duplikaty oceniane raz). C nie zmienia wyboru.

## Kryteria przed wynikiem
Primary SEARCH−TILT dev top1: średnia≥+1pp i≥3/4 dodatnich seedów.
Ponadto SEARCH−UNIFORM średnia>0; KL na C mniejsze od TILT na≥3/4 seedów.
Wszystkie trzy warunki = udany pilotaż, nie dowód uniwersalności.
Raportować także KL na B/C, NLL, wszystkie seedy, mean±SD, liczbę ocen,
profile szerokości i parametry, czas/koszt. Negatywny wynik zachować,
nie stroić protokołu po dev. Wybór wg KL może nie poprawić trafności.

## Kontrole i zasoby
Przed runem: CPU identity fold, zgodność param-count z zbudowanym modelem,
forward residual/FC, skończone wyjścia, niezmienność źródła, test izolowanego
sumowania/uśredniania oraz rozłączność A/B/C/dev/rezerwy. Bez porównań
naukowych przed zamrożeniem protokołu.
4×GPU0–3, seed42+i na GPUi, jeden launcher, limit90min. Conda attention,
batch128/GPU i128/przebieg. Cztery seedy to cztery niezależne checkpointy.
Bez nowych danych, treningu,96×96 i zapisu checkpointów. Nowe wyniki≤0.5GiB.
Zapis kodu i hashy, środowiska, przydziałów, wszystkich ocen search i statusów.
Przy błędzie zachować run i wyjaśnić; żadnych automatycznych powtórek.

## Ograniczenia
Pilotaż na jednym zbiorze/budżecie. Dev używany wcześniej w projekcie.
Train był użyty do treningu teacher; C niezależne od search, nie od treningu.
Nowa implementacja nie jest zweryfikowanym oficjalnym baseline1:1.
Wspólna procedura zapewnia porównanie wewnętrzne, nie identyczność z E015.
Brak equal-budget random-search baseline w pilotażu; nie wywodzić przewagi
algorytmu wyszukiwania nad innymi optymalizatorami. Koszt search raportowany.

## Odtworzenie
W katalogu projektu, Conda attention:
```
CUDA_VISIBLE_DEVICES='' python -B experiments/2026-10-31_E024_optymalizacja_alokacji/check.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B experiments/2026-10-31_E024_optymalizacja_alokacji/run.py
CUDA_VISIBLE_DEVICES='' python -B experiments/2026-10-31_E024_optymalizacja_alokacji/analyze.py --run-dir runs/E024/<run-ID>
```
