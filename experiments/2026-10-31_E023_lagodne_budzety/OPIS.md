# 2026-10-31_E023_lagodne_budzety

Status: **frozen**, 2026-10-31, przed oceną nowych modeli na dev.

## Pytanie
Czy nierówna alokacja kompresji między grupy foldingu poprawia jakość przy
łagodnej globalnej redukcji parametrów10/20%? Czy pozostaje użyteczna
bezwzględnie? To kontynuacja luki alokacji, nie nowa miara grupowania.

## Jawne rozszerzenie profilu
E014 stosuje clip[0.1,0.9] i base>=0. Minimalna redukcja UNIFORM jest
około19%, TILT+.30 około37%: nie można uczciwie użyć go dla10/20%.
E023 ma osobny solver, bez zmiany E014: r_g=clip(base+t*(2g/11-1),0,0.9),
t=0 dla UNIFORM, t=0.30 dla **TILT0+0.30** (0 oznacza nową dolną granicę).
base szukany bisekcją w[-abs(t),0.9+abs(t)], liczba parametrów liczona
dokładnie z dyskretnych szerokości E014. Nie stroimy t ani innych hiperparametrów.
UNIFORM: największa liczba parametrów pod nominalnym targetem osiągalna
na ścieżce jednolitego ratio. Wspólny target TILT0 = faktyczny UNIFORM.
TILT0: solver i repair_budget(min_ratio=0,max_ratio=.9), nigdy ponad target,
różnica liczby parametrów≤0.1% targetu. UNIFORM nominalna odchyłka≤0.5% targetu.
Raportować rzeczywistą redukcję, wszystkie ratios/widths, liczbę grup
nietkniętych oraz maksymalną zmianę ratio przez repair. Profil może skupiać
kompresję w późnych grupach. Nie łączyć go bez oznaczenia z TILT E014.

## Dane, modele, wykonanie
3 zbiory (CIFAR-10/CIFAR-100/Tiny64), seedy42–45, checkpointy E011/E015/E019
z zamrożonego inventory E020. 3×4×2 budżety×2 metody=48 ocen.
Wyłącznie istniejący train do korekty BN i dev do oceny; bez testu/rezerwy,
nowych danych, nowych wag, gradientów, fine-tuningu i96×96.
Operator E012/E014 bez zmian, preserve_fc_bias=True, fold_seed42,
BN seed20261024, batch128/GPU, pełny train. Cztery workery GPU0–3,
po jednym seedzie na GPU, każdy wykonuje12 zadań. Brak konkurencyjnego
launchera. Conda attention, limit90min i nowych artefaktów0.5GiB.
Zapis summary/diagnostics bez checkpointów/logitów. SHA256 kodu,
protokołu, inventory, checkpointów, splitów; wersje środowiska i logi.
Preflight CPU sprawdza rozwiązania i ich parametry bez oceny dev.
Pierwsze zadania pełnego biegu stanowią kontrolę wykonania; przy błędzie
zatrzymać i opisać, nie zmieniać operatora ani powtarzać udanych zadań.

## Kryteria ustalone przed wynikami
Primary: TILT0−UNIFORM post-BN top1 [pp] przy20%, parowanie po seedzie.
Wsparcie na zbiorze: średnia≥+0.5pp i≥3/4 różnic dodatnich.
Wsparcie wspólne:≥2/3 zbiorów spełnia warunek, żaden nie ma średniej<−0.5pp.
Przy10% wynik wtórny, bez zmiany decyzji primary.
Użyteczność opisowa: średni spadek względem baseline≤1pp i≤2pp, osobno
na każdym zbiorze/budżecie; nie utożsamiać przewagi względnej z użytecznością.
Raportować mean±SD(n=4), wszystkie pary, baseline/pre/post i NLL; strata
względem baseline oraz odzysk BN. Nie wybierać innej siły TILT po wynikach.
Integralność:48/48 completed, wszystkie4GPU exit0, dokładna liczba parametrów
po foldingu zgodna z solverem, hashe i baseline zgodne, BN params_unchanged.
Profil zależy od architektury/budżetu, nie od etykiet dev.

Ograniczenia: dev wielokrotnie używany w projekcie; seed42 selekcyjny
w dawnych doświadczeniach; jedna rodzina architektury, Tiny inny stem;
brak formalnego dowodu optymalności. Oceniamy ustaloną heurystykę alokacji.

## Odtworzenie
Z katalogu projektu, po `conda activate attention`:
```
CUDA_VISIBLE_DEVICES='' python -B experiments/2026-10-31_E023_lagodne_budzety/e023.py prepare
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B experiments/2026-10-31_E023_lagodne_budzety/e023.py run --prepared runs/E023/<prepare-ID>
CUDA_VISIBLE_DEVICES='' python -B experiments/2026-10-31_E023_lagodne_budzety/analyze.py --run-dir runs/E023/<run-ID>
```
Każdy run ma unikalny katalog. Nie uruchamiać ponownie po ukończeniu.
