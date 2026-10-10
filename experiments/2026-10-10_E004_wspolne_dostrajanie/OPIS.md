# E004 — wspólne dostrajanie po kompresji

Data: 2026-10-10 UTC. Wersja protokołu: **0.1-draft**.
Status: **projekt; trening jeszcze nieuruchomiony**. Konfiguracja szkicu:
[config.draft.json](config.draft.json). Wersja wykonania (frozen) powstanie
po implementacji i kontrolach CPU. Status wykonania zapisze manifest biegu.

## Pytanie i cel

Jak zmieniają się wyniki BASE/PRE/POST/RANDOM, gdy **wszystkie cztery
modele** przejdą identyczne gradientowe dostrajanie z równym budżetem?
Czy po naprawie kolejność metod (PRE vs POST vs RANDOM) się utrzymuje,
czy różnice znikają? Czy równy budżet domyka lukę do modelu bazowego?

E003 zmierzyło sam skutek kompresji, bez naprawy: 55,24% POST, 54,94% PRE,
49,48% RANDOM wobec 66,84% BASE z niedowartościowanym wyjaśnieniem, że
alternatywą jest wspólny trening. E004 jest tym dodatkowym, kontrolowanym
krokiem naprawy. Nie zmienia selekcji, szerokości ani operatora z E003;
nie nadpisuje jego wyników.

## Budżet dostrajania — adaptacja protokołu z literatury

Użytkowniczka poprosiła o klasyczny budżet dla CIFAR-100. Punkt odniesienia:
**GReg — Neural Pruning via Growing Regularization** (Wang, Qin, Zhang, Fu,
ICLR 2021, arXiv:2012.09243), sekcja4 i dodatek A1: dostrajanie po
pruning na CIFAR przez **120 epok**: SGD (momentum0,9, weight decay5e-4),
**batch256**, LR multi-step (0:1e-2, 60:1e-3, 90:1e-4). Dla porównania
Network Slimming (Liu i in., ICCV 2017) używa160epok z batch64, a DSP
(AAAI 2023) 80epok; wybieramy jeden spójny, opublikowany zestaw.

Jawna adaptacja do tego projektu: EfficientNet-B0 (nie VGG/ResNet),
obrazy32×32 powiększone do96×96 jak E001, CIFAR-100, augmentacja
z treningu E001 (RandomCrop32/padding4, poziome odbicie, resize, ta sama
normalizacja), AMP FP16 z GradScaler jak E001. Nie kopiujemy literatury
bez tych zastrzeżeń i nie stroimy ustawień pod wynik.

## Wejścia i inicjalizacja

Modele z E003 (inference-only, FP32, bez optymalizatora) — cztery warianty
o identycznych szerokościach i tej samej strukturze:
`runs/E003/20261010T095706Z_5a2f233_seed42__0o_uckk/models/{BASE,PRE,POST,RANDOM}.pt`.
Utrwalone SHA-256 plików z model_files.json E003:

- BASE `44c44691acd501caffdbbe359ecc3892163cbd3cee6a50500b8abea6393a2817`
- PRE `8186ff8155b3f9fca8a79aeebb15b65cb7b0816abcb3b943255939a992011882`
- POST `9f4e94ddafdc6986a5bfcb6551e988b1dcc4a4e262047a00f51e23c95a9c2b8d`
- RANDOM `c4ba8b4fec2fa00cff76f87a80a9a74e4cfe95504a23119979b1c4182c89002e`

To wagi po operatorze E003; BASE to oryginalny splot sprzed kompresji.
Ładujemy je przez materializację grafu i `strict=True`, bez ponownego
foldingu — tak samo jak walidacja E003. Każdy wariant zaczyna od **świeżego**
optymalizatora i własnego stanu RNG; nie przenosimy optymalizatora E001.

Dane: dokładnie ten sam podział co E001 — 40000 train, 5000 dev; zero
oficjalnego testu i rezerwy. Preprocessing i augmentacja jak wyżej.
Dev był używany do wyboru checkpointu E001 i oceny E003; nie jest
niezależnym testem generalizacji.

## Przebieg i 4 GPU

Cztery warianty dostrajamy **kolejno**, po jednym na raz, każdy na
dokładnie4GPU przez DDP: rank r widzi swoją część train (DistributedSampler,
jak E001). Batch64/GPU = globalny256, bez akumulacji. SyncBatchNorm.
Workery2/proces, in_order, deterministyczne algorytmy, AMP FP16,
TF32wyłączone. Seedy identyczne dla każdego wariantu (42+rank, loader
42+rank+10000×epoka); kolejność wariantów BASE→PRE→POST→RANDOM.

Każda epoka: trening i dev (5000, bez augmentacji, raz na obraz). Metryki
jak E001 (train_nll/count, dev_nll/top1). Główny wynik: **epoka120 (ostatnia)**
— jednakowa dla wszystkich, bez early stopping i bez wybierania epoki po
fakcie. Dodatkowo raportujemy epokę z najlepszym dev jako diagnostykę
(ta sama reguła dla wszystkich wariantów), wyraźnie oznaczoną jako wtórną.
Porównanie POST−PRE: sparowana różnica na dev na epoce120 + bootstrap
stratyfikowany po klasach (2000 replikacji, seed20261009) jak w E003,
oraz analogicznie wobec BASE i RANDOM.

## Kontrole i kryteria

Techniczny sukces: dokładnie120 epok/wariant (logowane liczniki kroków),
ten sam batch/przepływ danych/coverage ID, finite loss, dokładnie0
wykorzystania rezerwy/testu, hashe wejściowych modeli niezmienione po
treningu, baseline startu zgodny z wynikami E003 (66,84 dla BASE, itd.),
identyczne hiperparametry w czterech manifestach. Wynik jakościowy gorszy
lub brak separacji metod nie jest błędem; raportujemy wszystkie cztery.
Nie ustalamy po wyniku progu sukcesu naukowego.

Ograniczenia z góry: jeden rodzic (seed42), dev wcześniej używany,
budżet zaadaptowany z innej architektury, jeden poziom kompresji,
brak niezależnego testu. Cztery sekwencyjne treningi to cztery przebiegi
tego samego zadania na4GPU, nie niezależne seedy modelu. E004 nie dowodzi
nowości metody; może pokazać, czy PO równym budżecie różnica PRE/POST
w ogóle się utrzymuje.

## Artefakty i odtworzenie

Nowy `runs/E004/<UTC>_<git>_<unikalny>/`: manifest, źródła, konfiguracja,
środowisko/Conda/pip, komenda, hashe wejść i czterech modeli, logi per
wariant, metrics.jsonl/json, dev_preds/logits (epoka120), checkpointy
inference-only po dostrojeniu, bootstrap i artifacts.sha256. Wykresy
i raport w tym folderze E004, bez nadpisywania innych biegów. Planowany
czas: 4×120 epok na4GPU; zmierzony czas zapiszemy, nie zgadujemy go.

Komenda i hashe kodu: **do dopisania przed frozen**. Sam ten dokument
nie oznacza wykonania. Do zamrożenia brakuje implementacji przebiegu
dostrajania, kontroli CPU i instrukcji odtworzenia.
