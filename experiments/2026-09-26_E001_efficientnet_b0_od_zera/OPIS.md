# 2026-09-26_E001_efficientnet_b0_od_zera

Status: **draft — gotowy kod pilota, wybór CIFAR-100 roboczy; pełny trening nieuruchomiony**.
Wersja protokołu: 0.1-pilot. Data utworzenia: 2026-09-26 UTC.

## Pytanie i cel

Czy potrafimy od początku wytrenować EfficientNet-B0 z jego standardową atencją SE, zachowując dokładny zapis procesu i możliwość jego odtworzenia?

To eksperyment bazowy, który dostarczy checkpointu do przyszłej analizy atencji i kompresji. Nie testuje jeszcze nowej metody ani przewagi nad literaturą. Hipoteza robocza: trening zmniejszy NLL i zwiększy trafność na niewidzianym podczas uczenia zbiorze dev. Nie ustalamy arbitralnej docelowej accuracy; jakość pilota ocenimy przed wykorzystaniem wag do dalszych badań.

## Ustalony wariant roboczy

- Model: torchvision EfficientNet-B0, standardowe SE, `weights=None`, klasyfikator na 100 klas. Bez publicznych wag, bez importów i zasobów z attention_v2/attentionv3.
- Dane: oficjalny CIFAR-100. Obrazy 32×32 powiększamy do 96×96. To adaptacja pilota, a nie odtworzenie wyników ImageNet ani zmiana obrazu w dane o wyższej rzeczywistej rozdzielczości.
- Podział oficjalnych 50 000 obrazów treningowych: 400/50/50 na klasę, czyli 40 000 train, 5 000 dev i 5 000 rezerwy. Seed podziału: 20260926. Listy ID są zapisane w splits.json, a nie odtwarzane tylko z seeda.
- Oficjalny test 10 000 obrazów nie jest oceniany. Archiwum źródłowe zawiera ten plik, ale kod treningu go nie ładuje. Rezerwa także nie jest używana. Jej dokładna rola w kolejnych badaniach zostanie ustalona przed ich rozpoczęciem.
- Rozłączność sprawdzamy według ID. Audyt duplikatów zawartości między podziałami pozostaje do wykonania przed zamrożeniem badania potwierdzającego.
- Train: losowy crop 32 z padding=4, poziome odbicie p=0.5, resize bilinear z antialias do 96, tensor, normalizacja mean=std=(0.5,0.5,0.5). Dev: resize, tensor, ta sama normalizacja, bez losowych augmentacji. Stałe normalizacji nie są estymowane z dev/test.
- 200 epok, AdamW, lr=0.001, weight decay=0.01, domyślne betas=(0.9,0.999), eps=1e-8. Pięć epok liniowego warmupu (0.2,0.4,0.6,0.8,1.0 × lr), następnie cosine do zera w ostatniej epoce. Cross entropy bez label smoothing. Standardowe torchvision dropout=0.2 i stochastic_depth_prob=0.2.
- Dokładnie 4 GPU, po 32 obrazy na GPU: globalny batch=128. Bez akumulacji gradientów. Jeden model rozproszony DDP, SyncBatchNorm. To jedno powtórzenie treningu, nie cztery niezależne modele.
- Dwa workery danych na proces, jeden wątek CPU obliczeń na proces. AMP FP16 w treningu z GradScaler; ocena FP32. TF32 wyłączone. Algorytmy deterministyczne wymagane; błędu niedeterministycznej operacji nie ignorujemy.
- Seed treningu 42, oddzielny seed podziału. Inicjalizacja wag wspólna dla ranków; potem rank otrzymuje seed 42+rank. Seed loadera: 42+rank+10000×epoch. Seedy kalibracji/kompresji nie dotyczą tego eksperymentu.
- Sampler tasuje dane co epokę. `drop_last=True` w treningu: 39 936/40 000 obrazów na epokę przy tym batchu, odrzucana końcówka zmienia się z tasowaniem. Dev: każdy obraz liczony raz, bez dopełniania podziałów GPU.

To jawny przepis startowy, a nie optymalna receptura EfficientNet. Zmiany po obejrzeniu wyniku zapisujemy jako kolejną wersję protokołu; nie podmieniamy ustawień w zakończonym uruchomieniu.

## Metryki i wybór checkpointu

Po każdej epoce zapisujemy train NLL, liczbę trafnych odpowiedzi i liczbę próbek; dla dev te same dane oraz top-1. NLL to kara za przewidywane prawdopodobieństwo właściwej klasy; niższa jest lepsza. Top-1 to odsetek obrazów, dla których najwyżej oceniona klasa była poprawna.

Główna metryka jakości pilota: dev top-1. `best.pt` wybieramy przez najwyższe dev top-1; remis zachowuje wcześniejszy checkpoint. `last.pt` służy do wznowienia. Dev używany do wyboru checkpointu nie jest niezależnym testem końcowym.

Wznowienie po zakończonej epoce wymaga identycznego commita, konfiguracji, manifestu danych i zapisanego środowiska. Tworzy osobny katalog wyników i zapisuje hash checkpointu rodzica. Przenosi model, optymalizator, GradScaler i stany RNG wszystkich ranków. Nie obsługujemy dokładnego wznowienia w środku epoki. Jeśli w nowym odcinku nie ma poprawy, najlepszy checkpoint może pozostać w katalogu rodzica — zachowujemy cały łańcuch uruchomień.

## Warunki zakończenia

- Normalnie: 200 epok i komplet artefaktów.
- Błąd danych, NaN/Inf, OOM, brak 4 GPU, niezgodne kopie modelu: przerwanie z błędem. Nie zmniejszamy automatycznie batcha, obrazu ani liczby kart.
- Nie ma automatycznego early stopping ani strojenia hiperparametrów.
- Czas pełnego treningu i całkowity koszt nie zostały jeszcze zmierzone na CIFAR-100. Pomiar wydajności przeprowadzimy przed długim przebiegiem. Nie wnioskujemy o czasie z testu na sztucznych danych.
- Warunki przed pełnym biegiem: potwierdzony zbiór danych, zamrożony protokół, przygotowany manifest danych, czysty commit oraz uzgodniona trwała archiwizacja dużych artefaktów. Te elementy nie są wszystkie zamknięte.

## Reprodukowalność

Środowisko: Conda `attention`, Python 3.11.15, torch 2.12.0+cu130, torchvision 0.27.0+cu130, 4×RTX 2080 Ti 11 GB, sterownik 580.173.02. Eksporty są w ../../configs/environment/.

Każde uruchomienie ma osobny katalog `runs/E001/<UTC>_<git>_seed42_<unikalny>/`. Zapisujemy manifest z commitem, status i exit code, pełną konfigurację z hashem, wersje bibliotek i sprzętu, dane/splity z hashami, logi, metryki, predykcje najlepszego dev oraz checkpointy. Każda karta raportuje rank i szczyt pamięci; końcowe hashe modelu na czterech kartach muszą być identyczne. `artifacts.sha256` obejmuje pliki wynikowe po zamknięciu logów.

Test techniczny na syntetycznych danych sprawdza działanie DDP, powtarzalność i wznowienie. Kryterium dla dwóch identycznych testów w tym samym środowisku: identyczne liczbowe metryki (poza czasem) i hash tensorów modelu. Nie porównujemy bajtowo plików torch.save. Nie gwarantujemy zgodności bitowej na innym sprzęcie ani między wersjami bibliotek. Odtworzenie pełnego treningu w świeżym checkoutcie i środowisku pozostaje do sprawdzenia.

## Źródła i uruchomienie

- [Instrukcja uruchomienia](URUCHOMIENIE.md)
- [Proste wyjaśnienie](WYJASNIENIE.md)
- [Wyniki i status](WYNIKI.md)
- [EfficientNet w torchvision](https://docs.pytorch.org/vision/stable/models/efficientnet.html)
- [CIFAR-100 — źródło](https://www.cs.toronto.edu/~kriz/cifar.html)

Kryteria kompresji i candidate_manifest nie dotyczą E001: nie zmieniamy kanałów.
