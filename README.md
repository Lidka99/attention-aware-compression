# attention-aware-compression

Reproducible experiments on attention-aware channel merging and CNN compression.

Status: E001–E010 zakończone. E003/E004: przy 10% redukcji po równym dostrajaniu wybór kanałów nie dał przewagi (RANDOM dorównał). E005/E007: baseline'y Tiny ImageNet 96×96 (51,53%) i 224×224 (58,49%). E006: bramki SE nie przewidują szkody ponad s_post; wątek zamknięty. E008: po 60 epokach POST−PRE +0,51 pp (CI obejmuje 0). E009: po 120 epokach POST−PRE +0,28 pp (CI obejmuje 0), różnice między selekcjami znikają. E010: jedyny kandydat TAYLOR (+1,03 pp vs RANDOM na Tiny 30%, T=5) nie zreplikował się na CIFAR-100 → `survivor_not_replicated`. Nowość badawcza niepotwierdzona; następny krok: decyzja o E011.

## Od czego zacząć

**Aktualizacja: E011 i E012 zakończone.** E011: cztery checkpointy, średnia
dev95,46%. E012: folding bez fine-tuningu, po korekcie BN94,20/89,30/70,50%
dla 10/30/50% usuniętych kanałów; BASE po BN95,48%.
[Wyniki E012 i ograniczenia adaptacji](experiments/2026-10-24_E012_folding_referencja/WYNIKI.md).
Następny etap: zamrożone porównanie grupowania przy wspólnym operatorze.

**E011 uruchomiony 2026-10-24:** cztery niezależne treningi ResNet18/CIFAR-10
od zera, 200 epok, GPU0–3, seedy42–45. [Stan i run](experiments/2026-10-22_E011_cifar10_resnet18_checkpoint/STATUS.md).
[E012 — szkic kontroli foldingu](experiments/2026-10-24_E012_folding_referencja/OPIS.md)
oczekuje na checkpointy i testy operatora.

**Bieżący kierunek (2026-10-22): folding bez fine-tuningu.** Najpierw odtworzenie *Cut Less, Fold More* na CIFAR-10/ResNet18, następnie porównanie miar grupowania przy tym samym operatorze i budżecie. [Audyt kodu i plan](literature/AUDYT_CUT_LESS_FOLD_MORE.md). Wcześniejsza propozycja kontynuacji TAYLORA nie jest aktualnym następnym krokiem.

**[Aktualny plan i najbliższy krok](AKTUALNY_PLAN.md)** — bieżący punkt odniesienia.

**[E003 — wynik kompresji 10% całej sieci](experiments/2026-10-09_E003_kompresja_10proc_parametrow/WYNIKI.md)**,
[protokół](experiments/2026-10-09_E003_kompresja_10proc_parametrow/PROTOKOL_v1.md),
[wyjaśnienie](experiments/2026-10-09_E003_kompresja_10proc_parametrow/WYJASNIENIE.md).
Fizyczne modele 3 721 424 parametry; duża utrata jakości bez naprawy; POST−PRE +0,30 pp (przedział obejmuje brak różnicy).

**[E004 — wynik wspólnego dostrajania](experiments/2026-10-10_E004_wspolne_dostrajanie/WYNIKI.md)**,
[protokół](experiments/2026-10-10_E004_wspolne_dostrajanie/PROTOKOL_v1.md),
[wyjaśnienie](experiments/2026-10-10_E004_wspolne_dostrajanie/WYJASNIENIE.md).
Po 120 epokach SGD wszystkie warianty ~67%; POST−PRE +0,36 pp, przedział [−0,30; +1,02] pp; wybór losowy dorównał.

**[E005 — baseline Tiny ImageNet od zera](experiments/2026-10-10_E005_tiny_imagenet_od_zera/WYNIKI.md)**,
[protokół](experiments/2026-10-10_E005_tiny_imagenet_od_zera/PROTOKOL_v1.md).
Dev top-1 51,53% (100 epok, 3h10min na 4 GPU); przeuczenie obecne; pod publikację planowany trening 224×224 z mocniejszą recepturą.

**[E006 — wynik badania kryterium z bramek SE](experiments/2026-10-11_E006_kryterium_bramkowe_predykcja/WYNIKI.md)**,
[protokół](experiments/2026-10-11_E006_kryterium_bramkowe_predykcja/OPIS.md).
Wynik negatywny: g_pert ≈ 0 ponad s_post; wątek bramek zamknięty.

**[E007 — baseline 224×224 pod publikację](experiments/2026-10-11_E007_baseline_224/WYNIKI.md)**,
[protokół](experiments/2026-10-11_E007_baseline_224/PROTOKOL_v1.md).
100 epok od zera na 4 GPU (run `4de51b3`); dev top-1 58,49% (epoka 100), train 99,50%, dev NLL 2,0001. Zastępuje E005 jako główny punkt odniesienia.

**[E008 — wynik kompresji na baseline 224](experiments/2026-10-12_E008_kompresja_224/WYNIKI.md)**,
[protokół](experiments/2026-10-12_E008_kompresja_224/PROTOKOL_v1.md).
Etap 1 bez naprawy: 10% → PRE 47,4 / POST 47,1 / RANDOM 40,8; 30% → 9,7 / 6,7 / 1,0. Etap 2 (60 epok): BASE 59,45 / POST 58,32 / PRE 57,81 / RANDOM 57,55; POST−PRE +0,51 pp (CI [−0,18; +1,19]); POST−RANDOM na granicy istotności.

**[E009 — wynik naprawy 120 epok](experiments/2026-10-12_E009_naprawa_120/WYNIKI.md)**,
[protokół](experiments/2026-10-12_E009_naprawa_120/PROTOKOL_v1.md).
BASE 60,04 / POST 59,46 / PRE 59,18 / RANDOM 59,21; POST−PRE +0,28 pp (CI [−0,41; +0,96]) — przewaga atencji nie występuje także przy dłuższym budżecie.

**[E010 — benchmark kryteriów selekcji](experiments/2026-10-18_E010_benchmark_kryteriow/WYNIKI.md)**,
[protokół](experiments/2026-10-18_E010_benchmark_kryteriow/PROTOKOL_v1.md).
Jedyny kandydat TAYLOR przeszedł próg primary na Tiny 30%, T=5 (+1,03 pp vs RANDOM, Holm OK), ale nie zreplikował się na CIFAR-100 10% (+0,36 pp, CI obejmuje 0) → `survivor_not_replicated`.

[Sposób dokumentowania](docs/DOKUMENTOWANIE.md): audyt i wykresy zostają przy eksperymencie; każde uruchomienie ma osobny katalog wyników.

**[Raport audytu i wykresy E001](experiments/2026-09-26_E001_efficientnet_b0_od_zera/AUDYT_WYNIKI.md)** — zgodność checkpointu, przebieg uczenia i ograniczenia.

[E002 — wyniki i wykresy pierwszej obserwacji SE](experiments/2026-10-01_E002_obserwacja_se/WYNIKI.md), [protokół i uruchomienie](experiments/2026-10-01_E002_obserwacja_se/PROTOKOL_v1.md), [proste wyjaśnienie](experiments/2026-10-01_E002_obserwacja_se/WYJASNIENIE.md). 1000 obrazów train, bez zmiany wag: wybrane podobne pary mają zmienne wagi SE, ale pozostają bardzo podobne po SE. To obserwacja, nie dowód korzyści kompresji.

[Dodatkowa analiza E002: wynik kontroli stałej średniej](experiments/2026-10-01_E002_obserwacja_se/analizy/2026-10-02_skladowa_stala/WYNIKI.md) — wykonana na CPU bez nowej inferencji. Podobieństwo utrzymuje się po centrowaniu; stała średnia wybranych kanałów ma mały udział energii.

[Wynik kontroli szablonu przestrzennego E002](experiments/2026-10-01_E002_obserwacja_se/analizy/2026-10-07_szablon_przestrzenny/WYNIKI.md), [wyjaśnienie CPU/GPU](experiments/2026-10-01_E002_obserwacja_se/analizy/2026-10-07_szablon_przestrzenny/WYJASNIENIE.md) — podobieństwo utrzymuje się po odjęciu średniej mapy (mediana 0,952 przed SE i 0,959 po SE). Szablon nie dominuje w energii wybranych kanałów. Bez nowej inferencji i bez dowodu bezpieczeństwa scalenia.

1. [Cel badań — do czego dążymy](wyjasnienia/CEL_BADAN.md).
2. [Wyjaśnienia krok po kroku](wyjasnienia/WYJASNIENIA_KROK_PO_KROKU.md).
3. [Zasady organizacji przyszłych eksperymentów](experiments/README.md).

## Struktura

- `wyjasnienia/`: cel badań oraz proste objaśnienia, przekazywane też w rozmowie.
- `experiments/`: osobny folder dla każdego eksperymentu, wraz z jego audytami i wykresami.
- `experiments/_szablon/`: pusty wzór opisu eksperymentu.
- `experiments/_szkice/`: zachowane wcześniejsze propozycje, niezatwierdzone do wykonania.
- `configs/`: wcześniejsze szkice konfiguracji; nie są gotowe do uruchomienia.
- `literature/`: literatura i analiza nowości.
- `docs/`: zasady odtwarzania, zasoby, decyzje i kontekst.
- `reports/`: miejsce na przyszłe zestawienia przekrojowe.
- `runs/`: lokalne artefakty uruchomień, wyłączone z Git.
- `data/`, `checkpoints/`: lokalne duże zasoby, wyłączone z Git.

[Wcześniejszy plan badawczy](PLAN_DZIALANIA.md) pozostaje materiałem roboczym. Ustalamy kolejne kroki pojedynczo. [Zasady reprodukowalności](docs/REPRODUCIBILITY.md) wykorzystamy podczas przygotowywania konkretnego eksperymentu.

## Git i odtwarzanie

Repozytorium GitHub: [Lidka99/attention-aware-compression](https://github.com/Lidka99/attention-aware-compression). Gałąź robocza: main. Przed uruchomieniem eksperymentu zapiszemy wersję kodu w commicie oraz dokładne wersje danych, wag, środowiska i konfiguracji. Duże artefakty wymagają trwałego magazynu i identyfikatorów; samo wyłączenie z Git ich nie archiwizuje.

## Niezależny projekt — ustalenie 2026-09-26

Eksperymenty piszemy od nowa w tym repozytorium. Nie kopiujemy ani nie importujemy kodu, konfiguracji, wag czy wyników z attention_v2 ani attentionv3. Nie tworzymy zależności od tych folderów. Wcześniejsza propozycja użycia lokalnego checkpointu SE jest wycofana.

Dobór modelu i danych przeprowadzamy od początku. Możemy korzystać z udokumentowanych bibliotek i znanych architektur. Aktualna decyzja: trenujemy od losowych wag na 4 GPU w środowisku Conda attention, bez publicznych checkpointów.

## E001 — zakończony model bazowy

[Opis eksperymentu](experiments/2026-09-26_E001_efficientnet_b0_od_zera/OPIS.md), [instrukcja uruchomienia](experiments/2026-09-26_E001_efficientnet_b0_od_zera/URUCHOMIENIE.md) i [wyniki](experiments/2026-09-26_E001_efficientnet_b0_od_zera/WYNIKI.md). Conda attention, 4×RTX 2080 Ti 11 GB, jeden model od losowych wag. CIFAR-100: best dev 66,84% (epoka 177), last dev 66,54%; oficjalnego testu nie oceniano. Audyt 2026-09-29 potwierdził ocenę best.pt; minimum dev NLL przypada na epokę 24, a krzywe wskazują na przeuczenie.
