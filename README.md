# attention-aware-compression

Reproducible experiments on attention-aware channel merging and CNN compression.

Status: E001–E007 zakończone. E003/E004: przy 10% redukcji po równym dostrajaniu wybór kanałów nie dał przewagi (RANDOM dorównał). E005: baseline Tiny ImageNet od zera — dev 51,53% przy 96×96. E006: sygnał z bramek SE nie przewiduje szkody scalenia ponad podobieństwo po SE (primary_passed=false); wątek zamknięty. E007: baseline 224×224 pod publikację — dev 58,49% przy 100 epokach od zera. Nowość badawcza niepotwierdzona.

## Od czego zacząć

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
