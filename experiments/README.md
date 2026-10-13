# Eksperymenty

E001 zakończono: 200 epok treningu. Audyt checkpointu i przebiegu uczenia zakończony pozytywnie 2026-09-29.

| Eksperyment | Stan | Dokumentacja |
|---|---|---|
| E001 — EfficientNet-B0 od zera | Trening i audyt zakończone | [Wyniki](2026-09-26_E001_efficientnet_b0_od_zera/WYNIKI.md), [protokół audytu](2026-09-26_E001_efficientnet_b0_od_zera/AUDYT_OPIS_v1.md), [raport i wykresy](2026-09-26_E001_efficientnet_b0_od_zera/AUDYT_WYNIKI.md) |
| E002 — obserwacja SE w jednym bloku | Completed; 1000 obrazów train, 4 GPU, model niezmieniony | [Wyniki i wykresy](2026-10-01_E002_obserwacja_se/WYNIKI.md), [protokół](2026-10-01_E002_obserwacja_se/PROTOKOL_v1.md), [wyjaśnienie](2026-10-01_E002_obserwacja_se/WYJASNIENIE.md) |
| E003 — kompresja 10% parametrów całej sieci | Completed bez dostrajania; 3 721 424 param./model | [Wyniki i wykresy](2026-10-09_E003_kompresja_10proc_parametrow/WYNIKI.md), [protokół](2026-10-09_E003_kompresja_10proc_parametrow/PROTOKOL_v1.md), [wyjaśnienie](2026-10-09_E003_kompresja_10proc_parametrow/WYJASNIENIE.md) |
| E004 — wspólne dostrajanie po kompresji | Completed; wszystkie warianty ~67%, brak przewagi selekcji | [Wyniki](2026-10-10_E004_wspolne_dostrajanie/WYNIKI.md), [protokół](2026-10-10_E004_wspolne_dostrajanie/PROTOKOL_v1.md), [wyjaśnienie](2026-10-10_E004_wspolne_dostrajanie/WYJASNIENIE.md) |
| E005 — Tiny ImageNet od zera | Completed; dev 51,53% (96×96, 100 epok) | [Wyniki](2026-10-10_E005_tiny_imagenet_od_zera/WYNIKI.md), [protokół](2026-10-10_E005_tiny_imagenet_od_zera/PROTOKOL_v1.md) |
| E006 — kryterium z bramek SE (predykcja) | Completed; wynik negatywny (g_pert ≈ 0 ponad s_post) | [Wyniki](2026-10-11_E006_kryterium_bramkowe_predykcja/WYNIKI.md), [protokół](2026-10-11_E006_kryterium_bramkowe_predykcja/OPIS.md) |
| E007 — baseline 224×224 (Tiny ImageNet) | Completed; dev 58,49% (224×224, 100 epok) | [Wyniki](2026-10-11_E007_baseline_224/WYNIKI.md), [protokół](2026-10-11_E007_baseline_224/PROTOKOL_v1.md) |
| E008 — kompresja 10%/30% na baseline 224 | Completed; POST−PRE +0,51 pp (CI [−0,18; +1,19]); POST > RANDOM na granicy | [Wyniki](2026-10-12_E008_kompresja_224/WYNIKI.md), [protokół](2026-10-12_E008_kompresja_224/PROTOKOL_v1.md) |

E007, 2026-10-12: [baseline 224×224 pod publikację](2026-10-11_E007_baseline_224/WYNIKI.md), completed. 100 epok od zera na 4 GPU (run 4de51b3), dev top-1 58,49% (epoka 100); zastępuje E005 jako główny punkt odniesienia do badań kompresji.

E008, 2026-10-13: [wynik kompresji i naprawy 60 epok](2026-10-12_E008_kompresja_224/WYNIKI.md), completed. Etap 1 bez naprawy: selekcja > losowa, POST < PRE; etap 2: POST−PRE +0,51 pp (próg nieprzekroczony), POST istotnie ponad RANDOM na granicy. E009 — naprawa 120 epok: pierwszy bieg przerwany awarią maszyny 13.10; pełne powtórzenie od 2026-10-13 17:23 UTC, wynik ~14.10.

Uzupełnienie E006, 2026-10-11: [kontrola FWER sygnałów wtórnych (analiza na zamrożonym pairs.csv)](2026-10-11_E006_kryterium_bramkowe_predykcja/analizy/2026-10-11_kontrola_FWER_wtornych/WYNIKI.md), completed; bez wpływu na decyzję pierwszorzędną.

Uzupełnienie E002, 2026-10-02: [kontrola stałej średniej, wynik i wykresy](2026-10-01_E002_obserwacja_se/analizy/2026-10-02_skladowa_stala/WYNIKI.md), completed. CPU na istniejących mapach, bez nowego odczytu modelu.

Uzupełnienie E002, 2026-10-07: [kontrola szablonu i zmian odpowiedzi, wynik i wykresy](2026-10-01_E002_obserwacja_se/analizy/2026-10-07_szablon_przestrzenny/WYNIKI.md), completed. Kontrole E002 domknięte, etap 4 jeszcze nieuruchomiony.

Audyt i wykresy pozostają przy E001. Kolejny numer oznacza nową próbę badawczą,
nie nowy wykres. [Zasady dokumentowania](../docs/DOKUMENTOWANIE.md).

- [_szablon/OPIS.md](_szablon/OPIS.md): pusty wzór opisu do wykorzystania później.
- [_szkice/](_szkice/): wcześniejsze propozycje i ich historyczny rejestr; nie są zatwierdzonym harmonogramem ani wykonanymi eksperymentami.

Gdy wspólnie ustalimy konkretny eksperyment, utworzymy dla niego osobny folder `RRRR-MM-DD_E001_krotka_nazwa/` (kolejne otrzymają kolejne numery). Numery starych szkiców nie ustalają nowej kolejności.

W każdym takim folderze będą:

- `OPIS.md`: czego dotyczy eksperyment, dlaczego go robimy, pytanie, hipoteza, procedura i sposób odtworzenia.
- `WYJASNIENIE.md`: proste objaśnienie krok po kroku, przekazywane również w rozmowie.
- `WYNIKI.md`: po wykonaniu — wyniki, wnioski, błędy i odnośniki do uruchomień.
- `configs/`: dokładne konfiguracje, gdy przygotujemy implementację.

Pliki powstają w miarę pracy. Duże dane i wagi pozostają w katalogach zasobów, a surowe wyniki w `runs/`; opis eksperymentu musi wskazywać ich wersje, lokalizacje i identyfikatory. Nie kopiujemy ich do Git.

## Uruchamianie i opis kodu

- Eksperymenty uruchamiamy na 4 GPU. Konfiguracja i komenda odtworzenia muszą jawnie określać użycie czterech GPU. Nie przechodzimy automatycznie na mniejszą liczbę urządzeń; jeśli cztery GPU nie są dostępne, zgłaszamy przeszkodę.
- Dobieramy sposób podziału pracy do eksperymentu: obliczenia rozproszone lub niezależne zadania na czterech GPU. Samo udostępnienie czterech urządzeń nie oznacza ich wykorzystania. Sposób podziału pracy opisujemy w protokole i sprawdzamy w logach.
- Dla reprodukowalności zapisujemy modele GPU, wersje środowiska, przypisanie zadań lub procesów do GPU, seedy i sposób agregacji wyników. Przy treningu zapisujemy też batch size na GPU i efektywny globalny batch size, uwzględniając akumulację gradientów. Cztery GPU nie oznaczają automatycznie czterech niezależnych powtórzeń.
- Każdy plik z kodem eksperymentu zaczyna się krótkim komentarzem lub docstringiem: co robi, jakiego eksperymentu dotyczy, jakie dane przyjmuje i jakie wyniki zapisuje. Zawiera też odnośnik do pełnego OPIS.md.

E001 wytrenowano od losowych wag w środowisku Conda attention. Ukończony protokół i artefakty zachowujemy; późniejsze analizy dokumentujemy osobno w jego folderze.

## Daty w nazwach eksperymentów

Folder każdego nowego eksperymentu nazywamy `RRRR-MM-DD_E001_krotka_nazwa`, np. `2026-09-25_E001_nazwa_eksperymentu`. Jest to tylko przykład formatu, nie utworzony eksperyment. Data oznacza dzień utworzenia protokołu (UTC), a E001 jest stałym, kolejnym identyfikatorem. Pełną nazwę umieszczamy też w tytule OPIS.md i w opisie kodu. Nie zmieniamy daty w nazwie przy ponownym uruchomieniu; każde uruchomienie ma osobno zapisany znacznik czasu UTC.

## Pierwszy eksperyment

[E001 — trening EfficientNet-B0 od zera](2026-09-26_E001_efficientnet_b0_od_zera/OPIS.md), [instrukcja](2026-09-26_E001_efficientnet_b0_od_zera/URUCHOMIENIE.md). Protokół zamrożony; trening na CIFAR-100 zakończony. Szczegóły wykonania zawiera raport wyników.
