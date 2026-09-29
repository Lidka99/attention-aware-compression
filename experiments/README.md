# Eksperymenty

E001 zakończono: 200 epok treningu. Audyt checkpointu i przebiegu uczenia zakończony pozytywnie 2026-09-29.

| Eksperyment | Stan | Dokumentacja |
|---|---|---|
| E001 — EfficientNet-B0 od zera | Trening i audyt zakończone | [Wyniki](2026-09-26_E001_efficientnet_b0_od_zera/WYNIKI.md), [protokół audytu](2026-09-26_E001_efficientnet_b0_od_zera/AUDYT_OPIS_v1.md), [raport i wykresy](2026-09-26_E001_efficientnet_b0_od_zera/AUDYT_WYNIKI.md) |

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
