# Wyniki E001 — stan 2026-09-26

Pełny eksperyment naukowy: **nieuruchomiony**. CIFAR-100 nie został pobrany. Nie ma wyniku jakości klasyfikacji na prawdziwych danych.

## Sprawdzone technicznie

- Nowy kod działa na 4×RTX 2080 Ti w Condzie attention.
- Dwa niezależne uruchomienia testu syntetycznego dały identyczne metryki (pomijając czas) i identyczny hash tensorów końcowego modelu.
- Wznowienie od końca pierwszej epoki dało identyczne metryki końcowe i model jak bieg nieprzerwany.
- Kopie modelu na wszystkich czterech GPU były zgodne. Sprawdzono integralność artefaktów wszystkich trzech przebiegów.
- Test odbył się w roboczym drzewie przed commitem. Manifesty zawierają dirty=true i dokładne hashe plików Python. Szczegóły: [TEST_TECHNICZNY.json](TEST_TECHNICZNY.json).

## Lokalne artefakty

- `runs/E001/20260926T012024Z_e6dfc29_seed42_nDKlnW/`
- `runs/E001/20260926T012039Z_e6dfc29_seed42_vjsfw8/`
- `runs/E001/20260926T012116Z_e6dfc29_seed42_xanUr2/`

Ścieżki liczone od korzenia repo. Artefakty pozostają lokalne, nie są dodane do Git. Ich trwała zewnętrzna archiwizacja nie została ustalona.

Wcześniejsze trzy próby techniczne z 01:14–01:15 UTC także zachowano w runs/E001; miały zerowy learning rate w drugiej epoce. Końcowe sprawdzenie powyżej używa niezerowego learning rate w obu epokach, aby weryfikować także aktualizacje po wznowieniu. Nie usunięto wcześniejszych wyników.

## Ograniczenia

To dowód zgodności krótkich przebiegów na syntetycznych danych w tym samym środowisku. Nie sprawdza jeszcze pobierania CIFAR-100, treningu z workerami na rzeczywistych danych, odtworzenia w nowym środowisku ani 200-epokowego treningu. Następnie wybieramy dane, przygotowujemy ich manifest i weryfikujemy rzeczywisty przepływ danych. Wynik nie potwierdza hipotezy o kompresji.

## Start pełnego biegu — 2026-09-26

CIFAR-100 pobrany, zweryfikowany oficjalną sumą MD5, SHA-256 archiwum i plików zapisany w DANE.json. Podział gotowy: 40 000 train, 5 000 dev, 5 000 rezerwy. Zlecono pełny bieg 200 epok; proces i bieżący katalog wyników wskazuje lokalny runs/E001/active.json. Manifest w katalogu biegu jest źródłem jego aktualnego statusu.
