# Wyniki E001 — stan 2026-09-26

Pełny pilot zakończony: **200/200 epok**. Aktualny wynik i interpretacja znajdują się w sekcji na końcu tego dokumentu. Poniższe wcześniejsze wpisy zachowują historię przygotowania.

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

## Pełny trening zakończony — sprawdzono 2026-09-28

- Bieg: `20260926T013203Z_7cc3f46_seed42_qHKULg`.
- Start: 2026-09-26 01:32:10 UTC; koniec: 04:15:52 UTC. Około 2 h 44 min.
- Status completed, exit_code=0, 200/200 epok; stderr pusty.
- Najlepsza walidacja: **66,84% (3342/5000), epoka 177**.
- Ostatnia walidacja: **66,54% (3327/5000), epoka 200**.
- Ostatni train: 39918/39936 poprawnych odpowiedzi, około **99,95%**. Duża różnica train/dev wskazuje na przeuczenie; nie należy interpretować wyniku treningowego jako jakości na nowych danych.
- Oficjalny test nie był oceniany. To jeden model, seed=42, trenowany wspólnie na czterech GPU; nie cztery niezależne powtórzenia.
- best.pt i last.pt istnieją (około 50 MB każdy). Wszystkie pliki z artifacts.sha256 przeszły kontrolę integralności. Końcowe kopie modelu na czterech GPU mają identyczny hash.
- Pełne artefakty pozostają lokalnie w runs/E001/ pod powyższym ID; GitHub zawiera kod i podsumowanie, nie checkpointy.

Szczegóły: [PODSUMOWANIE_TRENINGU.json](PODSUMOWANIE_TRENINGU.json). Następny mały krok: obejrzeć przebieg uczenia i zweryfikować ocenę best.pt przed kompresją. Nie uruchomiono kolejnego eksperymentu ani oficjalnego testu.
