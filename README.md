# Atencja kanałowa i kompresja CNN

Status: projektowanie badań. Nowość niepotwierdzona; eksperymenty nieuruchomione.

Pytanie: czy zachowanie atencji dla różnych wejść wnosi informację przy wyborze kanałów do scalenia ponad podobieństwo aktywacji po atencji?

## Od czego zacząć

1. Wypełnić [audyt nowości E000](experiments/E000_novelty.md) i [macierz literatury](literature/MATRIX.md).
2. Wybrać model, checkpoint oraz dane w [rejestrze zasobów](docs/ASSETS.md). Skontrolować prawo dostępu i możliwość udostępnienia instrukcji odtworzenia.
3. Zamrozić protokół [E001 — model bazowy](experiments/E001_baseline.md) i przygotować rzeczywiste środowisko z dokładnym lockfile. Repo nie zawiera jeszcze implementacji ewaluacji ani zależności ML.
4. Przejść kolejno przez [E002 — poprawność scalania](experiments/E002_surgery.md) i [E003 — porównanie kryteriów](experiments/E003_criteria.md), zgodnie z ich bramkami wejścia.

Pełny kontekst: [plan badawczy](PLAN_DZIALANIA.md). Zasady odtwarzania: [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Struktura

- `experiments/`: wersjonowane opisy eksperymentów i szablon.
- `configs/`: konfiguracje; obecne pliki JSON są szkicami, nie konfiguracjami wykonawczymi.
- `literature/`: audyt nowości i porównanie metod.
- `docs/`: zasoby, protokół i decyzje.
- `reports/`: raporty i niewielkie tabele wyników, także negatywnych.
- `runs/`: lokalne surowe artefakty pojedynczych uruchomień, wyłączone z Git.
- `data/`, `checkpoints/`: lokalne duże zasoby, wyłączone z Git; ich pochodzenie i hashe zapisujemy osobno.

Rejestr etapów: [experiments/REGISTRY.csv](experiments/REGISTRY.csv). Jedno E-ID może mieć wiele run-ID. Protokół zmieniony po wynikach otrzymuje nową wersję i wpis w decyzjach; wyniki poprzedniej wersji pozostają w rejestrze.

## Aktualny stan wykonania

Utworzono dokumentację i lokalne repozytorium. Nie wybrano checkpointu ani podziałów, nie przygotowano lockfile ML, nie wykonano treningu ani ewaluacji. Nie ma jeszcze poprawnej komendy odtwarzającej E001–E003 — jej zapisanie i sprawdzenie jest warunkiem zmiany statusu na `frozen`.

## Git

Przed uruchomieniem zamrożonego eksperymentu zapisać wszystkie zmiany w commicie. Manifest uruchomienia zawiera SHA tego commita. Nie dodawać do Git danych, checkpointów, tokenów ani dużych plików wynikowych. Dla surowych artefaktów wymagany jest trwały magazyn i manifest hashy: samo ignorowanie ich w Git nie stanowi archiwizacji.

Repozytorium lokalne nie ma zdalnej kopii, dopóki nie skonfigurujemy remote. Początkowy plan zawierał odnośnik do sąsiedniego katalogu; jego treść kontekstowa jest teraz dołączona do `docs/context/`, aby repo było samodzielne.
