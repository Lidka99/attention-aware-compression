# E001 — odtworzenie modelu bazowego i pomiar aktywacji

Status: draft. Wersja: 0.1. Konfiguracja: `configs/E001_baseline.draft.json`.

## Pytanie
Czy potrafimy odtworzyć ewaluację wybranego checkpointu i poprawnie zebrać aktywacje przed/po atencji?

## Wejście i procedura
1. Wypełnić `docs/ASSETS.md`, przypiąć wersję modelu, checkpoint i preprocessing. Zapisać manifest danych i podziałów; zamrozić środowisko.
2. Ustawić model w eval, ustalić precision i deterministyczność, wyłączyć aktualizację BN. Ustalić metrykę zgodną z wybranym zadaniem.
3. Wykonać niezmodyfikowany model na zbiorze walidacyjnym. Zapis predykcji zawiera ID i etykiety.
4. Zebrać h, a i z w dokładnie określonych punktach grafu. Sprawdzić zgodność z = a*h, z poprawnym broadcastingiem i tolerancją zależną od precision. Hooki nie mogą zmieniać predykcji.
5. Powtórzyć przebieg w tym samym środowisku, następnie z nowego checkoutu i środowiska. Opisać zgodność bitową lub różnice w ustalonej tolerancji.

## Kontrole i metryki
Bez kompresji i bez naprawy. Pierwotnie: zgodność ewaluacji z checkpointem i powtórzeniem; dodatkowo accuracy, NLL, różnica wyjść z hookami i bez. Oczekiwana jakość, tolerancja i dokładna komenda: UNSET — ustalić przed `frozen`.

## Artefakty i bramka
Manifest uruchomienia, resolved config, lockfile, hashe wejść, predykcje, logi, opis kształtów h/a/z. Gdy reprodukcja lub hooki nie są poprawne, nie interpretować późniejszej kompresji. Raport: `reports/E001_baseline.md` (brak wyników). Limit czasu/GPU/RAM: UNSET.
