# E002 — poprawność fizycznego scalania

Status: draft. Wersja: 0.1. Wymaga poprawnego E001.

## Pytanie
Czy operator scalania implementuje zadeklarowaną transformację i daje legalny, mniejszy graf?

## Procedura
1. Wybrać blok i stałą liczbę kanałów wynikowych. Opisać zmiany producenta, BN, atencji, odbiorcy i gałęzi residual. ECA wymaga kontroli sąsiedztwa kanałów, SE wymiarów projekcji.
2. Zapisać algebraiczną definicję scalania oraz kontrakt kompensacji. Dla bloku z atencją dokładna równoważność jest możliwa tylko pod określonymi warunkami — same identyczne filtry nie wystarczą przy różnych bramkach.
3. Sprawdzić operację tożsamościową i mały przykład z identycznymi kanałami ORAZ zgodnymi bramkami, gdzie wynik powinien być zachowany. Ustalić osobne tolerancje przed pomiarem.
4. Dla rzeczywistego bloku wykonać legalne scalenie, zweryfikować redukcję wymiarów, parametrów, zgodność serializacji i ponownego wczytania.
5. Porównać aktywacje nauczyciela z rzeczywistym przeliczeniem atencji w mniejszym modelu. Zamrożone bramki nauczyciela mogą służyć tylko diagnozie, nie ocenie modelu końcowego.
6. Oddzielnie uruchomić brak naprawy i wspólną kompensację; zapisać jej dane, hiperparametry i koszt. Adaptację GRAIL opisać jako adaptację.

## Metryki i wynik
Dla dokładnych przypadków: maksymalny błąd wyjścia w tolerancji. Dla rzeczywistych scaleń: błąd wyjścia, liczba parametrów, wymiary i bajty artefaktu; utrata jakości sama nie oznacza błędu implementacji.

Architektura, bloki, konkretne grupy, konfiguracja, komenda, tolerancje i budżet: UNSET. Zamrozić przed wykonaniem. Raport `reports/E002_surgery.md` i pełne manifesty grup. Nie przechodzić do E003 przed poprawnymi kontrolami struktury i algebry.
