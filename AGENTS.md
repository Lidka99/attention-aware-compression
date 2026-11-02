# Zasady pracy w projekcie

Najpierw czytaj `STATUS_BADAN.md` (jedna strona: cel, wyniki, następny krok).
Szczegóły otwieraj tylko z `PODSUMOWANIE.md` lub `WYNIKI.md` danego eksperymentu.
Nie powtarzaj pełnej historii w nowych raportach; nowe podsumowania mają być krótkie.

Pracujemy krok po kroku. Wyjaśnienia przekazujemy po polsku zarówno w rozmowie, jak i w plikach Markdown. Każdy konkretny eksperyment ma osobny folder w experiments/ z opisem i instrukcją odtworzenia.

Ustalenie 2026-09-29: wykresy i audyt istniejącego eksperymentu pozostają w jego folderze. Nowy numer E oznacza nową próbę badawczą; kolejne uruchomienia/audyty mają unikalne katalogi wyników pod runs/<E-ID>/. Zobacz docs/DOKUMENTOWANIE.md. Do drobnych, ograniczonych zadań używaj podagenta general-flash; bez równoległych zmian tych samych plików i konkurencyjnych zadań GPU.

## Uruchamianie i opis kodu

Stan 2026-10-30: **E019 completed — kryterium spełnione.** Tiny ImageNet,
4 seedy, folding 48/48: TILT+0.30 vs UNIFORM **+3,71/+10,32/+7,60 pp**
przy 30/50/70% parametrów; 12/12 par dodatnich. Trzeci zbiór potwierdza
kierunek E014. Bezwzględna jakość po foldingu niska — ograniczenie.
Szczegóły: `STATUS_BADAN.md`,
`experiments/2026-10-29_E019_tinyimagenet_alokacja/WYNIKI.md`.
Wcześniej: **E018 completed, wynik negatywny** — replikacja cosine
na seedach 43–45: +0,47 pp na CIFAR-10/50% (2/3 seedów), przy 30% gorsza;
kryterium niespełnione. **Trop miar grupowania zamknięty**; standardowe
alternatywy nie biją referencji E012. Najmocniejszy wynik pozostaje
po stronie alokacji TILT (E014). Szczegóły: `STATUS_BADAN.md`,
`experiments/2026-10-29_E018_replikacja_cosine/WYNIKI.md`.
Wcześniej: **E016 S1 completed** — diagnostyka wrażliwości 98/98
itemów na CIFAR-10/CIFAR-100 seed42, bez nowego treningu. Szkoda post-BN
przy 75%: średnio 2,82 pp (CIFAR-10) i 9,54 pp (CIFAR-100); BN odzyskuje
większość szkody przed-BN. Primary `weight_reconstruction_rel_error` ma
Spearmana 0,800 / 0,702 ze szkodą post-BN; `depth_norm` sam praktycznie nie
przewiduje szkody na CIFAR-10. Bramki C1–C5 spełnione; **S2 (automatyczna
alokacja 50%) completed 2026-10-28, wynik negatywny**: AUTO−UNIFORM
+2,58/−10,48 pp (CIFAR-10/100), vs najlepszy TILT −1,66/−13,88 pp.
8/8 na czterech GPU, G1/G2 NIE, G3 TAK. Szczegóły w `WYNIKI_S2.md` E016.
Nie replikować tej reguły bez nowej hipotezy i protokołu. E015 pozostaje
completed z niespełnionym kryterium replikacji primary. Szczegóły:
`STATUS_BADAN.md`, `experiments/2026-10-27_E016_wrazliwosc_foldingu/WYNIKI.md`.

Ustalenie naprawcze 2026-10-28: bieżącą naprawę S2 prowadzi agent główny
bez podagentów. Nie delegować całego eksperymentu jako „drobnego zadania”.
Monitorować konkretny PID/kod zakończenia i manifest, z limitem czasu;
nie używać pętli `pgrep -f`, która może dopasować własne polecenie.

- Eksperymenty uruchamiamy na 4 GPU. Konfiguracja i komenda odtworzenia muszą jawnie określać użycie czterech GPU. Nie przechodzimy automatycznie na mniejszą liczbę urządzeń; jeśli cztery GPU nie są dostępne, zgłaszamy przeszkodę.
- Dobieramy sposób podziału pracy do eksperymentu: obliczenia rozproszone lub niezależne zadania na czterech GPU. Samo udostępnienie czterech urządzeń nie oznacza ich wykorzystania. Sposób podziału pracy opisujemy w protokole i sprawdzamy w logach.
- Dla reprodukowalności zapisujemy modele GPU, wersje środowiska, przypisanie zadań lub procesów do GPU, seedy i sposób agregacji wyników. Przy treningu zapisujemy też batch size na GPU i efektywny globalny batch size, uwzględniając akumulację gradientów. Cztery GPU nie oznaczają automatycznie czterech niezależnych powtórzeń.
- Każdy plik z kodem eksperymentu zaczyna się krótkim komentarzem lub docstringiem: co robi, jakiego eksperymentu dotyczy, jakie dane przyjmuje i jakie wyniki zapisuje. Zawiera też odnośnik do pełnego OPIS.md.

Historia E001–E015: skróty w `STATUS_BADAN.md` i `PODSUMOWANIE.md` każdego eksperymentu. Używamy środowiska Conda attention, bez publicznych wag i bez zmiany wcześniejszych wyników.

## Daty w nazwach eksperymentów

Aktualny kierunek: **alokacja globalnego budżetu foldingu bez fine-tuningu**.
Najpierw odtworzenie operatora, potem nierówne ratio per grupa. Audyt:
`literature/AUDYT_CUT_LESS_FOLD_MORE.md`. Nie traktować wcześniejszej
propozycji 3 seedów TAYLORA jako uzgodnionego kierunku. Brak istotności
w E009 nie dowodzi równoważności metod.

Folder każdego nowego eksperymentu nazywamy `RRRR-MM-DD_E001_krotka_nazwa`, np. `2026-09-25_E001_nazwa_eksperymentu`. Jest to tylko przykład formatu, nie utworzony eksperyment. Data oznacza dzień utworzenia protokołu (UTC), a E001 jest stałym, kolejnym identyfikatorem. Pełną nazwę umieszczamy też w tytule OPIS.md i w opisie kodu. Nie zmieniamy daty w nazwie przy ponownym uruchomieniu; każde uruchomienie ma osobno zapisany znacznik czasu UTC.

## Niezależny projekt — ustalenie 2026-09-26

Eksperymenty piszemy od nowa w tym repozytorium. Nie kopiujemy ani nie importujemy kodu, konfiguracji, wag czy wyników z attention_v2 ani attentionv3. Nie tworzymy zależności od tych folderów. Wcześniejsza propozycja użycia lokalnego checkpointu SE jest wycofana.

Dobór modelu i danych przeprowadzamy od początku. Możemy korzystać z udokumentowanych bibliotek i znanych architektur. Zgodnie z decyzją użytkowniczki modele trenujemy od losowej inicjalizacji na 4 GPU. Nie korzystamy z publicznych wytrenowanych wag.
