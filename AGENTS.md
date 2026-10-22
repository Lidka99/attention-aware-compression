# Zasady pracy w projekcie

Pracujemy krok po kroku. Wyjaśnienia przekazujemy po polsku zarówno w rozmowie, jak i w plikach Markdown. Każdy konkretny eksperyment ma osobny folder w experiments/ z opisem i instrukcją odtworzenia.

Ustalenie 2026-09-29: wykresy i audyt istniejącego eksperymentu pozostają w jego folderze. Nowy numer E oznacza nową próbę badawczą; kolejne uruchomienia/audyty mają unikalne katalogi wyników pod runs/<E-ID>/. Zobacz docs/DOKUMENTOWANIE.md. Do drobnych, ograniczonych zadań używaj podagenta general-flash; bez równoległych zmian tych samych plików i konkurencyjnych zadań GPU.

## Uruchamianie i opis kodu

- Eksperymenty uruchamiamy na 4 GPU. Konfiguracja i komenda odtworzenia muszą jawnie określać użycie czterech GPU. Nie przechodzimy automatycznie na mniejszą liczbę urządzeń; jeśli cztery GPU nie są dostępne, zgłaszamy przeszkodę.
- Dobieramy sposób podziału pracy do eksperymentu: obliczenia rozproszone lub niezależne zadania na czterech GPU. Samo udostępnienie czterech urządzeń nie oznacza ich wykorzystania. Sposób podziału pracy opisujemy w protokole i sprawdzamy w logach.
- Dla reprodukowalności zapisujemy modele GPU, wersje środowiska, przypisanie zadań lub procesów do GPU, seedy i sposób agregacji wyników. Przy treningu zapisujemy też batch size na GPU i efektywny globalny batch size, uwzględniając akumulację gradientów. Cztery GPU nie oznaczają automatycznie czterech niezależnych powtórzeń.
- Każdy plik z kodem eksperymentu zaczyna się krótkim komentarzem lub docstringiem: co robi, jakiego eksperymentu dotyczy, jakie dane przyjmuje i jakie wyniki zapisuje. Zawiera też odnośnik do pełnego OPIS.md.

E001–E010 ukończono. E003/E004: przy 10% redukcji wybór kanałów nie dał przewagi po równym dostrajaniu. E005: baseline 96×96, dev 51,53%. E006: bramki SE nie przewidują szkody ponad s_post; wątek zamknięty. E007: baseline 224×224, dev 58,49%. E008 (protokół 1.1): bez naprawy selekcja > losowa, ale POST < PRE; po 60 epokach POST−PRE +0,51 pp (CI [−0,18; +1,19]), BASE 59,45%. E009 (120 epok): BASE 60,04 / POST 59,46 / PRE 59,18 / RANDOM 59,21; POST−PRE +0,28 pp (CI [−0,41; +0,96]) — przewaga nie występuje. E010 (9 kryteriów, 2 datasety): tylko TAYLOR przeszedł primary (Tiny 30%, T=5: +1,03 pp vs RANDOM, Holm OK), ale nie zreplikował się na CIFAR-100 (+0,36 pp, CI obejmuje 0) → `survivor_not_replicated`; następny krok to decyzja o E011. Używamy środowiska Conda attention, bez publicznych wag i bez zmiany wcześniejszych wyników.

## Daty w nazwach eksperymentów

Aktualizacja kierunku 2026-10-22 (rozmowa z użytkowniczką): następny etap to odtworzenie foldingu z *Cut Less, Fold More*, a potem porównanie grupowania bez fine-tuningu. Nie traktować wcześniejszej propozycji 3 seedów TAYLORA jako uzgodnionego E011. Audyt: `literature/AUDYT_CUT_LESS_FOLD_MORE.md`. Obecnie brak zweryfikowanego checkpointu CIFAR-10/ResNet18; status przygotowania, bez nowego biegu. Brak istotności różnic w E009 nie dowodzi równoważności metod. E010 ma dodatnie wyniki lokalne, bez replikacji głównego kandydata.

Folder każdego nowego eksperymentu nazywamy `RRRR-MM-DD_E001_krotka_nazwa`, np. `2026-09-25_E001_nazwa_eksperymentu`. Jest to tylko przykład formatu, nie utworzony eksperyment. Data oznacza dzień utworzenia protokołu (UTC), a E001 jest stałym, kolejnym identyfikatorem. Pełną nazwę umieszczamy też w tytule OPIS.md i w opisie kodu. Nie zmieniamy daty w nazwie przy ponownym uruchomieniu; każde uruchomienie ma osobno zapisany znacznik czasu UTC.

## Niezależny projekt — ustalenie 2026-09-26

Eksperymenty piszemy od nowa w tym repozytorium. Nie kopiujemy ani nie importujemy kodu, konfiguracji, wag czy wyników z attention_v2 ani attentionv3. Nie tworzymy zależności od tych folderów. Wcześniejsza propozycja użycia lokalnego checkpointu SE jest wycofana.

Dobór modelu i danych przeprowadzamy od początku. Możemy korzystać z udokumentowanych bibliotek i znanych architektur. Zgodnie z decyzją użytkowniczki modele trenujemy od losowej inicjalizacji na 4 GPU. Nie korzystamy z publicznych wytrenowanych wag.
