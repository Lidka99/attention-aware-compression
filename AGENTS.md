# Zasady pracy w projekcie

Pracujemy krok po kroku. Wyjaśnienia przekazujemy po polsku zarówno w rozmowie, jak i w plikach Markdown. Każdy konkretny eksperyment ma osobny folder w experiments/ z opisem i instrukcją odtworzenia.

Ustalenie 2026-09-29: wykresy i audyt istniejącego eksperymentu pozostają w jego folderze. Nowy numer E oznacza nową próbę badawczą; kolejne uruchomienia/audyty mają unikalne katalogi wyników pod runs/<E-ID>/. Zobacz docs/DOKUMENTOWANIE.md. Do drobnych, ograniczonych zadań używaj podagenta general-flash; bez równoległych zmian tych samych plików i konkurencyjnych zadań GPU.

## Uruchamianie i opis kodu

- Eksperymenty uruchamiamy na 4 GPU. Konfiguracja i komenda odtworzenia muszą jawnie określać użycie czterech GPU. Nie przechodzimy automatycznie na mniejszą liczbę urządzeń; jeśli cztery GPU nie są dostępne, zgłaszamy przeszkodę.
- Dobieramy sposób podziału pracy do eksperymentu: obliczenia rozproszone lub niezależne zadania na czterech GPU. Samo udostępnienie czterech urządzeń nie oznacza ich wykorzystania. Sposób podziału pracy opisujemy w protokole i sprawdzamy w logach.
- Dla reprodukowalności zapisujemy modele GPU, wersje środowiska, przypisanie zadań lub procesów do GPU, seedy i sposób agregacji wyników. Przy treningu zapisujemy też batch size na GPU i efektywny globalny batch size, uwzględniając akumulację gradientów. Cztery GPU nie oznaczają automatycznie czterech niezależnych powtórzeń.
- Każdy plik z kodem eksperymentu zaczyna się krótkim komentarzem lub docstringiem: co robi, jakiego eksperymentu dotyczy, jakie dane przyjmuje i jakie wyniki zapisuje. Zawiera też odnośnik do pełnego OPIS.md.

E001 i E002 ukończono. Od 2026-10-09 użytkowniczka wybrała rzeczywistą kompresję co najmniej 10% parametrów całej sieci (E003), nie wynik oparty na jednym kanale lub jednym bloku. Pojedynczy kanał służy tylko kontroli implementacji. Przygotowany projekt E003 porównuje PRE/POST/RANDOM ze wspólnym budżetem i operatorem; kompresji jeszcze nie wykonano. Używamy środowiska Conda attention, bez publicznych wag. Bieżący status: AKTUALNY_PLAN.md.

## Daty w nazwach eksperymentów

Folder każdego nowego eksperymentu nazywamy `RRRR-MM-DD_E001_krotka_nazwa`, np. `2026-09-25_E001_nazwa_eksperymentu`. Jest to tylko przykład formatu, nie utworzony eksperyment. Data oznacza dzień utworzenia protokołu (UTC), a E001 jest stałym, kolejnym identyfikatorem. Pełną nazwę umieszczamy też w tytule OPIS.md i w opisie kodu. Nie zmieniamy daty w nazwie przy ponownym uruchomieniu; każde uruchomienie ma osobno zapisany znacznik czasu UTC.

## Niezależny projekt — ustalenie 2026-09-26

Eksperymenty piszemy od nowa w tym repozytorium. Nie kopiujemy ani nie importujemy kodu, konfiguracji, wag czy wyników z attention_v2 ani attentionv3. Nie tworzymy zależności od tych folderów. Wcześniejsza propozycja użycia lokalnego checkpointu SE jest wycofana.

Dobór modelu i danych przeprowadzamy od początku. Możemy korzystać z udokumentowanych bibliotek i znanych architektur. Zgodnie z decyzją użytkowniczki modele trenujemy od losowej inicjalizacji na 4 GPU. Nie korzystamy z publicznych wytrenowanych wag.
