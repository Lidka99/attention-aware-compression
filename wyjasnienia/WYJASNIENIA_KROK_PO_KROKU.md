# Wyjaśnienia krok po kroku

Ten plik zawiera proste objaśnienia kolejnych kroków badań. Uzupełniamy go w miarę pracy, bez planowania wszystkiego naraz. Szczegółowe protokoły i wyniki poszczególnych eksperymentów zapisujemy osobno.

## Krok 1 — co chcemy sprawdzić? (2026-09-25)

Zaczynamy od pytania, a dopiero później wybierzemy model, dane i napiszemy kod.

**Czy atencja inaczej traktuje kanały, które mają podobne aktywacje przed atencją?**

To propozycja pierwszej obserwacji diagnostycznej. Nie jest jeszcze gotowym protokołem ani potwierdzoną luką badawczą. Dotychczasowe karty eksperymentów pozostają szkicami.

### 1. Co to jest kanał?

Warstwa splotowa przetwarza obraz i tworzy wiele map liczb. Jedna taka mapa to kanał wyjściowy. W prostym przybliżeniu pokazuje, gdzie dany filtr zareagował na obraz. Nie zakładamy, że każdy kanał ma łatwą do nazwania rolę, taką jak wykrywanie oczu czy krawędzi.

### 2. Co to są aktywacje?

To liczby, które sieć oblicza dla konkretnego obrazu. Dwa kanały mogą dawać podobne mapy liczb. Taka obserwacja sugeruje, że warto sprawdzić możliwość ich połączenia, ale sama podobność nie dowodzi, że połączenie będzie bezpieczne.

### 3. Co robi atencja kanałowa?

W rozważanym prostym przypadku moduł atencji przypisuje każdemu kanałowi wagę i mnoży przez nią jego mapę. Wagi mogą zależeć od obrazu.

Oznaczamy to jako: `z = a * h`, gdzie:

- `h` to mapa kanału przed atencją;
- `a` to waga przypisana temu kanałowi;
- `z` to mapa po przemnożeniu przez wagę.

Przykład dla pojedynczej liczby: dwa kanały mają przed atencją wartości 10 i 10. Jeżeli dostaną wagi 0,9 i 0,1, po atencji otrzymamy 9 i 1. To ilustracja rachunku, a nie wynik eksperymentu.

### 4. Dlaczego interesuje nas to przy kompresji?

Kompresja ma prowadzić do mniejszej lub tańszej obliczeniowo sieci. Jedna z możliwości to łączenie kanałów. Chcemy sprawdzić, czy podobieństwo przed atencją wystarcza do wskazania kandydatów do takiego łączenia.

Samo mnożenie kanałów przez małe wagi nie zmniejsza liczby obliczeń już wykonanej warstwy splotowej. Aby wykazać kompresję, później trzeba rzeczywiście zmniejszyć sieć i zmierzyć skutki.

### 5. Co pokaże pierwsza obserwacja?

Sprawdzimy, czy kanały podobne przed atencją dostają podobne wagi na różnych obrazach. Najpierw musimy ustalić, jak mierzymy podobieństwo i na jakich danych.

Jeśli znajdziemy różnice wag, nie będzie to jeszcze dowód, że atencja poprawia wybór kanałów do połączenia. Nie będzie to również dowód innowacyjności metody. Później potrzebne będą porównania, między innymi z wyborem opartym bezpośrednio na aktywacjach po atencji.

## Jak będziemy opisywać każdy eksperyment?

Każdy eksperyment otrzyma osobną kartę Markdown zawierającą:

1. **Pytanie:** czego chcemy się dowiedzieć?
2. **Hipotezę:** jakiego wyniku się spodziewamy i dlaczego? To przypuszczenie, nie fakt.
3. **Materiały:** dokładny model, zapisane wagi, dane i identyfikatory użytych obrazów.
4. **Procedurę:** co robimy i w jakiej kolejności?
5. **Pomiar i kryterium oceny:** jakie liczby zapisujemy i jak zinterpretujemy wynik? Ustalamy to przed właściwym pomiarem.
6. **Instrukcję odtworzenia:** wersję kodu, środowisko, konfigurację i dokładną komendę uruchomienia.
7. **Wynik i wniosek:** co wyszło, czego możemy się dowiedzieć i czego wynik nie rozstrzyga. Zapisujemy także nieudane próby.

Przed uruchomieniem uzupełnimy wszystkie wymagane pola. Po uruchomieniu dopiszemy wynik, zachowując pierwotne założenia.

## Co znaczy „reprodukowalny”?

Inna osoba — albo my za kilka miesięcy — powinna móc użyć tej samej wersji kodu, modelu i danych, wykonać opisane kroki oraz uzyskać zgodny wynik w zadeklarowanej tolerancji.

Git zapisuje historię kodu i opisów. Sam nie wystarcza: potrzebne są również wersje bibliotek, dostęp do dokładnych wag i danych, ustawienia uruchomienia oraz ziarna losowości. Ziarno to liczba sterująca generowaniem liczb pseudolosowych; pomaga powtarzać losowania, ale nie gwarantuje identycznych wyników na każdym sprzęcie.

## Najbliższy mały krok

Ustalimy, co w pierwszym pomiarze oznaczają **podobne kanały**. Wyjaśnimy sposób pomiaru na prostym przykładzie, a następnie wpiszemy ustalenie do karty eksperymentu. Na tym etapie nie uruchamiamy jeszcze eksperymentów.

## Krok 2 — co oznaczają podobne kanały? (2026-09-25)

Na początek rozważamy podobieństwo wzorca aktywacji, niezależnie od jego skali. Przykład dla jednego obrazu: kanał A daje liczby [1, 2, 3], a kanał B [2, 4, 6]. Kanał B reaguje dwa razy mocniej, ale proporcje między pozycjami są takie same.

Proponowany pierwszy pomiar to podobieństwo cosinusowe. Układamy liczby z mapy kanału w listę, czyli wektor. Dla dwóch niezerowych wektorów wynik 1 oznacza ten sam kierunek: jeden jest dodatnią wielokrotnością drugiego. Wynik 0 oznacza prostopadłość wektorów, a nie ogólnie brak jakiegokolwiek związku. Przed aktywacją ograniczającą wartości do nieujemnych możliwe są również wyniki ujemne. Pary z zerową mapą wymagają osobnej reguły, bo standardowy wzór jest wtedy nieokreślony.

Dlaczego rozważamy taki pomiar? Szukamy kanałów reagujących w podobnych miejscach, nawet jeśli siła reakcji jest inna. To punkt startowy do diagnostyki, nie dowód ich wymienności ani ostateczna metoda doboru kanałów do kompresji. Pomiar pomija skalę, więc dodatkowo trzeba będzie zapisywać wielkość aktywacji.

Ważna pułapka: dodatnia skalarna waga atencji nie zmienia kierunku wektora kanału. Dlatego podobieństwo cosinusowe dwóch map na pojedynczym obrazie nie wykryje samego różnego przeskalowania ich przez atencję. Przed atencją [1, 2, 3] i [2, 4, 6] mają podobieństwo 1; po przemnożeniu przez 0,9 i 0,1 nadal mają podobieństwo 1. Różnice wag atencji trzeba mierzyć osobno. Nie będziemy traktować zmiany cosinusa po takim skalowaniu jako testu wpływu atencji.

Plan obserwacji: na wielu tych samych obrazach zmierzymy podobieństwo map przed atencją i osobno różnice wag przypisanych kanałom. Zanim uruchomimy pomiar, ustalimy sposób podsumowania wyników między obrazami, obsługę zerowych map i regułę wyboru par. Nie ustalamy jeszcze arbitralnego progu „podobieństwa”.

Następny krok: wybrać jeden model z atencją i dostępne wytrenowane wagi. Dopiero znając jego budowę, określimy dokładną warstwę i miejsca zapisu aktywacji. Nie rozpoczynamy jeszcze treningu ani kompresji.

## Ustalenie — 4 GPU i opis kodu (2026-09-25)

Każdy eksperyment ma wykorzystywać 4 GPU. W opisie zapiszemy, jak dzielimy między nie pracę, a dokładna komenda pozwoli odtworzyć takie uruchomienie. Cztery GPU mogą wspólnie wykonywać jedno zadanie albo realizować niezależne zadania; wybierzemy sposób odpowiedni do konkretnego eksperymentu. Nie zmniejszamy automatycznie liczby GPU, jeśli urządzenia są niedostępne.

Na początku każdego pliku z kodem eksperymentu umieścimy krótki opis: co robi, jakie dane przyjmuje i jakie wyniki zapisuje, oraz odnośnik do pełnego opisu eksperymentu. Na razie zapisujemy tylko wymagania.

## Daty w nazwach eksperymentów

Folder każdego nowego eksperymentu nazywamy `RRRR-MM-DD_E001_krotka_nazwa`, np. `2026-09-25_E001_nazwa_eksperymentu`. Jest to tylko przykład formatu, nie utworzony eksperyment. Data oznacza dzień utworzenia protokołu (UTC), a E001 jest stałym, kolejnym identyfikatorem. Pełną nazwę umieszczamy też w tytule OPIS.md i w opisie kodu. Nie zmieniamy daty w nazwie przy ponownym uruchomieniu; każde uruchomienie ma osobno zapisany znacznik czasu UTC.

## 2026-09-26 — zaczynamy od modelu do pierwszego pomiaru

> Nieaktualna propozycja: opisany poniżej pomysł użycia checkpointu z attention_v2 został wycofany. Obowiązuje niezależny projekt opisany w kolejnym wpisie.

Pierwszy krok to wybranie jednej istniejącej sieci z atencją i sprawdzenie, czy możemy wiarygodnie odtworzyć jej działanie.

Model to konstrukcja sieci. Wagi to liczby, których sieć nauczyła się podczas treningu. Checkpoint to plik zapisujący te wagi, czasem również inne informacje o treningu. Gotowy checkpoint pozwala badać wytrenowaną sieć bez rozpoczynania treningu od zera.

W projekcie znaleziono kod ResNet dla CIFAR z obsługą atencji SE oraz plik attention_v2/results_publication_cifar10/se_b0.2_seed42.pth. To kandydat do pierwszego pomiaru, jeszcze nie zatwierdzony model. Sama nazwa pliku nie potwierdza jego zawartości ani jakości. Ogólna konfiguracja cifar10_resnet18.yaml wskazuje atencję static, więc nie można bez sprawdzenia przypisać jej do checkpointu SE.

Najbliższe zadanie to sprawdzić pochodzenie tego checkpointu: jaki dokładnie model wytrenowano, na jakich danych, z jakim przetwarzaniem obrazów i jaki wynik wcześniej uzyskano. Jeżeli tych informacji nie da się odtworzyć, wybierzemy lepiej udokumentowany punkt startowy.

Pierwszy właściwy eksperyment powinien potwierdzić, że potrafimy odtworzyć działanie niezmodyfikowanej sieci. To punkt odniesienia: bez niego nie będziemy wiedzieć, czy późniejszy spadek jakości wynika z kompresji, czy z błędu w uruchomieniu. Dopiero potem zmierzymy podobieństwo kanałów i zachowanie atencji.

Dziś sprawdzono wyłącznie pliki i dokumentację; nie wczytywano wag ani nie uruchamiano eksperymentu. Przyszłe eksperymenty będą miały datę w nazwie, osobny folder, opis i instrukcję odtworzenia, krótkie objaśnienie w kodzie oraz wykorzystają 4 GPU.

## Niezależny projekt — ustalenie 2026-09-26

Eksperymenty piszemy od nowa w tym repozytorium. Nie kopiujemy ani nie importujemy kodu, konfiguracji, wag czy wyników z attention_v2 ani attentionv3. Nie tworzymy zależności od tych folderów. Wcześniejsza propozycja użycia lokalnego checkpointu SE jest wycofana.

Dobór modelu i danych przeprowadzamy od początku. Możemy korzystać z udokumentowanych bibliotek i publicznych wag po ustaleniu ich źródła i wersji; pisanie nowych eksperymentów nie oznacza automatycznie konieczności trenowania modelu od zera. Decyzja o źródle wag pozostaje otwarta.

Następny mały krok: ustalić jeden model z atencją i jeden zbiór danych dla nowego projektu. Pierwszy eksperyment sprawdzi działanie modelu przed kompresją, aby późniejsze zmiany miały wiarygodny punkt odniesienia. Nadal obowiązują 4 GPU, daty w nazwach, osobne foldery eksperymentów, opisy w kodzie oraz instrukcje odtworzenia.
