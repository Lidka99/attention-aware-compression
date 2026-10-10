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

## 2026-09-30 — gdzie jesteśmy i co dalej

Powyższe wpisy są historią wcześniejszych ustaleń. Obecnie źródło wag nie jest
otwarte: obowiązuje trening od losowych wag, bez publicznych checkpointów.

**Zakończyliśmy etap 2.** E001 dostarczył EfficientNet-B0 z SE, wytrenowany
od zera przez 200 epok. Audyt best.pt odtworzył 3342/5000 odpowiedzi dev
(66,84%, epoka 177), bez różnic predykcji ani NLL. Cztery GPU oceniły po
1250 rozłącznych obrazów; wagi i bufory pozostały niezmienione. Raport:
[AUDYT_WYNIKI.md](../experiments/2026-09-26_E001_efficientnet_b0_od_zera/AUDYT_WYNIKI.md).

Krzywe pokazują mocne oznaki przeuczenia: minimum dev NLL w epoce 24,
późniejszy wzrost tej straty i prawie 100% trafności treningowej. To nie
jest powód do automatycznego treningu kolejnego modelu. Nie odtworzyliśmy
pełnego treningu, nie ocenialiśmy testu ani rezerwy, nie potwierdziliśmy
korzyści atencji dla kompresji.

**Najbliższy mały krok: przygotowanie protokołu etapu 3**, bez wykonywania
pomiaru w tej notatce. Pytanie: czy pary kanałów podobne przed SE są różnie
ważone zależnie od obrazu i czy różnice zmieniają ich podobieństwo po SE?

Proponowany zakres: jeden blok SE, zweryfikowany best.pt i niewielki,
zapisany podzbiór train. Dobór bloku opiszemy na podstawie budowy sieci,
przed obejrzeniem wyników kanałów. Podzbiór train ma służyć diagnostyce;
nie nazywamy go niezależną walidacją. Dokładna liczba obrazów, ID, seed,
reprezentacja aktywacji, miary i reguła wyboru par wymagają ustalenia
w protokole przed pomiarem. Cztery GPU mają przetwarzać rozłączne części
tej samej listy obrazów.

Zapiszemy h (mapy przed SE), a (wagi SE) i z=a*h (mapy po SE), a następnie
porównamy podobieństwo przed i po atencji oraz różnice wag między obrazami.
Ważna pułapka: cosinus map na pojedynczym obrazie nie zmienia się po
dodatnim przeskalowaniu kanałów. Dlatego taki pomiar sam nie wykryje
wpływu SE; potrzebna będzie jawna miara uwzględniająca skalę lub wspólna
reprezentacja wielu obrazów. Normy aktywacji i zerowe kanały też wymagają kontroli.

Wynik tej obserwacji może uzasadnić późniejsze próby łączenia, ale nie dowodzi
jeszcze ich bezpieczeństwa ani przewagi nowej metody. Kontrola oparta na
aktywacjach po SE pozostaje konieczna. Audyt i wykresy pozostają przy E001;
nowa próba obserwacyjna otrzyma własną kartę eksperymentu po określeniu zakresu.
Na razie nie uruchomiono etapu 3, kompresji ani kolejnego treningu.

Do małych zadań, takich jak przegląd dokumentacji lub kontrola punktów
odczytu aktywacji, używamy general-flash. Plan naukowy i weryfikacja wyników
pozostają odpowiedzialnością głównego agenta.

## 2026-10-01 — szkic pierwszej obserwacji SE

Przygotowano [E002](../experiments/2026-10-01_E002_obserwacja_se/OPIS.md),
wersja 0.1-draft. Wybrany blok: features.4.1, 480 kanałów wewnętrznych,
mapy 6×6 i SE 480→20→480. Wybór oparto na strukturze biblioteki, nie na
wynikach kanałów wytrenowanego modelu.

Proponujemy 1000 obrazów train (po 10/klasę), dwie rozłączne połowy A/B.
Na A wybierzemy 100 najbardziej podobnych par przed SE, a na B sprawdzimy
te same pary. Zapiszemy pełną tabelę wszystkich 114960 par, różnice siły
aktywacji oraz zmienność wag i ich stosunku. Nie użyjemy testu, rezerwy ani dev.
Szczegóły i prosty przykład: [WYJASNIENIE.md](../experiments/2026-10-01_E002_obserwacja_se/WYJASNIENIE.md).

Przegląd general-flash i przykłady algebraiczne NumPy na CPU potwierdziły,
że wspólna reprezentacja wielu obrazów może wykryć zmienne skalowanie,
którego nie widzi cosinus pojedynczych map. Nie wykonano obserwacji
checkpointu na obrazach CIFAR. To kontrola sposobu pomiaru, nie wynik hipotezy.

Następny krok: kod odczytu h/a/z, jego kontrole i zamrożenie protokołu.
Pomiar E002 pozostaje nieuruchomiony; nie rozpoczęto kompresji ani treningu.

## 2026-10-01 — E002 przygotowane do wykonania

Na prośbę użytkowniczki przygotowano odczyt SE na dokładnie 4 GPU i analizę
CPU. Zamrożona wersja: [PROTOKOL_v1.md](../experiments/2026-10-01_E002_obserwacja_se/PROTOKOL_v1.md).
Listy A/B zapisano przed inferencją. Miary, zerowe kanały, reguły remisów,
kształty i niezmienność hooków sprawdzono na sztucznych danych CPU;
wynik TESTY_CPU_v1.json to passed. Nie jest to obserwacja realnego modelu.
Główny agent odpowiada za odczyt i integrację; general-flash przygotował
czyste miary par i przejrzał kod. Teraz uruchomimy 4 rozłączne części po
250 obrazów, bez treningu, dev, rezerwy i testu. Status wykonania będzie
potwierdzony w manifestach i raporcie, także w przypadku błędu.

## 2026-10-01 — wynik pierwszego pomiaru E002

Pomiar zakończony: run-ID `20261001T051649Z_e0b3121_seed42_a8l9uuks`,
1000 unikalnych ID i cztery GPU po 250 obrazów, model niezmieniony.
Dowody i wykresy: [WYNIKI.md](../experiments/2026-10-01_E002_obserwacja_se/WYNIKI.md).

Na B mediana podobieństwa 100 wybranych par wynosi 0,9462 przed SE i
0,9579 po SE. Względne wagi są zmienne, lecz typowa zmiana cosinusa jest
dodatnia. Nie potwierdzamy więc scenariusza typowego rozdzielania tych
par przez SE ani przydatności wag atencji do kompresji. Wybrane pary
obejmują tylko 19 kanałów z 480. Kontrole ID, hashy i hooków przeszły;
general-flash sprawdził artefakty i przeliczył bezpośrednio trzy pary.

Proponujemy teraz sprawdzić, czy wysoki cosinus tej grupy nie wynika ze
wspólnej składowej stałej, na już zapisanych aktywacjach. To dalsza analiza
E002, jeszcze nie wykonana; nie przechodzimy do scalenia ani treningu.

## 2026-10-02 — kontrola stałego tła, przygotowanie

Przygotowano [dodatkową analizę E002](../experiments/2026-10-01_E002_obserwacja_se/analizy/2026-10-02_skladowa_stala/OPIS.md).
Rozdzielimy stałą średnią kanału, średnią zmienną między obrazami i wzory
przestrzenne wewnątrz map. Dwa kanały z mapami [9,11] i [11,9] mają
surowy cosinus około 0,98, ale po odjęciu tła 10 są przeciwne (cosinus -1).
To przykład syntetyczny, nie wynik wytrenowanego modelu.

General-flash sprawdził wzory i przygotował czyste miary NumPy; kontrole
CPU potwierdziły rozkład energii, centrowanie i obsługę kanałów zerowych.
Zastosujemy je do tych samych zapisanych map i par, bez nowego rankingu.
To analiza post hoc po obejrzeniu E002; nie niezależne potwierdzenie.
Wyłącznie CPU, bez checkpointu i nowych obrazów. Źródłowy odczyt nadal
pochodzi z czterech GPU, nie deklarujemy nowego wykorzystania kart.
Dotychczasowych artefaktów nie zmieniamy. Wynik tej analizy jeszcze
nie został obliczony, nie wykonujemy też kompresji ani treningu.

## 2026-10-02 — wynik kontroli stałej średniej

Analiza CPU zakończona: [raport](../experiments/2026-10-01_E002_obserwacja_se/analizy/2026-10-02_skladowa_stala/WYNIKI.md),
run-ID `20261002T055900Z_9dc2720_constant_kc4jiyub`. Na B dla 100 par
cosinus przed SE: surowy 0,9462, po odjęciu stałej średniej 0,9476,
po odjęciu średniej każdej mapy 0,9469. Udział stałej średniej przed SE
to medianowo tylko 0,61% energii wybranych 19 kanałów. Po SE stała ma
jeszcze mniejszy udział, a podobieństwo również utrzymuje się.

Nie ma więc potwierdzenia, że wysoki cosinus tej grupy wynika głównie ze
wspólnej średniej. Nie jest to jednak kontrola stałego przestrzennego
szablonu, który mógłby powtarzać się między obrazami. Tę różnicę zaznaczamy
w raporcie i proponujemy jako następny mały krok na istniejących mapach.
Nie wykonano nowej inferencji ani scalenia. Zachowano źródłowe aktywacje
i ich hashe; general-flash niezależnie sprawdził pierwsze trzy pary oraz
rozkład energii jednego kanału. Ta analiza post hoc nie dowodzi kompresji.

## 2026-10-07 — kontrola całego szablonu przestrzennego

Przygotowano [protokół i kod](../experiments/2026-10-01_E002_obserwacja_se/analizy/2026-10-07_szablon_przestrzenny/OPIS.md)
odjęcia średniej mapy po obrazach. Użyjemy tych samych map h/z i 100 par.
Rozkład rozróżnia stałą średnią, zmienną średnią obrazu, stały wzór
przestrzenny i zmiany wzoru zależne od obrazu. Wzory i kod sprawdził
general-flash; kontrole na sztucznych mapach CPU przeszły.

Wyjaśnienie komunikatu „no model, no GPU”: dotyczy kontroli technicznej
na sztucznych tablicach. Rzeczywiste aktywacje pochodzą z E002 wykonanego
na czterech GPU, po 250 obrazów. Teraz porównamy zapisane mapy na CPU;
nie trzeba ponownie wczytywać modelu ani przepuszczać obrazów przez sieć.
Nie oznacza to braku kart. Dowód kontroli kodu nie jest wynikiem badania
modelu; wyniki na rzeczywistych mapach powstaną po tej kontroli.
Nie wykonujemy nowej inferencji ani treningu. Po tej analizie zatrzymamy
się przed etapem 4 i zaproponujemy przygotowanie jednej próby scalenia.

## 2026-10-07 — wynik kontroli szablonu i domknięcie E002

Analiza rzeczywistych zapisanych map zakończona: [raport](../experiments/2026-10-01_E002_obserwacja_se/analizy/2026-10-07_szablon_przestrzenny/WYNIKI.md),
run-ID `20261007T140917Z_7388d23_template_qwki6pnw`. Na B mediana energii
szablonu wybranych kanałów wynosi 4,07% przed SE i 1,72% po SE.
Po odjęciu całej średniej mapy cosinus zmian odpowiedzi nadal jest wysoki:
0,9520 przed SE i 0,9591 po SE. Kontrole średniej i szablonu przemawiają
przeciw ich dominacji w podobieństwie tej grupy. To współzmienność na
badanych obrazach, nie dowód znaczenia dla klasyfikacji lub bezpiecznego scalenia.

Analiza na CPU po odczycie źródłowym na 4 GPU; bez nowego forwardu i bez
zmiany wag. Źródło i poprzednia analiza zachowały hashe. General-flash
sprawdził niezależnie pierwsze trzy pary i energie jednego kanału.
Domykamy zaplanowane kontrole E002; następny krok to przygotowanie
protokołu fizycznego scalenia jednej pary i pomiaru jego skutków.
Etapu 4 jeszcze nieuruchomiono, nie projektujemy nowego kryterium na
podstawie samych różnic wag atencji.

## 2026-10-09 — rzeczywisty eksperyment kompresji całej sieci

Użytkowniczka wybrała **10% parametrów całego modelu**, a nie redukcję
jednego kanału. Poprzednia propozycja pojedynczej pary staje się jedynie
wewnętrznym testem implementacji. Nie tworzymy dla niej osobnego wyniku badawczego.

[E003](../experiments/2026-10-09_E003_kompresja_10proc_parametrow/OPIS.md)
ma porównać jakość rzeczywistych mniejszych modeli przy wspólnym budżecie:
wybór par przed SE, po SE i losowy. Statycznie wyznaczono 15 szerokości,
usunięcie 1016 kanałów i 414224 parametrów: 4 135 648→3 721 424,
czyli 10,0159% mniej całej sieci. Budżet nie korzysta z jakości modelu.
To plan architektury, **nie wykonana jeszcze kompresja wytrenowanych wag**.

Wszystkie metody użyją tej samej kalibracji A, wspólnego operatora i zera
optymalizacji/rekalibracji BN. Dopasowanie alpha dotyczy wejścia avgpool
SE, beta pełnej mapy po SE; rzeczywisty moduł atencji działa po zmianie.
Nie deklarujemy nowej metody ani przewagi osobnych wag atencji. Następny
wynik ma obejmować top-1/NLL na dev, rozmiar i czas forwardu mniejszych sieci.
Forwardy i ewaluacja na dokładnie 4 GPU; statystyki i analiza mogą działać
na CPU. Nie używamy rezerwy ani testu; źródłowy checkpoint pozostaje niezmieniony.

Przegląd general-flash potwierdził budżet i strukturę operatora. Następny
krok: implementacja, kontrole kodu i uruchomienie E003 po zamrożeniu
działającego protokołu. Nie wracamy teraz do kolejnych analiz E002 ani
do treningu. Użytkowniczka upoważniła lokalne commity projektu/kodu/raportu E003.
