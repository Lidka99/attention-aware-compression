# Alokacja budżetu w foldingu — plan publikacji

## Po E045: zachowanie jakości przy łagodnym budżecie — 2026-11-22

[E045](experiments/2026-11-22_E045_lagodny_uniform_tiny/WYNIKI.md) dodaje
na Tiny **dev** krzywą 0/10/20/30/50% dla UNIFORM przy tej samej CMA.
Pełny 51,2200%; b10 46,3625% (−4,8575 pp, 10,35% redukcji P i 10,67%
MAC), b20 40,5800% (−10,6400 pp). W [szkicu 0.6](SZKIC_ARTYKULU.md)
krzywa jest kontekstem jakości bezwzględnej, obok E043/E044 z osobnymi
rolami danych. Nie stosujemy 50% jako obowiązkowego budżetu.

Następne pytanie: czy jedna zamrożona niejednolita alokacja może zmniejszyć
stratę w b10 pod rzeczywistymi pułapami P/MAC UNIFORM z E045? Zacząć od
protokołu i kontroli CPU. Oryginalne ograniczenie 90% szerokości wymuszałoby
przynajmniej 19,21% redukcji P; szersze grupy to jawna zmiana warunków
projekcji, a nie wynik wcześniej wykazany dla L06.

## Po E044: transfer do istniejącego Tiny — 2026-11-22

[E044](experiments/2026-11-22_E044_l06_tiny/WYNIKI.md) daje **+2,100 pp
(4/4)** przy b50 i **+0,255 pp (3/4)** przy b30 na Tiny. Oba koszty nie
przekraczają U. Tiny nie uczestniczył w wyborze L06, ale val był używany
w E035. W [szkicu](SZKIC_ARTYKULU.md) raportujemy transfer obok E043,
z osobnymi rolami danych i jawną stratą do pełnego modelu. Następny krok:
redakcja tezy i figur pod format artykułu; L06 pozostaje referencją dla
ewentualnego osobnego badania selektora.

## Po E043 — 2026-11-22

[E043](experiments/2026-11-21_E043_l06_potwierdzenie_transfer/WYNIKI.md)
potwierdza zamrożony L06 na odłożonych obrazach CIFAR-100 przy ~50% redukcji
P (**+1,215 pp vs U, 3/4**), pod oboma pułapami kosztu. Przy ~30% kierunek
pozostaje dodatni (**+0,945 pp, 4/4**), lecz nie spełnia progu +1 pp.
W raporcie rozdzielić pilot E042 na dev od jednorazowego E043; to te same
cztery checkpointy, a nie nowe treningi ani sprawdzony automatyczny dobór.
Rezerwa CIFAR-100 jest już użyta. Ewentualny selektor porównać z L06 jako
tanią referencją, na innych niewidzianych danych w osobnym protokole.

## Po E042 — 2026-11-21

[E042](experiments/2026-11-21_E042_zapas_pod_uniform/WYNIKI.md) pokazuje
**zapas przy P i MAC nie większych niż w UNIFORM**: najlepszy z 24 profili
na dev daje +1,175 pp (3/4) na CIFAR-100 i +1,475 pp (4/4) na kontrolnym
CIFAR-10. Mediana na CIFAR-100 to −1,265 pp. W obu zbiorach wygrało to
samo losowe przesunięcie L06 przed projekcją; wykorzystano oba zbiory do
wyboru, więc nie jest to niezależny transfer.

**Najbliższe pytanie:** czy dobry profil można z góry wybrać i przenieść?
Zaprojektować jeden osobny protokół, w którym punktem odniesienia jest
**zamrożony L06**: te same przesunięcia `k/n` w kolejności 12 grup,
rzutowane na pułapy U przy nowym budżecie, np. 30% redukcji P na tym samym
ResNet18. Podobna trafność L06 na nowych danych bez selekcji poparłaby wkład
o transferze konkretnej reguły. Wkład o **automatycznym doborze** wymaga
dodatkowo wyniku jednej z góry ustalonej reguły wybierającej po odrębnej
kalibracji; porównać ją z zamrożonym L06, U, T-proj i losowaniem przy
równym limicie ocen oraz obu pułapach. Wygraną z U oddzielić od wygranej
nad samą (tanią) regułą L06. Najlepszy profil z dev E042 jest punktem
diagnostycznym, a nie testem tej nowej reguły.

Przed potwierdzeniem zamrozić budżety, kluczowy kontrast, ogólną regułę,
próg praktyczny, zbiór i skrót wejść. Testy E035 są użyte. Rezerwa
CIFAR-100 jest kandydatem do osobnego, jednorazowego pomiaru po sprawdzeniu
rozłączności. Transfer do innej rodziny architektur albo receptury treningu
to następny etap mocniejszej pracy. Dotychczasowy [szkic](SZKIC_ARTYKULU.md)
zawiera figury i wynik E042 jako **eksploracyjną** kontrolę; materiał z
niezależnego E035 nadal stanowi główny wynik empiryczny.

## Projekt poprzedzający E042 — 2026-11-21

**Mamy wynik o alokacji przy stałym P, ale nie mamy sprawdzonego
optymalizatora.** [E035](experiments/2026-11-07_E035_ocena_niezalezna/WYNIKI.md)
potwierdza poprawę po CMA w 15/15 konfiguracji; koszt z E036/E037
i mieszany E041 ograniczają zakres tezy. Pytanie o optymalizację pochodzi
wprost z [Wang i in., s. 13](https://arxiv.org/pdf/2502.10216v2#page=13),
a [CLFM, rys. 1 i 8](https://www.alphaxiv.org/abs/2602.18116v1) daje
punkt odniesienia z jednolitym ratio. Liniowy profil ma precedensy (ATP,
Linear Decay), a sam dobór szerokości pod koszt ma precedensy poza
foldingiem (m.in. [BALF, §4.3 i §5](https://arxiv.org/pdf/2509.25136v4#page=9)
dla rang pod P **lub** FLOP).

**Teraz: domknąć studium empiryczne na istniejących wynikach.** Włączyć
E041 do głównej narracji i pokazać rzeczywiste szerokości, top1 oraz stratę
względem nieskompresowanego modelu. Zestawić jakość, P i MAC dla U/T/U'
wyłącznie na wspólnym dev; osobno pokazać niezależne U/T z E035. Udostępnić
zamrożone konfiguracje, wyniki per seed i instrukcję odtworzenia. To
materiał na wąską pracę empiryczną; nowość i format ocenić na gotowym szkicu.

**Jeśli celem jest mocniejsze rozwiązanie problemu: najpierw zmierzyć
rezerwę jakości pod dwoma pułapami.** Poniżej projekt decyzji badawczej,
nie zatwierdzony protokół ani kontynuacja bramki E040/E041.

1. **Pytanie:** czy zmiana całkowitoliczbowych szerokości legalnych grup kanałów
   poprawia top1 po foldingu i tej samej BN przy `P(k) ≤ P(U)` oraz
   `MAC(k) ≤ MAC(U)`? Zacząć od ResNet18/50% redukcji P na CIFAR-100
   i zachować CIFAR-10 jako z góry ustalony zbiór kontrolny; istniejące
   cztery checkpointy na zbiór, wyłącznie dev. Pułapy pochodzą z UNIFORM,
   więc są niższe obliczeniowo niż w E041, gdzie dopasowano kontrole do
   TILT. Licznik obu kosztów i przykładowe wykonalne profile sprawdzono w
   [E040](experiments/2026-11-09_E040_interakcje_po_bn/WYNIKI.md#budżety-i-rzeczywiste-rozmiary);
   jego bramka predykcji nie rozstrzygnęła jakości top1.
2. **Mały pomiar bez modelu zastępczego:** na CPU zamrozić wspólną dla
   seedów pulę (np. do 24 różnych profili na zbiór), wykorzystując różne
   przesunięcia szerokości między grupami oraz ustaloną próbę losową.
   Wyliczyć P i MAC z geometrii całej sieci, sprawdzić zgodność z pułapami
   i połączeniami residualnymi oraz odnotować wykorzystanie budżetów.
   Ocenić całe modele po identycznym CMA, z U i deterministyczną
   projekcją zamrożonego TILT pod te same limity jako referencjami. Ustalić
   regułę projekcji, tolerancje, liczbę prób i kryteria przed pomiarem.
   Najlepszy **jeden** profil wg średniej na dev mierzy zapas w tej puli;
   jego wynik nie jest oceną alokatora ani globalnym optimum.
3. **Decyzja po pilocie:** przykładowa bramka przejścia to zysk średni
   ≥1 pp względem mocniejszej z zamrożonych referencji na CIFAR-100
   i poprawa w ≥3/4 sparowanych checkpointów, przy dotrzymanych obu
   pułapach. Pokazać też wynik CIFAR-10, wielkość straty wobec modelu
   nieskompresowanego i wszystkie rzeczywiste koszty. Jeśli sygnał jest
   za mały, zakończyć próbę na tej puli i pozostać przy studium. Jeśli
   jest wyraźny, sprawdzić **jedną** regułę doboru na wydzielonych danych
   kalibracyjnych i porównać z losowaniem + BN przy takim samym limicie
   ocen (precedens [EagleEye, §3.4](https://arxiv.org/pdf/2007.02491v2#page=9),
   tu adaptacja bez końcowego fine-tuningu) oraz z MAG2 przy tych samych
   szerokościach. Jeśli teza dotyczy całego pipeline'u, pozwolić MAG2
   także dobrać własny profil pod te same pułapy i dodać porównywalną
   metodę z literatury bez fine-tuningu, np. BALF. Zysk po zmianie solvera
   lub limitu prób sam nie wystarcza.
4. **Mocny wynik do zgłoszenia:** praktyczna przewaga nad najlepszą
   wykonalną referencją przy obu limitach, potwierdzona dla zamrożonej
   metody na niewidzianych danych oraz na dodatkowej architekturze lub
   reżimie treningu, także przy umiarkowanej redukcji. Z góry ustalić
   minimalny efekt i akceptowalną stratę względem modelu oryginalnego,
   liczebność niezależnych checkpointów i raportowanie niepewności
   różnic sparowanych. Jeśli wniosek dotyczy wdrożenia, zmierzyć również
   latencję, przepustowość i pamięć; osobno podać koszt samego doboru.
   Testy E035 są użyte. Rezerwa CIFAR-100 pozostaje zamknięta do osobnej
   decyzji i sprawdzenia rozłączności z treningiem, kalibracją i doborem.

Pilot wykonano jako E042 na istniejących checkpointach. E040, E041
i wcześniejsze przeszukiwanie mają zamknięte werdykty.

## Rozstrzygnięcie E041 — 2026-11-19

**Dodatkowa kontrola przed artykułem wykonana: E041.**
[Wyniki](experiments/2026-11-15_E041_kontrola_p_mac/WYNIKI.md): przy P/MAC
dopasowanych≤0,1% TILT−średnia4kontroli wynosi po CMA C10+0,11250pp(2/4),
C100+2,10375pp(4/4). Łączne primary nie przeszło przez CIFAR-10;
jedna kontrola naC10 wygrywa z TILT na4/4seedach. To zależność od zbioru,
nie ogólna przewaga ani nowy optymalizator. Audyt PASS,4GPU/exit0.

Tiny od początku nie wchodził do tego pilota. Dodatkowy pomiar na nim
byłby osobnym rozszerzeniem i nie zmieni wyniku bramki obu CIFAR-ów.
Zgodnie z protokołem E041 nie dokładamy wariantów ratunkowych.

Następny krok: ująć E041 jako kontrolę ograniczającą twierdzenia w wąskim
studium empirycznym, ocenić cały szkic i format publikacji. Nie traktować
wyników jako mocnego nowum metodologicznego; liniowy profil ma precedensy.

## Etap przygotowania — 2026-11-15

**Rozpoczęto [polski szkic artykułu](SZKIC_ARTYKULU.md): studium empiryczne
liniowej alokacji w foldingu, kosztu oraz zależności od operatora.** Szkic
zawiera abstrakt, trzy tezy, metodę, tabelę E035, wyniki kontroli
i ograniczenia. To początek redakcji, nie wersja gotowa do zgłoszenia.

Przegląd wskazał bezpośrednie precedensy: ATP, równanie (10), s.8
([2502.14770v1](https://www.alphaxiv.org/abs/2502.14770v1)) oraz Linear Decay,
dod.C.2, s.14 ([2009.11094v2](https://www.alphaxiv.org/abs/2009.11094v2)).
Nie przedstawiać TILT jako wynalezienia liniowego profilu. Nasz zakres
to adaptacja do grup foldingu i kontrolowane zbadanie jej kompromisów;
pierwszeństwo tej adaptacji nie zostało ustalone.

**Doprecyzowanie pytania użytkowniczki:** najpierw rozstrzygnąć, czy przed
artykułem potrzebny jest dodatkowy eksperyment. Szkic jest materiałem do
tej oceny, nie zatwierdzeniem dalszej redakcji. Kandydatem do dyskusji
jest kontrola różnych profili przy jednocześnie dopasowanych P i MAC;
jej wykonalność wymaga sprawdzenia geometrii i osobnego protokołu.

Najbliższe zadania redakcyjne: publikacyjne profile U/T, odniesienie do modelu oryginalnego,
wspólne figury jakość–P/MAC na dev oraz dopracowanie metody i related work.
Po ich złożeniu ocenić cały szkic i adekwatny format publikacji. Gotowość
do pisania nie oznacza potwierdzenia mocnego nowum metodologicznego.

## Decyzja po E040 — 2026-11-11

**Wracamy do pełnego szkicu artykułu na E035–E039.**
[E040](experiments/2026-11-09_E040_interakcje_po_bn/WYNIKI.md) zakończył
ograniczoną próbę mocniejszej metody: model par po BN pogorszył predykcję
szkody na obu CIFAR-ach,0/4seedów z poprawą na każdym; stop przed dev.
Audyt PASS,4GPU/exit0, bez nowego treningu, testu i rezerwy.
Nie wykazano mocniejszego wkładu metodologicznego. Zgodnie z ustaloną regułą
nie dokładamy wariantów ani eksperymentów ratunkowych. E040 można krótko
przedstawić jako negatywną kontrolę, bez wniosku o jego trafności top1.
Zakres twierdzeń TILT i wyniki E035–E039 pozostają bez zmian.

**Ocena wkładu i figur:** [audyt2026-11-11](OCENA_WKLADU_I_ILUSTRACJI.md).
Wkład empiryczny jest rozpoznawalny, ale przyrostowy; gotowość do napisania
szkicu nie jest potwierdzeniem wystarczającej nowości publikacyjnej.
Istnieją figury wynikowe, lecz brakuje schematu procedury, faktycznych profili
alokacji i wspólnej prezentacji jakość–P/MAC; potrzebne też odniesienie do
teacher i redakcja publikacyjna. Te braki można uzupełnić bez nowych pomiarów.

## Poprzednia decyzja projektowa — 2026-11-09

Użytkowniczka zmieniła kolejność: **najpierw ocena jednej propozycji mocniejszej
metody i ograniczonego pilota**, przed pełną redakcją artykułu.
[Notatka projektowa](NOTATKA_PROJEKTOWA_2026-11-09.md) oraz
[E040 — projekt pilota, nieuruchomiony](experiments/2026-11-09_E040_interakcje_po_bn/OPIS.md).
Obecny etap obejmuje literaturę i projekt, bez GPU i treningu. Wynik negatywny
lub brak potwierdzenia nowego składnika przywraca poniższy plan redakcji.
Dotychczasowe wyniki i zakres twierdzeń pozostają bez zmian.

## Plan redakcji z 2026-11-08 — przywrócony po rozstrzygnięciu E040

**Teraz spisujemy wyniki i przygotowujemy pierwszą pełną wersję artykułu.**
Materiał E035–E039 wystarcza do zbudowania spójnej pracy empirycznej.
Rozpoczęcie pisania nie oznacza jeszcze gotowości do zgłoszenia ani
potwierdzenia pierwszeństwa metody. Kolejne braki oceniamy na gotowym szkicu.

## Co jest podstawą tej decyzji?

| Dowód | Znaczenie dla artykułu |
|---|---|
| [E035](experiments/2026-11-07_E035_ocena_niezalezna/WYNIKI.md) | Zamrożony TILT poprawia top1 względem UNIFORM przy dopasowanych parametrach:15/15 konfiguracji po CMA, każda4/4 seedy na niezależnych danych. |
| [E036](experiments/2026-11-07_E036_audyt_kosztu/WYNIKI.md) i [E037](experiments/2026-11-08_E037_kontrola_mac/WYNIKI.md) | Poprawa ma koszt; nie wykazaliśmy ogólnej przewagi przy dopasowanym MAC. U' zmienia również liczbę parametrów, więc nie rozdziela przyczynowo wpływu compute i pojemności. |
| [E038](experiments/2026-11-08_E038_operator_alokacja/WYNIKI.md) | TILT pomaga także pruningowi; brak ogólnego potwierdzenia mechanizmu specyficznego dla foldingu. |
| [E039](experiments/2026-11-08_E039_przestawienia_tilt/WYNIKI.md) | Uporządkowany profil wygrywa ze średnią4 ustalonych przestawień w9/9 konfiguracji, każda4/4; kolejność pozostaje sprzężona z MAC. |

**Robocza teza:** badamy nierównomierną alokację globalnego budżetu
parametrów w foldingu bez fine-tuningu, z rekalibracją BN. Pokazujemy
skuteczność prostej reguły TILT, jej korzyść również dla pruningu oraz
koszt i granice zastosowania.

Nowość oceniamy jako potencjalny, przyrostowy wkład empiryczno-metodologiczny.
Samo zastosowanie różnych ratio i preferowanie wcześniejszych warstw nie
wystarcza do deklaracji pierwszeństwa. Przegląd musi uwzględnić IFM,
Wang i in. oraz *Cut Less, Fold More*. Wskazanie future work uzasadnia
pytanie badawcze, ale samo nie dowodzi nowości naszego rozwiązania.

## Najbliższe zadania — na istniejących wynikach

1. **Spisać pytanie, wkład i metodę:** definicja grup kanałów, wzór TILT,
   dopasowanie budżetu, zaokrąglenia, pseudokod, pochodzenie wyboru+0.30
   i procedura BN. Rozróżnić redukcję kanałów, parametrów i MAC.
2. **Złożyć główne tabele i rysunki:** sparowane seedy, średnie/SD,
   bezwzględna trafność top1 i strata względem modelu wyjściowego. Ocenę
   niezależną E035 odróżnić od audytu kosztu E036 i kontroli dev E037–E039.
3. **Powiązać każde twierdzenie z dowodem i ograniczeniem:** obok korzyści
   jakości pokazać MAC, przepustowość, pamięć, wynik E038 i słabsze przypadki.
   Cztery seedy i15 konfiguracji nie stanowią60 niezależnych powtórzeń.
4. **Domknąć najbliższe prace pokrewne i ocenę nowości:** porównać dokładny
   problem, budżet, operator i wymagania dotyczące danych; deklarację
   „pierwsza metoda” rozważać dopiero po takim przeglądzie.
5. **Napisać dyskusję i wnioski**, a potem ocenić szkic pod konkretne
   miejsce publikacji. Wtedy wskazać jeden rzeczywisty brak, jeśli pozostanie.

Rezultat tego etapu: kompletny szkic z tabelą „twierdzenie → dowód →
ograniczenie”, odtwarzalnymi rysunkami i jasno określonym zakresem nowości.

## Co oznaczałaby praca nad większym nowum?

**To osobny kierunek badawczy, nie dalsza siatka współczynników TILT.**
Sensownym kandydatem byłaby nowa reguła alokacji uwzględniająca jednocześnie
limit parametrów i MAC. To hipoteza do sprawdzenia, nie wynik ani
zaplanowany eksperyment. Nie wznawiamy automatycznie search z E024–E028.

Taki kierunek wymaga osobnej decyzji użytkowniczki, nowej hipotezy,
zamrożonego protokołu i kryteriów sukcesu przed oceną, porównania z
UNIFORM/TILT/MAG2 przy jawnych ograniczeniach oraz zaplanowanej niezależnej
oceny. Nowe obliczenia prowadzimy na4GPU; nowy trening wymaga wyraźnej zgody.
Testy użyte w E035 nie są niewidzianą walidacją dla przyszłego strojenia.

Szkic już istnieje; zakres kolejnego testu po E042 jest opisany na początku
pliku.
