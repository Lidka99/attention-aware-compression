# Ocena wkładu i gotowości ilustracji — 2026-11-11

## Uzupełnienie — 2026-11-21: zapas pod pułapami U i nowe figury

[E042](experiments/2026-11-21_E042_zapas_pod_uniform/WYNIKI.md) wykazał
na dev, że w badanej puli istnieje profil poprawiający top‑1 przy liczbie
parametrów i MAC nie większych niż U/b50: C100 **+1,175 pp (3/4)** i C10
**+1,475 pp (4/4)**. Na C100 tylko 5/24 profili miało dodatnią średnią.
Wynik uzupełnia wcześniejszą lukę o kontrolę **pod niższym MAC UNIFORM**,
nie o skuteczność selektora: L06 wybrano na tym samym dev.

Powstały odtwarzalne figury [szerokości i jakość względem modelu wyjściowego
(E035)](experiments/2026-11-07_E035_ocena_niezalezna/figures/publikacja/README.md),
[schemat procedury](experiments/2026-11-07_E035_ocena_niezalezna/figures/schemat/README.md),
[jakość–P/MAC, wspólny dev](experiments/2026-11-08_E037_kontrola_mac/figures/publikacja/README.md)
i [pełny rozkład zapasu E042](experiments/2026-11-21_E042_zapas_pod_uniform/wykresy/20261121T212257Z_207ae03f/README.md).
Wąskie studium jest mocniejsze empirycznie, nadal bez tezy o nowej metodzie.
Inwentarz braków poniżej opisuje wcześniejszy stan, przed tym uzupełnieniem.

## Uzupełnienie — 2026-11-19: E041 po dopasowaniu P i MAC

[E041](experiments/2026-11-15_E041_kontrola_p_mac/WYNIKI.md) domknął jedną
brakującą kontrolę. TILT−średnia4kontroli po CMA: C10+0,11250pp(2/4),
C100+2,10375pp(4/4), przy różnicach P/MAC≤0,1%. Łączne kryterium
niespełnione; kontrola104 naC10 wygrywa z TILT o0,3850pp na4/4seedach.
Nie ma podstaw do równoważności naC10 ani ogólnej dominacji TILT.

Wkład pozostaje empiryczny: wpływ profilu przy budżecie P oraz jego
zależność od kosztu, operatora i zbioru. E041 wzmacnia to rozpoznanie na
C100, ale nie ustanawia ogólnej metody. Tiny nie był objęty E041.
Wszystkie wyniki tej kontroli to dev, nie nowa niezależna walidacja.

## Uzupełnienie — 2026-11-15: precedensy samego profilu

**Liniowy kształt alokacji TILT nie jest nową ogólną ideą.** ATP podaje
`s_i = S − β(L−1)/2 + β(i−1)` (§3.3, równanie10, s.8,
[2502.14770v1](https://www.alphaxiv.org/abs/2502.14770v1)). Przed clippingiem
jest to ta sama rodzina co TILT przy `g=i−1`, `G=L`, `b=S` oraz
`α=β(G−1)/2`. ATP obejmuje także eksperymenty vision/structured pruning
(dod.F.4–F.5, s.24–25). Nasz rzeczywisty budżet P sprzężonych szerokości
różni się od średniej sparsity warstw w ATP; nie ustanawia to samoistnie
nowej zasady alokacji.

Wcześniej Su i in. badali *Linear Decay*, keep-ratio proporcjonalne do
`L−l+1`, z normalizacją do budżetu (NeurIPS2020, dod.C.1–C.2, tabela5,
s.14, [2009.11094v2](https://www.alphaxiv.org/abs/2009.11094v2)). Dotyczyło
to głównie pruningu nieustrukturyzowanego z treningiem podsieci.

Nazwa i implementacja TILT są lokalne. Nie ustalono pierwszeństwa
zastosowania tej rodziny do foldingu. Rekomendowany zakres pozostaje
empiryczny: adaptacja do grup foldingu oraz charakterystyka jakości,
kosztu i operatora. Rozpoczęto [szkic](SZKIC_ARTYKULU.md), który jawnie
uwzględnia oba precedensy. Poniższa ocena wyników zachowuje ważność.

## Werdykt

**Mamy rozpoznawalny, ale przyrostowy wkład empiryczny.** Wyniki uzasadniają
przygotowanie i krytyczną ocenę szkicu wąskiego studium alokacji w foldingu.
Nie są potwierdzeniem dostatecznej nowości do mocnej pełnej publikacji ani
podstawą przedstawienia TILT jako istotnie nowej metody kompresji.
Wcześniejsze „materiał wystarcza do artykułu” oznacza gotowość do złożenia
spójnego szkicu, nie gwarancję publikowalności.

Rzetelność, własne checkpointy, cztery seedy i audyty zwiększają wiarygodność.
Nie zastępują oryginalności. Liczba przeprowadzonych eksperymentów również
nie jest samodzielnym wkładem.

## Co wynika z najbliższej literatury

Ponownie sprawdzono PDF pod kątem faktycznie wykonanych porównań.

| Praca i miejsce | Co jest już znane | Nasza węższa różnica |
|---|---|---|
| [ATP, 2502.14770v1](https://www.alphaxiv.org/abs/2502.14770v1), równanie10 s.8, dod.F.4–F.5 s.24–25 | Liniowo rosnąca sparsity, ta sama nieobcięta rodzina profili co TILT; także vision i structured pruning. | Sprzężone grupy foldingu, clipping i rzeczywisty globalny budżet parametrów; nie nowy liniowy kształt. |
| [Random Tickets / Linear Decay, 2009.11094v2](https://www.alphaxiv.org/abs/2009.11094v2), dod.C.1–C.2 s.14 | Malejące liniowo keep-ratio, normalizacja do budżetu i porównania z uniform/odwróceniem profilu. | Folding gęstych szerokości bez fine-tuningu z BN, zamiast głównie nieustrukturyzowanego pruningu z treningiem. |
| [Cut Less, Fold More, 2602.18116v1](https://www.alphaxiv.org/abs/2602.18116v1), §3 s.4–5, dod.F.2 s.23 i tab.9 s.31 | Duże badanie foldingu/pruningu, >1000checkpointów, różne przepisy treningu; uniform per-layer ratio. FOLD i MAG mają dla danego profilu te same kształty, parametry i FLOPs. | Badamy zmianę **profilu alokacji** przy globalnym budżecie P oraz jej koszt i zależność od operatora. **E036 nie podważa uczciwości ich porównania FOLD/MAG przy wspólnych szerokościach.** |
| [BALF, 2509.25136v3](https://www.alphaxiv.org/abs/2509.25136v3), §5 s.9–11 | Alokacja bez fine-tuningu, porównania z uniform i zależność jakości od tego, czy celem jest P czy FLOP; konflikt tych miar pokazany wprost. | Nasze pomiary dotyczą konkretnego operatora foldingu i porównania z MAG2. Hasła „alokacja ma znaczenie” i „P≠compute” nie będą naszą nowością. |
| [Relative Repairability, 2605.25508v1](https://www.alphaxiv.org/abs/2605.25508v1), §3 s.3–4, §5 s.7–8, dod.J s.23–24 | Alokacja uwzględniająca naprawę, wyniki zależne od reżimu oraz niedoskonały związek lokalnych diagnostyk z całą siecią. | Odróżnia nas folding gęstych szerokości i kontrolowany zestaw U/TILT/MAG2; ogólna teza o alokacji/naprawie oraz negatywny proxy nie są nowe. |

Szersze porównanie IFM, NetAdapt, Wang i in., GRAIL i prac interakcyjnych:
[notatka projektowa](NOTATKA_PROJEKTOWA_2026-11-09.md). Nie znaleziono identycznego
zestawu kontroli, ale nie jest to dowód pierwszeństwa.

## Czy zaadresowaliśmy problem pozostawiony przez autorów foldingu?

**Częściowo: tak jako badanie alokacji i zweryfikowana heurystyka przy P;
nie jako ogólny, skuteczny optymalizator jakości pod P/MAC.**

Wang i in., [2502.10216v2](https://www.alphaxiv.org/abs/2502.10216v2), s.13:
„it does not optimize sparsity levels per layer, leaving this for future work”.
To rzeczywiste wskazanie ograniczenia ich metody. Nie wynika z niego, że
alokacja per warstwa nie ma precedensów w literaturze lub że każdy nierówny
profil automatycznie rozwiązuje wskazany problem.

- TILT ustala kształt profilu; dopasowanie jego przesunięcia do globalnego P
  zapewnia wykonalność. Nie jest doborem wektora szerokości przez skutecznie
  zweryfikowany model przewidywanej jakości. E035 potwierdza użyteczność
  konkretnej reguły, nie rozwiązanie całego problemu alokacji.
- Brak dowodu globalnego optimum sam w sobie nie dyskwalifikuje heurystyki
  optymalizacyjnej. Ważniejsze jest tutaj ograniczenie do ustalonego kształtu
  i brak potwierdzonej przewagi skuteczniejszego ogólnego alokatora.
- Search E024–E028 nie dał stabilnego zysku nad TILT na obu zbiorach; E040
  zatrzymano na predykcji. E036/E037 ograniczają interpretację zysku TILT,
  a E042 potwierdza poprawę profilu wybranego na dev pod limitami P i MAC,
  bez potwierdzenia skuteczności alokatora.
- Ważne doprecyzowanie literatury: Wang i in. już pokazują szczególne
  niejednolite wzorce — kompresję ostatnich dwóch bloków dla porównania z IFM
  (s.11–12, fig.7) i różne poziomy dla grup bloków LLaMA (s.12).
  Nie możemy więc twierdzić, że pierwsi zastosowaliśmy nierówny folding.
  CLFM bada głównie wspólne uniform ratio, bez naszego badania doboru alokacji.

Bezpieczne sformułowanie wkładu:
„Podejmujemy kwestię alokacji pozostawioną w pierwotnej pracy o foldingu.
Badamy prostą, nierównomierną regułę pod globalnym budżetem parametrów,
potwierdzamy jej korzyść względem uniform w badanych ustawieniach oraz
charakteryzujemy koszt MAC i brak ogólnej specyficzności dla foldingu”.
Nie zastępować tego stwierdzeniem „rozwiązujemy otwarty problem optymalizacji”.

## Wkład, którego możemy bronić

Robocze pytanie: **jak nierówna alokacja zmienia jakość foldingu bez fine-tuningu
i co pozostaje z tej korzyści po uwzględnieniu kosztu oraz zmiany operatora?**

1. **E035:** zamrożony prosty profil poprawia top1 względem U przy dopasowanych
   parametrach w15/15konfiguracji CMA, każda4/4seedy na niezależnych wówczas
   danych. To replikowany wynik o określonym profilu, nie nowa ogólna zasada.
2. **E036/E037:** ilościowo określono koszt1,105–1,823×MAC; po poszerzeniu U
   do zbliżonego MAC brak ogólnej przewagi TILT. U' ma więcej P, więc nie
   rozdzielono przyczynowo compute od pojemności. Wartością jest charakterystyka
   tego kompromisu w badanym ustawieniu, nie odkrycie samej różnicy P/MAC.
3. **E038:** kontrola operator×alokacja pokazuje, że korzyść nie jest ogólnie
   specyficzna dla foldingu. Przy50% dodatni kontrast przyrostów tylko1/5par.
   To użyteczna informacja ograniczająca interpretację TILT.

E039 wspiera znaczenie badanego uporządkowanego profilu względem ustalonej
puli przestawień, ale MAC jest sprzężony z kolejnością. Nie daje mechanizmu
przyczynowego. E040 to negatywna kontrola predykcji, bez oceny top1; sensowna
do dodatku, nie zastępnik brakującego mocnego wkładu metodologicznego.

**Ryzyko recenzenckie:** znana idea alokacji, prosta heurystyka, dwie pokrewne
architektury, trzy małe zbiory, jeden przepis treningu i brak porównania z
pełnymi współczesnymi pipeline'ami. Nie można konkurować ze skalą CLFM samą
liczbą naszych seedów ani deklarować SOTA. Duży zysk względem U przy70% nie
usuwa niskiej jakości bezwzględnej na C100/Tiny.

## Inwentarz ilustracji — co już istnieje

Sprawdzono pliki, README i reprezentatywne PNG, nie tylko zapowiedzi w planie.
Poniższe lokalne zestawy mają PNG+PDF oraz kod i metadane źródeł.

| Materiał | Plik / katalog | Proponowane użycie |
|---|---|---|
| Niezależny top1 U/T | [E035/figures/e035_niezalezny_top1.png](experiments/2026-11-07_E035_ocena_niezalezna/figures/e035_niezalezny_top1.png) | Główny wynik; dodać odniesienie do teacher CMA. |
| Sparowane T−U | [E035/figures/e035_sparowane_delta.png](experiments/2026-11-07_E035_ocena_niezalezna/figures/e035_sparowane_delta.png) | Główny wynik; dobrze pokazuje4seedy i słaby R/C100. |
| MAC i throughput | [E036/figures/e036_mac_i_przepustowosc.png](experiments/2026-11-07_E036_audyt_kosztu/figures/e036_mac_i_przepustowosc.png) | Główny tekst, nie ukrywać kosztu w dodatku. |
| U/T/U' i kontrast przy zbliżonym MAC | [E037/figures/README.md](experiments/2026-11-08_E037_kontrola_mac/figures/README.md) | Główny tekst; zaznaczyć dev i większe P(U'). |
| Operator×alokacja | [E038/figures/e038_interakcja.png](experiments/2026-11-08_E038_operator_alokacja/figures/e038_interakcja.png) | Główny tekst; tabela absolute FOLD/MAG2 obok. |
| Cztery przestawienia | [E039/figures/e039_kolejnosc.png](experiments/2026-11-08_E039_przestawienia_tilt/figures/e039_kolejnosc.png) | Raczej dodatek; obowiązkowo ograniczenie MAC w podpisie/panelu kosztu. |
| Druga architektura / dev vs test | E034/figures oraz E035/figures/e035_dev_vs_niezalezny | Dodatek; nie dublować głównej figury E035. |
| Latencja batch1 | E036/figures/e036_batch1_ratio | Dodatek; SD nie dowodzi równoważności. |
| Nieudany model par | [E040/wykresy/e040_mae.png](experiments/2026-11-09_E040_interakcje_po_bn/wykresy/e040_mae.png) | Dodatek; to MAE predykcji, nie top1. |

## Braki w pakiecie do artykułu

**Nie mamy jeszcze kompletnego pakietu ilustracyjnego.** Poniższe braki
można uzupełnić z istniejących zapisów i rysunków koncepcyjnych, bez nowych
ocen modeli, GPU, treningu lub otwierania danych.

- [ ] **Autorski schemat procedury:** znany folding vs pruning, producent
  i konsument kanałów, grupy związane residualami; checkpoint→profil→operator
  →BN na train→ocena. Wyraźnie oddzielić znany operator od naszej alokacji.
- [ ] **Faktyczna alokacja U/T:** szerokości lub k/n per legalna grupa dla
  30/50/70% budżetu P, z podanymi rzeczywistymi P/MAC. Dane są w E035/frozen.json.
  Używać nazw grup i poprawnej kolejności upstream; nie utożsamiać50%P z50%kanałów.
- [ ] **Jedna figura jakość–zasoby:** top1 vs rzeczywiste P i vs MAC dla U/T/U'
  z E035-dev/E037. Wszystkie porównywane punkty muszą pochodzić z **dev**;
  nie wolno dołożyć U' z dev do krzywych testowych U/T. To kilka obserwowanych
  punktów, nie udowodniona granica Pareto ani optimum.
- [ ] **Odniesienie do modelu oryginalnego:** teacher CMA i/lub strata względem
  niego w głównym wykresie jakości. Dane są w controls E035 i WYNIKI.md;
  obecna figura absolute U/T nie ma linii teacher.
- [ ] **Redakcja publikacyjna:** spójne nazwy/kolory, podpisy z P/MAC i fazą
  danych, czytelność w docelowej kolumnie, usunięcie wewnętrznych E-ID z tytułów.
  Dla tekstu anglojęzycznego także tłumaczenie etykiet. PDF wystarcza jako
  format wektorowy; brak SVG nie jest sam w sobie problemem.

Jeśli rola BN ma być osobną tezą, potrzebna jest także zwarta tabela/figura
PRE/CMA/repair z istniejących wyników. Samo użycie BN opisujemy jako znaną
procedurę kalibracyjną, bez deklaracji nowości i bez „calibration-free”.

## Decyzja redakcyjna

Zacząć od krótkiego abstraktu, trzech precyzyjnych twierdzeń i porównania
z CLFM/BALF/RR, następnie ułożyć około5–6figur głównych. Dodatkowe ilustracje
nie wytworzą nowości. Jeśli po tym zestawieniu wkład sprowadza się do
„znana heurystyka pomaga na kolejnych benchmarkach”, adekwatniejsza będzie
krótsza praca empiryczna/replikacyjna niż rozbudowany artykuł metodologiczny.
Dobór miejsca publikacji wymaga gotowego szkicu; nie zakładamy z góry akceptacji.
E040 pozostaje zamknięty; ten audyt nie uruchamia nowej linii eksperymentów.
