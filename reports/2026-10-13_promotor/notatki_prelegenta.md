# Notatki prelegenta — prezentacja 2026-10-13

Pomoc do prezentacji: co powiedzieć na każdym slajdzie i co odpowiedzieć
na najczęstsze pytania. Liczby zgodne z `prezentacja.md` (stan 13.10.2026, E009 w toku).

## Jak wyjaśnić wszystko od zera (bez żargonu) — gotowe zdania

Ta sekcja jest po to, żebyś umiała opowiedzieć całość osobie, która **nic nie wie o temacie**.
Zdania w cudzysłowach możesz mówić prawie dosłownie.

### 1. Najpierw trzy pojęcia, na których stoi cała praca

- **Sieć neuronowa (CNN):** „to maszyna, która ogląda obraz i mówi, co widzi. Obraz przechodzi
  przez nią warstwa po warstwie — najpierw wyłapywane są krawędzie, potem kształty, potem całe obiekty."
- **Kanał:** „wyobraź sobie, że w każdej warstwie pracuje kilkaset osób i każda patrzy na obraz
  pod innym kątem: jedna widzi krawędzie, druga kolory, trzecia fakturę. Każda taka osoba
  to jeden **kanał**."
- **Kompresja:** „chcemy, żeby sieć była mniejsza i szybsza, więc zwalniamy część tych osób.
  Żeby nie stracić informacji, **łączymy ich obowiązki z innymi pracownikami** — to jest
  scalanie kanałów. Całe pytanie brzmi: kogo z kim połączyć, żeby stracić jak najmniej?"

### 2. Co to jest SE — najprościej

- „W sieci jest mały moduł — **SE** — który działa jak **kierownik zmiany**. Patrzy na każde
  zdjęcie osobno i mówi, którzy pracownicy (kanały) są teraz ważni, a którzy mogą mówić ciszej.
  Każdy dostaje **głośność od 0 do 1**."
- „To się sprawdza w praktyce: sieć z SE jest dokładniejsza za bardzo mały koszt —
  dlatego jest standardem w nowoczesnych sieciach."

### 3. Czego chciałam (jednym oddechem)

„Skoro ten kierownik (SE) wie, kto jest ważny dla każdego obrazu, to **może też wiedzieć,
kogo można zwolnić albo z kim połączyć bez szkody**. Chciałam sprawdzić, czy używanie SE
do wyboru kanałów do scalenia daje lepszy wynik niż wybieranie ich bez SE albo na chybił trafił."

### 4. Jak to sprawdziłam (najprostsze porównanie świata)

„Robię **dokładnie to samo cięcie sieci na trzy sposoby**:
- **po SE** — patrzę na wskazówki kierownika,
- **przed SE** — patrzę tylko na podobieństwo pracowników (bez kierownika),
- **losowo** — zwalniam/łączę na chybił trafił, jako punkt odniesienia.
Potem każdą wersję **tak samo naprawiam** (doszkalam) i sprawdzam, która jest najlepsza.
Jeśli SE daje przewagę — to zobaczę. Jeśli nie daje — też zobaczę."

### 5. Co wyszło (cała prawda w trzech zdaniach)

1. „**Mądry wybór kanałów działa** — jest wyraźnie lepszy niż łączenie na chybił trafił."
2. „**Ale wskazówki kierownika (SE) nic nie dodają** — wybór bez SE jest równie dobry.
   Sprawdziłam to przy dwóch siłach kompresji (10% i 30%) i po różnej długości naprawy (60 i 120 epok)."
3. „Trzecia rzecz: **bramki SE nie przewidują szkody** — nie mówią, które konkretne połączenie
   zaszkodzi bardziej niż inne. Ten wątek zamknęłam osobno."

### 6. Jak opowiedzieć E008/E009 bez liczb

„Biorę dobry model odniesienia i **usuwam 30% kanałów**, potem **doszkalam każdą wersję po równo**.
Wynik: model bez cięcia 59,45%; po SE 58,32%; przed SE 57,81%; losowo 57,55%.
Skoro po SE i przed SE dzieli praktycznie błąd pomiaru — **SE nie pomaga**.
Ostatni eksperyment (E009) to ta sama próba z dłuższym doszkalaniem, żeby mieć pewność."

### 7. Co to znaczy (dlaczego to nie jest porażka)

- „Hipoteza »SE pomoże w kompresji« — **niepotwierdzona** w moich warunkach. To **wynik
  negatywny, ale pełnoprawny**: w nauce też jest wiedzą, bo zamyka wątek uczciwie
  i mówi innym, żeby nie tracili na to czasu."
- „Przy okazji mam **solidne fundamenty**: dobrze wytrenowane modele odniesienia i całą procedurę
  kompresji z kontrolami i przedziałami niepewności."
- „To już **trzeci etap doktoratu** z podobnym wzorcem (v2: bramkowanie, v3: routing — też
  negatywy, z jednym obiecującym wyjątkiem). Wniosek: proste, mocne mechanizmy wygrywają
  z wymyślnymi; następne kroki wybieram na podstawie danych."

### 8. Trzy zdania na start rozmowy (do zapamiętania)

- „Zajmuję się tym, czy da się zmniejszać sieci neuronowe tak, żeby nie traciły jakości."
- „Sprawdziłam konkretną hipotezę: że mechanizm uwagi w sieci (SE) pomoże wybrać, które kanały połączyć."
- „Wynik jest negatywny, ale rzetelny: samo wybieranie działa, SE nic nie dodaje —
  i wiem, co sprawdzić dalej."

### 9. Ściągawka: każde słowo ze slajdów powiedz po ludzku

| Skrót na slajdach | Jak to powiedzieć |
|---|---|
| POST | wybór kanałów **z użyciem** SE („pytamy kierownika") |
| PRE | wybór **bez** SE („same notatki pracowników") |
| RANDOM | wybór **na chybił trafił** (punkt odniesienia) |
| CI [a; b] | **przedział niepewności**; jeśli nie obejmuje zera, różnica jest realna |
| pp | **punkty procentowe** (różnica w dokładności, np. 58% vs 57%) |
| epoka | jeden pełny przebieg nauki przez wszystkie obrazy |
| dev | zbiór, na którym sprawdzamy jakość („sprawdzian", nie egzamin) |
| baseline | **model odniesienia**, z którym porównujemy wszystkie warianty |
| kompresja 30% | usuwamy 30% „pracowników" (kanałów) |
| scalanie / folding | łączenie obowiązków pracowników, zamiast ich zwalniania |

### 10. Trzy rzeczy, o których nie zapomnieć

- **Zawsze najpierw odpowiedz na „po co"**, a dopiero potem pokazuj liczby.
- **Mów „uwaga" albo „bramki", nie „attention"** — i od razu dodaj „to ten kierownik z slajdu".
- **Nie przepraszaj za wynik negatywny** — powiedz, co dzięki niemu wiadomo i co dalej.

## Jak opowiedzieć całość w trzech zdaniach

1. „Badałam, czy **wewnętrzna uwaga sieci** — bramki SE, które mówią, które kanały cech są ważne
   dla danego obrazu — pomaga w **kompresji**: tak scalać kanały, żeby sieć straciła jak najmniej jakości."
2. „Zrobiłam **9 kontrolowanych eksperymentów**; w każdym to samo porównanie: wybór kanałów
   **po SE** vs **przed SE** vs **losowo**, przy równym budżecie naprawy."
3. „Wynik: **sam wybór kanałów działa** (lepiej niż losowanie), ale **SE nic ponad to nie dodaje**.
   Dziś domyka to ostatni eksperyment (120 epok naprawy). To rzetelny wynik negatywny —
   wiemy, gdzie szukać dalej, zamiast zgadywać."

## SE w 60 sekund (gdy ktoś zapyta „co to jest SE?")

- „SE to **mały moduł doklejony do warstwy splotowej**. Dla każdego obrazu osobno sprawdza,
  które kanały (mapy cech) są teraz ważne, i przemnaża je przez wagi 0–1.
  Analogia: **automatyczny korektor dźwięku** — dla każdego utworu ustawia głośność pasm."
- Kroki: **squeeze** (średnia po mapie → jedna liczba na kanał) → **excitation**
  (dwie warstwy FC + sigmoid → wagi 0–1) → **scale** (mnożenie kanału przez wagę).
- „Stosuje się, bo za **kilka procent parametrów** daje sieci uwagę zależną od treści —
  standard m.in. w EfficientNetach."
- „Moja hipoteza: skoro SE wie, co jest ważne dla tego obrazu, to może też wiedzieć,
  **których kanałów nie scalać** przy kompresji." (Na slajdzie jest schemat do pokazania.)

## Slajd po slajdzie

1. **Tytuł.** Jedno zdanie ramy: „Sprawdzamy, czy atencja kanałowa pomaga wybierać kanały
   do scalenia; dziś rozstrzyga to ostatni eksperyment E009." Wspomnieć: od zera, bez
   publicznych wag; 4×RTX 2080 Ti.
2. **W skrócie.** Przejść 4 punkty: chciałam / jak sprawdziłam / co wyszło / co z tego mam.
   **To najważniejszy slajd — reszta to szczegóły.** Jeśli brakuje czasu, pokazać tylko ten
   i slajdy 18–20.
3. **Cel i pytania badawcze.** Podkreślić kontrolę: ten sam operator scalenia i równy budżet
   naprawy. Zaznaczyć **kandydata na nowość (RQ1 ★)** — kontrolowany test *istniejących*,
   nietrenowanych pod kompresję bramek SE; uczciwie dodać, że audyt macierzy nowości jest w toku
   (to kandydat, nie potwierdzona luka). Wynik negatywny = pełnoprawny wynik.
4. **Czym jest SE.** Pokazać schemat, użyć analogii korektora; nie wchodzić w matematykę.
   Zamknąć zdaniem o hipotezie pracy.
5. **Literatura (1).** SE i ECA; autorzy SE sami wskazują użytek wag ważności w kompresji — stąd hipoteza.
6. **Literatura (2).** Neuron Merging (co scalać ma znaczenie), folding (Cut Less/Fold More),
   GRAIL (naprawa po fakcie), ADSC (uczy scorer atencją — my pytamy o *istniejące* bramki);
   audyt nowości nieukończony.
7. **Hipoteza i kryteria.** Zdefiniować POST/PRE/RANDOM; test główny = POST−PRE po równym
   dostrojeniu, sukces = dolna granica 95% CI > 0. RANDOM to kontrola konieczna.
8. **Metodologia.** Protokoły zamrożone przed biegiem; wynik = ostatnia epoka; bootstrap 10 000
   po klasach. Ograniczenie: **jeden seed** — do rozstrzygnięcia przed publikacją.
9. **Plan badań.** Szybko przejść tabelę; zatrzymać się na kolumnie „po co".
10. **E001.** Przeuczenie (train ≈99,9% vs dev 66,84%) — dlatego Tiny ImageNet.
11. **E002.** Podobne pary zostają podobne po SE; wniosek: mierzyć szkodę, nie zgadywać z wag bramek.
12. **E003.** −10,02% parametrów; POST−PRE +0,30 pp, CI obejmuje 0; obie selekcje > losowa surowo.
13. **E004.** Po równym dostrojeniu różnice znikają (CI [−0,30; +1,02]); RANDOM dorównał —
    stąd program E008/E009.
14. **E006.** Bramki nie przewidują szkody (ρ(g_pert) = +0,044; Δρ CI zawiera 0; FWER p = 0,081);
    wzór uczciwego zamykania hipotez.
15. **E005/E007.** 51,53% → **58,49%** vs literatura 60,54% (ich val, inny batch/warmup) — ten sam rejon.
16. **E008 etap 1.** 30% bez naprawy: rozpad (1–10%); POST < PRE surowo.
17. **E008 etap 2 (60 ep).** POST−PRE +0,51 pp, CI [−0,18; +1,19] — próg nieprzekroczony;
    POST−RANDOM +0,77 pp, CI [+0,03; +1,50] — graniczna przewaga; kompresja kosztuje 1,1–1,9 pp.
18. **E009 (120 ep).** Izoluje długość naprawy. Stan: BASE **60,04%**, PRE **59,18%** po pełnych
    120 epokach (+0,59 i +1,37 pp vs 60 epok); POST w toku, RANDOM w kolejce; werdykt po południu.
19. **Co wyszło / nie wyszło (po ludzku).** Jedno zdanie na wejściu (jak wyżej), potem bilanse.
20. **Wnioski.** Scenariusz A: zamknięcie wątku selekcji → decyzja o kształcie doktoratu.
    Scenariusz B: ścieżka publikacyjna. Pytania do omówienia — patrz slajd.
21. **attention_v2 (etap I).** Bramkowanie kanałów w ResNet-18: brak przewagi jakościowej
    i brak zysku czasu (91,58% vs 93,48% standard; latency bez poprawy); framework M1–M6
    i audyt wiarygodności danych zostają jako dorobek metodologiczny.
22. **attentionv3 (etap II).** Routing budżetu: seria negatywów (UCLA, value-of-compute,
    oracle cloning, semantic feedback); po pivocie early exit: pilot wieloseedowy −0,07 pp
    przy 85% kosztu, latency batch-1 −14,6% — obiecujący, wymaga domknięcia (static-depth
    seed 2026, latency end-to-end). Obecna linia E001–E009 jest **niezależna** od v2/v3.

## Możliwe pytania i odpowiedzi

- **„Co dokładnie chciała pani osiągnąć?"** → zdanie 1 z sekcji „Jak opowiedzieć całość";
  cel praktyczny: wynik publikowalny (też negatywny) i świadomy wybór dalszego kierunku.
- **„Czym jest SE?"** → sekcja „SE w 60 sekund" + slajd ze schematem.
- **„Po co było to wszystko, jeśli wynik jest negatywny?"** → seria kontrolowanych eksperymentów
  **wykluczyła konkretną hipotezę** (SE jako sygnał do scalania) i pokazała, co działa
  (podobieństwo kanałów), a co nie. To domyka wątek rzetelnie, a nie „nic nie wyszło".
- **Dlaczego ocena na dev, nie na teście?** Rezerwa i test są celowo nietknięte pod finalną
  publikację; dev wspólny dla wszystkich wariantów (porównania sparowane).
- **Jeden seed — czy wyniki są wiarygodne?** To pilotaż z kontrolą; CI opisuje zmienność obrazów,
  nie między treningami. Przed publikacją: dodatkowe seedy (pytanie 3 na slajdzie wniosków).
- **Dlaczego E009 kończy się tak późno?** Warianty sekwencyjnie na 4 GPU (DDP):
  4 × 120 epok × ~170 s ≈ 22 h; chroni to równy budżet i determinizm.
- **Co jeśli E009 pokaże POST > PRE istotnie?** Scenariusz B: rozbudowa kryterium, więcej seedów,
  pełna kompresja; literaturowo budować na foldingu + GRAIL.
- **Dlaczego nie kwantyzacja / większe kompresje od razu?** Możliwe kierunki (pytanie 2);
  wybór chcemy oprzeć na wyniku E009, nie na przeczuciu.
- **Czy POST−RANDOM na granicy to nie artefakt?** CI [+0,03; +1,50] to dolna granica „o włos"
  powyżej zera; opisujemy to jako słaby sygnał, a nie dowód.
- **Czy bramki SE na pewno były dobrze zmierzone?** E006: kontrole CPU, osobny pomiar szkody na
  rozłącznych parach, poprawka protokołu 1.0→1.1 w DECISIONS; nieudany bieg zachowany.
