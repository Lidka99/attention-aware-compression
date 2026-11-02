# E022 — mechanizm alokacji TILT

Status: **completed**, 2026-10-31. Mechanizm częściowo wyjaśniony;
**prosta teza o ochronie najbardziej wrażliwych grup post-BN niepotwierdzona**.
Odwrócony kierunek jest szkodliwy na wszystkich trzech zbiorach, również
na CIFAR-100. Nie wynika z tego optymalność TILT ani addytywność szkód.

## Wykonanie i źródła

- Zamrożony przed nowymi pomiarami [OPIS.md](OPIS.md).
- Run: `runs/E022/20261031T111804_ccc634a5`, **8/8 nowych ablacji**,
  GPU0–3 po dwa zadania, exit0 na każdym. 4×RTX2080Ti, Conda attention,
  Python3.11.15, PyTorch2.12.0+cu130, CUDA13.0. Batch128 na GPU/przebieg,
  bez gradientów/treningu wag. Czas od manifestu do ukończenia: **147,85 s**.
  Artefakty poniżej 1 MiB po analizie; limit0,5GiB spełniony.
- Ponownie wykorzystano **98/98** E016 S1, **49/49** E020 Tiny,
  **28** zakończonych ablacji E020 i UNIFORM/TILT z E014/E015/E019.
  Razem **147/147** pomiarów z kontrolami i **36/36** ablacji.
- Pominięty wcześniej w statusie E020 nie został nadpisany. Jego ablacja
  ma historyczny manifest `running`, 28 completed, 8 brakujących i puste
  logi. Przyczyna przerwania nieustalona. Brakujące zadania są wyłącznie
  w nowym runie E022. Kod/helpery/config E020 zgodne z zapisanym źródłem;
  wrapper E022 pomija tylko zapis logitów, zachowując ocenę i metryki.
- Analiza finalna:
  `runs/E022/20261031T111804_ccc634a5/analysis_20261031T114947Z/`:
  `analysis.json`, `groups.csv`, `profiles.csv`, `ablations.csv`,
  `alignment.csv`, `allocations.csv`, `TABLES.md`, `profile75.png`.
  Poprzednia analiza `analysis_20261031T112506Z` zachowana; finalna dodaje
  jawne werdykty kryteriów, SD różnic i pełne tabele alokacji pre/post BN.
- Sprawdzono hashe checkpointów i splitów z inwentarza, 28 ponownie
  użytych summary, protokołu i uruchomionego kodu E022; analiza zawiera
  hashe wszystkich odczytanych JSON oraz własnego kodu. Kontrole, zgodność
  baseline, kompletność grup i tolerancja budżetów przechodzą asercje.
- Train służył tylko korekcie BN, ocena na dev. Oficjalny test i rezerwa
  nietknięte. Brak nowych danych, checkpointów, treningu i 96×96.

## 1. Głębokość a szkoda przed i po BN

Pojedyncza grupa, **75% kompresji kanałów w tej grupie**, pozostałe grupy
nieskompresowane; seed42, 12 grup na zbiór. To inna wielkość niż globalne
30/50/70% redukcji parametrów w eksperymentach alokacji.

| Zbiór | ρ(głębokość, szkoda pre-BN) | ρ(głębokość, szkoda post-BN) |
|---|---:|---:|
| CIFAR-10 | −0,720 | −0,028 |
| CIFAR-100 | −0,077 | +0,643 |
| Tiny ImageNet | −0,252 | +0,091 |

**Wczesne grupy są wyraźniej wrażliwe przed BN na CIFAR-10, ale po BN
ten trend znika.** Na CIFAR-100 po BN szkoda rośnie z głębokością; na
Tiny nie ma silnego monotonicznego trendu. Wniosek jest podobny przy
25/50%: rho post CIFAR-100=0,581/0,650, CIFAR-10=−0,007/−0,133,
Tiny=0,238/0,203. Nie można opisywać wszystkich zbiorów jednym zdaniem
„im wcześniej, tym bardziej wrażliwie”.

## 2. Ile szkody odzyskuje BN i co pozostaje krytyczne?

Średnia po 12 pojedynczych grupach przy 75%. Odzysk procentowy to iloraz
sum odzysku i szkody pre, nie średnia niestabilnych ilorazów per grupa.

| Zbiór | Szkoda pre [pp] | Szkoda post [pp] | Odzysk BN [pp] | Odzysk szkody |
|---|---:|---:|---:|---:|
| CIFAR-10 | 13,57 | 2,82 | 10,76 | 79,3% |
| CIFAR-100 | 28,98 | 9,54 | 19,45 | 67,1% |
| Tiny ImageNet | 21,97 | 13,61 | 8,36 | 38,1% |

**BN nie odzyskuje większości szkody na Tiny.** Nie należy rozszerzać
wcześniejszego wniosku z dwóch CIFAR na trzeci zbiór.

Najbardziej krytyczne grupy post-BN, szkoda [pp]:

| Zbiór | 1. | 2. | 3. |
|---|---|---|---|
| CIFAR-10 | `layer3.0.relu2`: 11,68 | `layer2.0.relu2`: 9,38 | `relu_conv`: 4,56 |
| CIFAR-100 | `layer4.0.relu2`: 30,14 | `layer3.0.relu2`: 22,10 | `layer4.1.relu1`: 19,66 |
| Tiny | `layer3.0.relu2`: 32,09 | `relu_conv`: 27,06 | `layer2.0.relu2`: 25,31 |

Przykład: na CIFAR-100 `layer4.0.relu2` traci 36,16pp przed BN,
a BN odzyskuje tylko6,02pp. Na Tiny stem traci46,85pp, odzyskując19,79pp.

**Kontrola samej korekty BN:** przy ratio0 szkoda pre=0 i agreement=1
na wszystkich zbiorach. BN bez kompresji zmienia trafność o +0,20pp
(CIFAR-10), +0,06pp (CIFAR-100), −0,46pp (Tiny). Szkoda post przy75%
względem tej kontroli wynosi odpowiednio **3,02 / 9,60 / 13,15pp**.
Ranking grup i korelacje pozostają takie same (stałe przesunięcie).

## 3. Dlaczego CIFAR-100 ma słabszy zysk z TILT?

Wyniki wspierają **konflikt między ochroną wczesnych grup a kosztem
kompresji późnych**, nie wyjaśnienie samą liczbą klas lub baseline.

1. **Koszt późnych grup:** CIFAR-100 ma silny dodatni związek głębokości
   ze szkodą post-BN. TILT zwiększa kompresję właśnie późnych grup.
   Na CIFAR-10/Tiny szczyty są w stem/layer2/layer3, a nie głównie layer4.
2. **BN zmniejsza przewagę TILT nad UNIFORM:** przy globalnym50%
   różnica pre→post zmienia się następująco (średnie po4 seedach):

| Zbiór | TILT−UNIFORM przed BN [pp] | Po BN [pp] |
|---|---:|---:|
| CIFAR-10 | +35,69 | +3,77 |
| CIFAR-100 | +25,84 | +1,96 |
| Tiny | +13,40 | +10,32 |

   Na CIFAR-100 BN podnosi UNIFORM z18,55 do53,25%, a TILT z44,39
   do55,21%. Zatem duża przewaga przed BN prawie znika. Na Tiny odzysk
   nie wyrównuje wyników tak mocno. To rozkład zmierzonego efektu,
   nie izolowany dowód przyczyny różnic między datasetami.
3. **Nie wystarcza ogólna „trudność zbioru”:** względna średnia szkoda
   post przy75% to 2,95% baseline CIFAR-10, 12,72% CIFAR-100 i26,64%
   Tiny. Tiny jest bardziej uszkodzony bezwzględnie i względnie, a mimo
   to ma większy zysk z TILT. Również liczba klas200 nie tłumaczy prostym
   trendem słabości CIFAR-100 względem Tiny.
4. **Odwrócenie kierunku nie rozwiązuje konfliktu:** ANTI-TILT jest
   katastrofalnie gorszy także na CIFAR-100 (poniżej). Pojedyncze grupy
   nie opisują w pełni skutków kompresji całej sieci.

## 4. Czy TILT chroni grupy o wysokiej zmierzonej szkodzie?

Ochrona=`ratio_UNIFORM−ratio_TILT`; ratio jest **odsetkiem kompresji**
(operator: szerokość=`floor(n*(1−ratio))`). Dodatnia ochrona oznacza
więcej pozostawionych kanałów niż w UNIFORM. Szkodę porównujemy przy
jednakowym75% kompresji grupy, aby nie mieszać wrażliwości i ratio.

| Zbiór | ρ(ochrona, szkoda pre) przy globalnym50% | ρ(ochrona, szkoda post) |
|---|---:|---:|
| CIFAR-10 | +0,620 | −0,039 |
| CIFAR-100 | +0,046 | −0,553 |
| Tiny | +0,188 | −0,099 |

**Silna ochrona grup o dużej szkodzie post-BN (ρ≥0,5) nie zachodzi
na żadnym zbiorze.** Na CIFAR-100 związek jest wręcz przeciwny. Na
CIFAR-10 TILT lepiej pasuje do szkody przed BN. Profile szkody mają
tylko seed42; identyczne profile szerokości na kolejnych seedach nie
są niezależną replikacją tego związku.

Interpolowana suma zmiany szkód pojedynczych grup UNIFORM−TILT przy50%
wynosi **+1,67 / −0,53 / +12,61pp**. Na CIFAR-100 ma przeciwny znak
niż rzeczywisty zysk +1,96pp. To **opisowy wskaźnik, nie addytywna
predykcja wyniku całego modelu**.

### Ablacja kierunku, dopasowane parametry, po BN

| Zbiór, globalny50% | ANTI−UNIFORM [pp], średnia±SD | Ujemne seedy | ANTI−TILT [pp] |
|---|---:|---:|---:|
| CIFAR-10 | −32,52±3,18 | 4/4 | −36,29 |
| CIFAR-100 | −29,29±4,52 | 4/4 | −31,24 |
| Tiny | −15,72±1,72 | 4/4 | −26,03 |

Przy30/70% ANTI−UNIFORM również ujemne na wszystkich seedach: **36/36**
ujemnych par w całej ablacji. Wynik wspiera znaczenie kierunku alokacji,
ale **obala hipotezę, że ochrona późnych grup przez ANTI naprawi CIFAR-100**.
Odwrócenie zmienia cały profil równocześnie, nie identyfikuje pojedynczej
grupy odpowiedzialnej za przewagę.

## 5. Czy rekonstrukcja wag przewiduje szkodę przy stałej kompresji?

Spearman błąd rekonstrukcji–szkoda post-BN; po12 grup, seed42:

| Zbiór | 25% | 50% | 75% | 90% | Prógρ≥0,5 |
|---|---:|---:|---:|---:|---|
| CIFAR-10 | 0,734 | 0,762 | 0,706 | 0,483 | 3/4 |
| CIFAR-100 | 0,816 | 0,755 | 0,748 | 0,357 | 3/4 |
| Tiny | 0,706 | 0,566 | 0,510 | 0,252 | 3/4 |

**Tak, jako predyktor rang szkody pojedynczych grup przy25–75%; nie
jest to tylko artefakt mieszania poziomów.** Przy90% związek słabnie
na wszystkich zbiorach. Zamrożone kryterium3/4 poziomów na≥2/3 zbiorów
spełnione (3/3). Tiny przy75% leży blisko progu. Dla NLL rho przy75%
wynosi0,517/0,748/0,469 — siła zależy też od miary szkody.

Korelacje pooled0,800/0,702/0,739 nie powinny zastępować tej tabeli.
To diagnostyka, nie dowód, że predyktor zbuduje dobrą globalną alokację;
negatywny wynik AUTO50 pozostaje obowiązujący.

## Kryteria i ograniczenia

| Kryterium OPIS | Wynik |
|---|---|
| Kompletność, kontrola pre, 4GPU, exit0, budżety | TAK |
| Czas≤90min, nowe wyniki≤0,5GiB | TAK |
| Silna ochrona szkody post przy globalnym50% | NIE, 0/3 zbiorów |
| ANTI gorszy od UNIFORM na CIFAR-10/Tiny przy50%, ≥3/4 | TAK, oba4/4 |
| ANTI pomaga CIFAR-100 względem UNIFORM lub TILT przy50% | NIE |
| Rekonstrukcja:≥3/4 poziomów na≥2/3 zbiorów | TAK, 3/3 |

Reanaliza istniejących wyników + osiem uzupełnień, nie nowa niezależna
replikacja. Profile grupowe tylko seed42, ablacja seedy42–45. Dev,
jedna rodzina architektury; Tiny ma inny stem i rozdzielczość. Nie
izolujemy wpływu liczby klas, danych i przepisu treningu. Progi są
opisowe, brak podstaw do populacyjnych twierdzeń przyczynowych.

## Decyzja po mechanizmie

Zachować TILT jako empirycznie skuteczną alokację, **zrezygnować z mocnej
narracji „chroni najwrażliwsze grupy po BN”**. Wsparcie dotyczy kierunku
alokacji i różnic profilu szkód/odzysku BN, z wyraźnymi interakcjami
nieopisanymi przez pomiary pojedynczych grup. Następny proponowany krok
to zamrożony protokół łagodnych budżetów10/20% z planu; **nie uruchomiony**.
Nowy trening wymaga osobnej zgody użytkowniczki.
