# Czy warto szukać mocniejszej metody? — 2026-11-09

**Rozstrzygnięcie2026-11-11:** [E040 zakończony negatywnie](experiments/2026-11-09_E040_interakcje_po_bn/WYNIKI.md).
Bramka predykcji niespełniona na obu zbiorach,0/4poprawy na każdym;
próba zatrzymana przed dev. Zgodnie z poniższą regułą wracamy do artykułu.
Pozostała treść jest historycznym projektem sprzed pomiarów.

**Rekomendacja: jeden ograniczony pilot, z ostrożną oceną nowości.**
Istnieje konkretna różnica warta falsyfikacji: empiryczny model **nieaddytywnej
szkody całej procedury folding + BN**, profilowany w skompresowanym otoczeniu.
Szanse na mocniejszy wkład są umiarkowanie małe; sama kombinacja alokacji,
kosztu i naprawy nie wystarczy. Nie ma obecnie dowodu skuteczności ani pierwszeństwa.
Nawet dodatni pilot potwierdziłby użyteczność tej operacjonalizacji, nie sam
w sobie duże nowum względem znanego rozwinięcia drugiego rzędu.
Poniżej jedna propozycja; [szczegółowy protokół E040](experiments/2026-11-09_E040_interakcje_po_bn/OPIS.md).
**Etap wykonany: projekt i przegląd, bez GPU, treningu i odczytu obrazów/wag.**

## 1. Najbliższa literatura — co rzeczywiście jest już znane

Przeczytano wskazane wersje PDF, nie tylko abstrakty. Strony dotyczą PDF.

| Praca | Precedens i granica względem propozycji |
|---|---|
| [Wang i in., model folding, 2502.10216v2](https://www.alphaxiv.org/abs/2502.10216v2), §3, s.5–8; s.13,19 | Clustering i naprawa statystyk; grupowanie uwzględnia **zarówno producenta, jak i konsumenta** kanałów. Nie można przedstawiać tego jako foldingu ignorującego zależności warstw. Stosowane są jednolite ratio; lokalne sprzężenie wag w jednym foldingu nie jest modelem interakcji **wyborów szerokości różnych grup**. Wzmianka o przyszłej alokacji nie dowodzi naszego pierwszeństwa. |
| [Cut Less, Fold More, 2602.18116v1](https://www.alphaxiv.org/abs/2602.18116v1), §2–3, s.2–6; s.9 | Projekcje, błąd rekonstrukcji, porównania FOLD/MAG1/MAG2 przy jednolitych ratio. Gwarancja dopuszcza różnicę rzędu o jeden i dotyczy pojedynczej warstwy; nie daje optimum całej sieci po BN. CNN-y korzystają z danych do REPAIR mimo szerokiego określenia „calibration-free” w pracy. |
| [IFM, 2310.06756v2](https://www.alphaxiv.org/abs/2310.06756v2), §4, s.6–7; alg.1, s.16 | Iteracyjne łączenie cech według wag obu sąsiednich warstw; próg β może dawać nierówną kompresję bez fine-tuningu i danych. Ani samo merging bez treningu, ani niejednolite szerokości nie są nowością. Brak naszego pomiaru szkody par alokacji po BN. |
| [NetAdapt, 1804.03230v2](https://www.alphaxiv.org/abs/1804.03230v2), §3, s.4–8 | Jawne **wielorakie ograniczenia zasobów**, aktualizacja zależnych kanałów i ocena propozycji w aktualnym modelu. Zależności są więc częściowo uwzględniane przez pomiar. Algorytm wymaga krótkiego i długiego fine-tuningu; nie estymuje proponowanej tabeli interakcji po rekalibracji. MAC nie zastępuje jego pomiarów latencji. |
| [BALF, 2509.25136v3](https://www.alphaxiv.org/abs/2509.25136v3), §4.3–4.4, s.7–8; dod.C, s.21–23 | Kalibracyjna alokacja rang bez fine-tuningu na podstawie zniekształcenia aktywacji, knapsack i relaksacja Lagrange'a. Optymalizuje addytywny profil pod FLOP **lub** P; równoczesne ograniczenia wymieniono w ograniczeniach, s.12. Sam kosztowy alokator bez treningu jest już znany. Operator zachowuje wymiary między warstwami, inaczej niż folding. |
| [How Sparsity Allocation Shapes Label-Free Post-Pruning Recoverability, 2605.21972v1](https://www.alphaxiv.org/abs/2605.21972v1), §3–6, s.3–9 | ERK/LAMP przy tym samym globalnym sparsity i stałym ASR pokazują interakcję **alokacja × naprawa**. To nie nowy alokator par grup; chodzi o bardzo wysoki, nieustrukturyzowany pruning, nie o gęste szerokości/MAC. Ogólnej tezy „alokacja wpływa na naprawialność” nie możemy zawłaszczyć. |
| **[Relative Repairability, 2605.25508v1](https://www.alphaxiv.org/abs/2605.25508v1)**, §3, s.3–5; dod.C/J, s.18,23 | Jeszcze bliższy precedens: **alokacja według szkody pozostałej po określonej naprawie**. RR = residual/raw, diagnostyka jednej warstwy przy pozostałych gęstych, addytywny proxy i greedy. Autorzy jawnie wskazują nieuchwycone interakcje przy jednoczesnym pruningu. Różnicą może być ich bezpośredni pomiar, a nie samo „repair-aware”. |
| [Shapley Pruning, 2407.15875v1](https://www.alphaxiv.org/abs/2407.15875v1), §4, s.3–4; §5.3, s.7 | Ocena koalicji i szkody wspólnego usunięcia kanałów ma precedens. W tej pracy ranking dotyczy neuronów w warstwie i wyniki kompresji obejmują retraining. Różnice skończone i uwzględnianie synergii nie są naszym nowym algorytmem matematycznym. |
| [Task-Aware QUBO Allocation, 2609.22238v2](https://www.alphaxiv.org/abs/2609.22238v2), §3, s.2–4; §6, s.9 | Dyskretny kwadratowy surrogate alokacji z interakcjami i oceną przed/po recovery już istnieje. Interakcje jakości są tu dobranymi priorytetami/iloczynami szkód, nie kontrastami zmierzonymi po BN; soft BOP i LSQ+ różnią się od naszych twardych limitów i braku treningu. Samo użycie QUBO nie będzie wkładem. |
| [GRAIL, 2602.23795v2](https://www.alphaxiv.org/abs/2602.23795v2), §3–4, s.3–6; s.9 | Kalibracyjna rekonstrukcja producent–konsument naprawia również folding bez gradientów. To mocny precedens dla zachowania funkcji i kompensacji; nie alokator szerokości. Przewaga w pilocie przy stałym BN nie wystarczy do twierdzenia o najlepszym pełnym pipeline kompresji; takie twierdzenie wymaga porównania także z GRAIL/BALF. |

Przegląd sześciu prac wejściowych uzupełniono dwoma wyszukaniami prac o
interakcjach/alokacji/naprawie. **Brak znalezionego identycznego rozwiązania
nie jest dowodem braku precedensu.**

## 2. Jedna proponowana różnica

Roboczy opis, bez nowego akronimu: **dyskretny model drugiego rzędu szkody
foldingu po rekalibracji BN**. Nowości upatrujemy warunkowo w sposobie
estymacji i wykorzystania odpowiedzi konkretnej procedury, nie w solverze.

Niech F(k) będzie stałym, deterministycznym operatorem foldingu, R_A — CMA BN,
a f₀ — oryginalnym checkpointem. Zachowujemy cel z E028, aby zmiana celu nie
udawała nowej metody:

`D_post(k) = mean_{x∈B} 1[argmax R_A(F(k))(x) ≠ argmax f₀(x)]`.

Dla 12 legalnych grup wybieramy po dwie szerokości: H=⌊0,8n⌋ i L=⌊0,6n⌋.
Punkt odniesienia k⁰ jest dopuszczalną projekcją U50 na to menu, najbliższą
w znormalizowanej odległości szerokości, wybraną wyłącznie z geometrii.
z_g=1 oznacza zamianę k⁰_g na drugą szerokość. Otoczenie jest już skompresowane
pod oba limity. Z 79 ustalonych pomiarów **na operator/checkpoint** wyznaczamy:

`a_g = D(e_g) − D(0)`;
`I_gh = D(e_g+e_h) − D(e_g) − D(e_h) + D(0)`;
`Q_post(z) = D(0) + Σ a_g z_g + Σ_{g<h} I_gh z_g z_h`.

Wybór: minimum Q_post wśród profili spełniających oba rzeczywiste limity,
wyfiltrowanych z **wszystkich 4096** kombinacji. To standardowe rozwinięcie
funkcji zbioru do drugiego rzędu i mała enumeracja, nie nowy algorytm optymalizacji. Profil nie jest
aktualizowany po obejrzeniu wybranego modelu. Z analogicznej tabeli przed BN
otrzymujemy `I_BN = I_post − I_pre`; znak i znaczenie tego składnika są empiryczne.

- **Względem AUTO50:** zastępujemy znormalizowany ranking rekonstrukcji wag
  i stałą amplitudę 0,30 pomiarem funkcjonalnej szkody po BN oraz jawnych par
  zmian. AUTO50 już liczyło sprzężone P — nie przypisujemy mu braku tej kontroli.
- **Względem E024–E028:** zachowujemy dokładnie disagreement z E027/E028
  (E024/E026 używały KL) oraz rozdzielenie danych, ale zamiast
  kolejnych donor→receiver i wyboru najlepszej zmierzonej pełnej sieci powstaje
  zamrożony profil odpowiedzi. Wybrana kombinacja zwykle nie była mierzona;
  sprawdzamy jej przewidywalność na osobnych alokacjach. Nie powiększamy starej puli.
- **Hipoteza:** drugi rząd wystarcza w tym ograniczonym otoczeniu i przekłada
  się na top1; profil po BN jest użyteczniejszy niż przed BN. Interakcje wyższych
  rzędów, szum kalibracji i niezgodność disagreement z trafnością mogą to obalić.
- **Jeszcze nie do obrony:** pierwsza metoda interakcyjna/repair-aware,
  specyficzność dla foldingu, globalne optimum, calibration-free, przewaga
  latencji lub SOTA. E038 nadal ogranicza narrację o specyficzności.

## 3. Ograniczony pilot E040 — decyzje ustalone przed oceną

| Element | Ustalenie |
|---|---|
| Jedno pytanie | Czy empiryczne interakcje **po BN** poprawiają przewidywanie skutków całych alokacji oraz dają praktyczną, stabilną przewagę top1 pod oboma limitami, ponad addytywny profil po BN i profil par przed BN? |
| Zakres | Własne ResNet18 E011/E015, CIFAR-10 i CIFAR-100, seedy42–45, jeden budżet nominalny50% parametrów. Dwa zbiory kontrolują transfer; 50% jest kompromisem między sufitem przy łagodnej kompresji a głęboką degradacją przy70%. E028 ostrzega o małym marginesie C10, ale nie wyznacza granicy dla tego zadania P+MAC. 12 grup ogranicza koszt par do79 pomiarów na operator/checkpoint. |
| Dokładne limity | C10: **P≤5 584 580, MAC≤276 383 716**; C100: **P≤5 581 542, MAC≤275 335 732**, tj. rzeczywiste U/b50 z E035/E037. Tolerancja przekroczenia **0**; niedowykorzystanie jawnie raportowane. Wspólne limity nie oznaczają dokładnego dopasowania P i MAC. |
| Dane/BN | Istniejące rozłączne A4096/B2048/C2048 z train. A: crop/flip, reset+CMA, batch128; B: nieetykietowany dobór; C: audyt predykcji po zamrożeniu wyborów; dev5000 dopiero potem, jeden odczyt wyników. Dev i C nie są nowymi niewidzianymi danymi w historii projektu. |
| Mocne porównania | FOLD/UNIFORM, TILT z β=0:0,05:0,30 dostosowanym do obu limitów, projekcja TILT i U50 na tę samą przestrzeń; profile addytywny-post i par-pre. MAG2: te same referencje oraz **własne** profile addytywny-post i par-post, nie tylko przeniesione szerokości FOLD. |
| Bramka predykcji | Tylko FOLD: na16 z góry ustalonych dopuszczalnych alokacjach zmieniających≥3 grupy względem k⁰, przy≥90% wykorzystania obu limitów: MAE modelu par-post≤0,80×MAE addytywnego-post średnio po seedach, osobno na obu zbiorach; poprawa≥3/4 seedów na każdym. To ograniczona kontrola predykcji, nie walidacja każdego profilu ani predykcji MAG2. |
| Praktyczna poprawa | Na **każdym** zbiorze średni zysk≥1,00pp wobec każdego z zamrożonych punktów U/TILT/MAG2, kierunek≥3/4 i brak straty>0,50pp na pojedynczym seedzie. Ponadto vs addytywny-post≥0,50pp, vs par-pre≥0,25pp; oba kontrasty dodatnie≥3/4 na każdym zbiorze. |
| Agregacja | Sparowane różnice po czterech seedach **osobno dla każdego zbioru**; średnia, SD(ddof=1), wszystkie punkty. Brak ratowania wyniku przez wspólną średnią C10/C100 lub najlepszy seed. Progi są decyzyjne, nie testem istotności. |
| Limit/stop | ≤200 konfiguracji operator×szerokości/checkpoint, ≤1600 łącznie, ≤120min na4GPU (≤8GPU-h), z wliczonym profilowaniem i oceną. Niespełnienie którejkolwiek bramki kończy próbę. Bez kolejnej siatki, zmiany miary lub zwiększenia limitu na tych wynikach. |

**Sprawdzone teraz wyłącznie na CPU:** formuła geometrii odtwarza P/MAC U50
z zamrożonego E037. W menu jest1857/1759 profili dopuszczalnych (C10/C100),
w tym564/565 zmian co najmniej3 grup względem k⁰ wykorzystujących≥90% obu
limitów. Anchor k⁰ jest dopuszczalny; z jego79sond39/38 spełnia oba limity.
Przewidywanie kombinacji≥3 zmian pozostaje ryzykownym przybliżeniem drugiego
rzędu. Nie wyznaczono jakości żadnego nowego modelu. Sondy poza limitami nie
są konkurencyjnymi modelami do wdrożenia.

## 4. Ocena niezależna i reguła powrotu do artykułu

Po spełnieniu **wszystkich** kryteriów: zamrożenie implementacji, listy
porównań, profili, checkpointów i hashy **przed** osobnym otwarciem rezerwy
CIFAR-100. Rezerwa5000 jest kandydatem do jednorazowej oceny potwierdzającej,
dopiero w następnym etapie i po wyraźnej decyzji o jej użyciu. Te same limity,
cztery sparowane seedy i kryterium praktyczne; bez ponownego doboru alokacji.
Dla niezależnego twierdzenia o CIFAR-10 potrzebny jest nowy zewnętrzny
holdout, np. CIFAR-10.1 po ustaleniu wersji i protokołu przed dostępem.
Testy E035 mogą być co najwyżej oceną wtórną na danych już użytych.

W tym etapie ani pilocie **nie otwieramy rezerwy CIFAR-100**. Cztery GPU mają
przyszłe jawne przypisanie seed42/43/44/45→GPU0/1/2/3, na każdym oba zbiory
kolejno, Conda `attention`. Wagi pozostają własne; nowy trening wymaga osobnej
zgody. BN korzysta z danych; MAC nie jest pomiarem latencji.

**Wynik negatywny lub brak wkładu ablacyjnego ⇒ powrót do redakcji artykułu
na E035–E039**, z dotychczasowym zakresem twierdzeń. Sam lepszy surrogate,
wygrana tylko na jednym zbiorze albo efekt samego dostosowania TILT do MAC
nie uzasadniają rozszerzenia. Następny krok to decyzja o wdrożeniu tego pilota.
