# 2026-11-09_E040_interakcje_po_bn — projekt ograniczonego pilota

**Status wykonania: nieuruchomiony.** Wersja projektu v1, 2026-11-09 UTC.
Decyzje metodyczne i kryteria poniżej ustalono przed nowymi pomiarami.
Nie ma jeszcze implementacji ani wykonawczego manifestu `frozen.json`.
Przed pierwszą oceną należy zahashować ten protokół, implementację i wejścia;
nie wolno dobierać wariantów na podstawie wyników pilota.
[Uzasadnienie i krytyczny przegląd literatury](../../NOTATKA_PROJEKTOWA_2026-11-09.md).

Przed realizacją: [uzupełnienie wykonawcze](UZUPELNIENIE_WYKONAWCZE.md)
wyjaśnia mapowanie nazw grup do upstream i przechowywanie stanów w RAM.

## 1. Pytanie i zakres

Czy bezpośrednio zmierzony drugi rząd szkody **folding + rekalibracja BN**
w skompresowanym otoczeniu przewiduje całe alokacje i daje stabilną, praktyczną
przewagę pod jednoczesnymi limitami P i MAC, ponad profil addytywny po BN,
profil par przed BN oraz mocne dopuszczalne U/TILT/MAG2?

Jeden budżet nominalny50% parametrów, ResNet18/CIFAR-10 i CIFAR-100,
seedy42–45. To ograniczenie odróżnia użyteczność modelu interakcji od rozszerzania
siatki architektur/budżetów. 50% jest środkiem dotychczasowego zakresu:
30% daje mniejszy margines jakości, 70% grozi dużą degradacją bezwzględną.
E028 wskazuje mały margines na C10 w **innym** zadaniu, przy samym budżecie P;
nie jest to górne ograniczenie dla niniejszych dwóch limitów i innej przestrzeni.

Własne checkpointy `seed_{42,43,44,45}/last.pt`:
- C10: `runs/E011/20261024T192205Z_018f3f2_seeds42-45_b2da2012/`;
- C100: `runs/E015/20261027T190911Z_cb9ffaf_seeds42-45_82c98b12/`.

Wybór `last.pt` stały; hashe z manifestów/artifacts należy sprawdzić przed oceną.
Bez nowych/publicznych wag, treningu i zależności od attention_v2/attentionv3.

## 2. Dane i stały operator

- Oryginalny split E011: train45000/dev5000; E015: train40000/dev5000/rezerwa5000.
- A/B/C: odpowiednio4096/2048/2048 rozłącznych indeksów wyłącznie z train.
  Odtworzenie C100: `E024/data.py`, permutacja RNG20261031; C10:
  `E028/data10.py`, RNG20261102. Są to źródła indeksów, nie zaproszenie do
  ponownego uruchomienia search. Manifest ma zawierać indeksy i ich SHA256.
- A do BN: crop32, padding4, horizontal flip; utrwalone widoki według
  istniejącej reguły E025/E028 (`20261101 + indeks obrazu`). Bez rotation.
  Reset running mean/var/licznika, momentum=None, 32batche×128, bez gradientów.
- B do wszystkich współczynników, wyboru alokacji i wyboru TILT*;
  B/C bez augmentacji, normalizacja istniejącego zbioru. Etykiety A/B/C nie są
  odczytywane przez alokator. C tylko do audytu po zamrożeniu wyborów.
- Dev: jeden etap oceny po bramce C, bez wyboru profilu ani metody.
  Dev oraz A/B/C były wcześniej używane w projekcie: to pilot rozwojowy,
  nie niezależny test. Testy E035 i rezerwa C100 pozostają poza loaderem pilota.

FOLD: operator upstream z audytu E029 (`694691556e8314a835178a741938dd3d4d80c171`),
ze zweryfikowanym adapterem szerokości i zachowaniem biasu FC. Stała kolejność
grup i ustawienia grupowania E029, w tym seed grupowania42; nie dobieramy
metryki ani inicjalizacji. Każda konfiguracja powstaje od oryginalnego
checkpointu, nie przez dalszy folding poprzedniego kandydata. MAG2: zweryfikowany
operator `E038/pruning.py`, ranking L2 oryginalnych filtrów, stabilny remis
po indeksie, te same grupy/kształty i zachowany bias. Wszystkie modele końcowe
przechodzą tę samą CMA na A, także ablacja wybierana z profilu przed BN.

## 3. Legalne grupy, koszty i dwie szerokości

Kolejność12 grup upstream i ich rozmiary n:

| Indeks | Grupa | n |
|---:|---|---:|
| 0 | relu_conv | 64 |
| 1,2 | layer1.0.relu1, layer1.1.relu1 | 64,64 |
| 3,4,5 | layer2.0.relu1, layer2.1.relu1, layer2.0.relu2 | 128,128,128 |
| 6,7,8 | layer3.0.relu1, layer3.1.relu1, layer3.0.relu2 | 256,256,256 |
| 9,10,11 | layer4.0.relu1, layer4.1.relu1, layer4.0.relu2 | 512,512,512 |

Zmieniamy całe grupy zależności: wspólne wyjścia residual/identity, downsample,
wejścia konsumentów i właściwe BN. FC ma stałą liczbę klas i wejście związane
z grupą11. Nie są to niezależne szerokości pojedynczych Conv.

`L_g=floor(0.6 n_g)`, `H_g=floor(0.8 n_g)`.
L=(38,38,38,76,76,76,153,153,153,307,307,307),
H=(51,51,51,102,102,102,204,204,204,409,409,409).
Menu ustalono dla ograniczenia kosztu identyfikacji, nie wybrano go z jakości.
Nie interpolujemy ani nie uzupełniamy kanałów poza wybraną konfiguracją.

P liczy **wszystkie** `nn.Parameter`, niezależnie od requires_grad, z BN γ/β
i biasem FC, bez buforów BN. MAC na obraz3×32×32: Conv `h_out w_out c_out
c_in k_h k_w` i Linear `c_in c_out`; 1multiply-add=1MAC, bez bias/BN/ReLU/pool.
Wzory uwzględniają iloczyny aktualnych szerokości obu końców każdej Conv,
w tym downsample. To dokładny koszt geometrii, nie addytywna suma oszczędności
pojedynczych grup. Implementacja ma odtworzyć `E024/Geometry.count` i
`E037/mac_plan`; kontrola na zrealizowanych modelach: numel i hooki Conv/Linear.

| Zbiór | U50 — stałe szerokości | P_max | MAC_max |
|---|---|---:|---:|
| C10 | (45,45,45,90,90,90,181,181,181,362,362,362) | 5 584 580 | 276 383 716 |
| C100 | (45,45,45,90,90,90,180,180,180,361,361,361) | 5 581 542 | 275 335 732 |

Źródło: `E037/frozen.json`, `plans.resnet_cifar{10,100}.b50` — rzeczywiste U
z E035. **Tolerancja przekroczenia0 parametrów i0MAC**; arytmetyka całkowita.
Brak dolnego limitu wykorzystania dla modeli końcowych. Raport obejmuje
P, MAC, ilorazy do limitów i obie luki. Wspólne limity to nie „matched P+MAC”.

Weryfikacja projektowa2026-11-09: czysto arytmetyczna enumeracja CPU, Python
Conda attention z pustym CUDA_VISIBLE_DEVICES, bez importu torch i bez I/O
danych/wag. Odtworzono dokładne koszty U50; teacher P/MAC wynosi
11 173 962/555 422 720 dla C10 i11 220 132/555 468 800 dla C100.
Wśród4096 profili dopuszczalne są1857/1759.

**Punkt odniesienia k⁰:** spośród dopuszczalnych k∈{L,H} wybieramy minimum
`Σ_g ((k_g−U50_g)/n_g)²`. Remis: większe P, mniejsze MAC, leksykograficzny
wektor v_g=1[k_g=L_g]. To projekcja geometryczna, bez danych i priorytetu
głębokości. Współrzędne z oznaczają **zamianę** szerokości względem k⁰:
`k_g(z)=k⁰_g+(L_g+H_g−2k⁰_g)z_g`, z∈{0,1}¹².
Każdy z4096profili menu nadal jest osiągalny. Zweryfikowane CPU:

| Zbiór | k⁰ | P(k⁰) | MAC(k⁰) | Sondy dopuszczalne /79 |
|---|---|---:|---:|---:|
| C10 | (38,51,51,102,102,76,204,204,204,409,307,307) | 5 246 333 | 275 824 206 | 39 |
| C100 | (38,51,51,102,76,102,204,204,153,409,409,307) | 5 426 105 | 273 062 716 | 38 |

Anchor spełnia oba limity, ale40/41z79sond jest poza nimi. To narzędzia
profilowania, nie konkurencyjne punkty jakości pod zadanym budżetem.
Przejście od sond≤2zmian do alokacji z≥3zmianami jest przybliżeniem,
którego powodzenie bada bramka C, nie domyślnie poprawną ekstrapolacją.
Teacher jest referencją nieskompresowaną, nie konkurentem spełniającym limity.

## 4. Definicja szkody, pomiary i alokacja

Dla operatora O∈{FOLD,MAG2}, oryginalnego teacher f₀ i zbioru B:

`D_O^t(z) = (1/|B|) Σ_x 1[argmax f_O^t(k(z);x) ≠ argmax f₀(x)]`,
gdzie pre oznacza natywne bufory po kompresji, post — rekalibrację R_A.
Teacher dla obu wersji jest **tym samym oryginalnym checkpointem w eval()**.
D to funkcjonalna niezgodność, nie prawdziwa strata trafności; może karać
poprawienie błędu teacher. Top1 na etykietach pozostaje kryterium użyteczności.

Pełny plan sond na operator/checkpoint: z=0, 12wektorów e_g i66par e_g+e_h,
razem79. Pozostałe grupy zachowują szerokości k⁰, czyli otoczenie jest już
skompresowane pod oba limity. Zmiany mogą zwężać albo poszerzać daną grupę.
Na tym samym B mierzymy pre i post; każda sonda ma własną CMA z identycznego A.

`a_g^t = D^t(e_g) − D^t(0)`;
`I_gh^t = D^t(e_g+e_h) − D^t(e_g) − D^t(e_h) + D^t(0)`;
`Q_2^t(z) = D^t(0) + Σ_g a_g^t z_g + Σ_{g<h} I_gh^t z_g z_h`.

Nie fitujemy kar, progów czy siły interakcji: współczynniki kontrastów mają
mnożnik1; ujemne interakcje pozostają. Q może wyjść poza[0,1]; nie obcinamy go
w optymalizacji i raportujemy ten objaw niedopasowania. Przeszukujemy CPU
wszystkie4096 kombinacji, odrzucając każdą naruszającą P lub MAC.
Remis: mniejsze Q, następnie większe P, mniejsze MAC, leksykograficzne z.
Jedna alokacja na model/wersję; bez kolejnych rund i bez wyboru najlepszego
z empirycznie ocenionych kandydatów po dopasowaniu modelu odpowiedzi.

Założenie: pomijalny trzeci i wyższe rzędy w tym menu. Koszt profilu
`1+G+G(G−1)/2`, tj.79 pełnych kompresji i rekalibracji, a nie79 tanich
lokalnych błędów wag. Koszt solvera `O(2^G G²)` jest mały tylko dla tego G.
Sondy obejmują również zmiany partycji klastrów indukowane innymi szerokościami.
I nie jest zatem czystym, uniwersalnym współczynnikiem mechanizmu kanałów.
`I_BN=I_post−I_pre` opisuje zmianę kontrastu przez ustaloną procedurę BN.
Nie wprowadzamy nowego operatora naprawy.

## 5. Porównania i ablacje

Wszystkie punkty konkurencyjne spełniają **oba** limity i otrzymują tę samą CMA.

1. **UNIFORM:** dokładnie U50 powyżej.
2. **TILT z dwoma limitami:** β∈{0,0.05,0.10,0.15,0.20,0.25,0.30},
   `d_g=−1+2g/11`, `r_g(a)=clip(a+βd_g,0.1,0.9)`,
   `k_g=floor(n_g(1−r_g))`. Dla każdego β wybór najszerszego dopuszczalnego
   profilu w tej monotonicznej rodzinie przez enumerację punktów zmiany
   szerokości w a; bez danych i bez fill według samego P. β=0 odtwarza U50.
3. **Projekcja TILT:** w tym samym menu4096 wybór dopuszczalnego z minimalnym
   `Σ_g ((k_g−k_g^T50)/n_g)²`, gdzie k^T50 to zamrożone T/b50 z E035.
   Remis: większe P, mniejsze MAC, leksykograficzne z. To znana projekcja
   dyskretna, dodatkowa kontrola kosztowa, nie nowa metoda. Zapobiega oparciu
   porównania wyłącznie na TILT, który po skalowaniu mocno nie wykorzystuje P.
   Kontrola CPU: C10 ma P4 882 317/MAC276 237 070; C100 P4 487 553/MAC269 508 220.
4. **Addytywny-post:** `Q_1^post = D^post(0)+Σ a_g^post z_g`, ten sam solver,
   menu i budżety. To bezpośrednia ablacja nowych interakcji i mocny kalibracyjny
   baseline. Identyczne sondy jednostkowe; praktyczny koszt samodzielny13,
   wobec79 dla par. Nie jest to pełna reprodukcja RR ani BALF.
5. **Pary-pre:** Q₂ z pomiarów przed BN; model końcowy nadal przechodzi CMA.
   Izoluje wartość profilowania po naprawie, nie samo stosowanie BN w ocenie.
6. **Pary-post:** proponowana reguła.

MAG2 otrzymuje niezależne79sond z własnego operatora i własne rozwiązania
addytywny-post oraz pary-post, oprócz tych samych8szerokości U/TILT.
Dla obu operatorów raportujemy też **k⁰, projekcję U50**, jako osobny punkt
odniesienia z już zmierzonych79sond. To kontrola wartości alokatora ponad
sam geometryczny wybór anchora. Łącznie8U/TILT oznacza7wartości β
(**w tym** UNIFORM jako β=0) plus1projekcję TILT, a nie8dodatkowych TILT.
Tym samym nie osłabiamy MAG2 przez narzucenie mu alokacji wybranej dla foldingu.
Nie wymaga się, aby efekt interakcji był specyficzny dla FOLD.

Na B zapisujemy także TILT* — najmniejsze D_post spośród8punktów U/TILT
osobno dla FOLD/MAG2, przy remisie mniejsze MAC, większe P, kolejność listy.
W raporcie dev pokazujemy **wszystkie** punkty, a nie wyłącznie TILT*.
Bramka praktyczna używa mocniejszej obwiedni wszystkich ustalonych punktów,
więc nie skorzysta na pomyłce selektora TILT*. Jest to porównanie po ocenie,
nie reguła wyboru modelu z etykiet dev.

## 6. Zamrożenie i dwie bramki

Przed jakimkolwiek odczytem C/dev zapisujemy wszystkie współczynniki B,
wybrane szerokości, TILT*, koszty i hashe stanów końcowych oraz16 masek
diagnostycznych. Implementacja nie może aktualizować wyborów na podstawie C.

**Bramka predykcji C, tylko FOLD:** ze zbioru dopuszczalnych z z co najmniej3
jedynkami i `min(P/P_max,MAC/MAC_max)≥0.90` bierzemy16masek o najmniejszym
SHA256 UTF-8 napisu `20261109|cifar10|<12bitów z>` albo odpowiednio `cifar100`.
Serializacja: dokładnie12znaków0/1 w kolejności z₀,z₁,…,z₁₁, bez spacji,
nawiasów i końca wiersza; e₀ zapisuje się `100000000000`. Sortowanie rosnąco
po32bajtach skrótu, remis po tak zapisanym z. z oznacza zamiany względem k⁰.
To stałe sondy diagnostyczne, nie nowe kandydatury do wyboru. Jest ich dość:
564/565 przed pobraniem16. Nie występują w79sondach dopasowania.
Mierzymy D_post na C i MAE prognoz Q₁_post/Q₂_post z B, bez ponownego fitowania.
`MAE_j = (1/16) Σ_{z w sondach} |Q_j^post(z) − D_C^post(z)|` na każdym seedzie.
Osobno na każdym zbiorze:
- średnia4seedów MAE₂≤0.80×średnia4seedów MAE₁;
- MAE₂<MAE₁ na co najmniej3/4seedów.
Jeśli średnia4seedów MAE₁=0, bramka jest niespełniona; nie dzielimy przez0.
Pojedynczy seed z MAE₁=0 nie może liczyć się jako poprawa, ale sam nie
unieważnia reguły3/4.
Spearman i rozkład I_BN są opisowe. Niespełnienie bramki na którymkolwiek
zbiorze kończy próbę przed dev.

**Bramka użyteczności dev:** Δ zawsze w pp, dla tych samych seedów.
Na **każdym** z obu zbiorów wymagamy łącznie:
- Pary-post FOLD minus **każdy** spośród20punktów odniesienia:8U/TILT FOLD,
  8U/TILT MAG2, k⁰ dla obu operatorów oraz własne addytywny-post/pary-post MAG2:
  średnia≥1.00pp, Δ>0 na≥3/4seedów, minimum Δ≥−0.50pp.
- Wkład interakcji: Pary-post FOLD minus addytywny-post FOLD:
  średnia≥0.50pp i Δ>0 na≥3/4seedów.
- Wkład profilowania po BN: Pary-post FOLD minus pary-pre FOLD:
  średnia≥0.25pp i Δ>0 na≥3/4seedów.

Średnia i SD(ddof=1) sparowanych różnic, komplet4punktów, bez wykluczania
seedów. Nie łączymy zbiorów w n=8 niezależnych powtórzeń; każdy zbiór ma4
checkpointy. Opcjonalne przedziały opisowe nie zmieniają bramek. Zapisujemy
top1, NLL, stratę względem teacher, P/MAC/luki, wszystkie wybory i koszty
profilowania. Dodatnia zgodność z teacher nie zastępuje poprawy top1.
Próg1pp zachowuje skalę praktyczną E024–E028; mniejsze progi ablacji0.50/0.25pp
oznaczają wymagany dodatkowy wkład interakcji/BN. To ustalone kryteria decyzji,
nie progi istotności statystycznej ani estymaty uzyskane z wyników E040.

**Sukces = obie bramki i wszystkie ablacje na obu zbiorach.** Wynik tylko
C100, tylko lepsza predykcja lub brak wkładu BN/interakcji nie uzasadnia
kontynuacji tego kierunku jako mocniejszej metody. Nie wybieramy po wynikach
między Q₁, Q₂_pre, Q₂_post i nową miarą; nie dokładamy trzeciego rzędu,
punktów menu, kandydatów ani innych budżetów, żeby uratować pilot.

## 7. Koszt, wykonanie i reguła zatrzymania

Na checkpoint:79FOLD+79MAG2+16weryfikacyjnych FOLD
+(8FOLD+8MAG2)U/TILT+3rozwiązania FOLD+2rozwiązania MAG2+1teacher
= **najwyżej196** konfiguracji **łącznie dla obu operatorów**.
Kontrole k⁰ są już w79sondach, bez dodatkowych kompresji/CMA.
Duplikaty ponownie wykorzystujemy. Dla profili do estymacji: pre-B2048,
CMA-A4096, post-B2048; C dodatkowo dla sond i końcowych wyborów. Dev ma
najwyżej23różnych modeli skompresowanych/checkpoint +teacher. Przechowanie
końcowych stanów na CPU pozwala uniknąć ponownej rekalibracji do oceny.

Twarde limity: **200konfiguracji (operator, szerokości) łącznie dla FOLD+MAG2
na checkpoint**, 1600
łącznie; najwyżej200przebiegów CMA/checkpoint łącznie z kontrolami. Całe
wykonanie na czterech GPU, w tym profilowanie, kontrole i ocena, ma limit
**120min czasu ściennego od startu workerów, ≤8GPU-h**. Optymalizacja CPU:
≤60s/checkpoint. To limity, nie zmierzone przewidywania czasu.

Conda `attention`, 4procesy: seed42→GPU0,43→GPU1,44→GPU2,45→GPU3;
każdy w danej fazie kolejno C10, potem C100. Globalne bariery po doborze na B
i po bramce C: dev dopiero po komplecie4seedów i PASS zagregowanych kryteriów
na obu zbiorach.
Wykorzystanie wszystkich czterech sprawdzane
w logach. Brak4urządzeń oznacza przeszkodę, bez zejścia na mniej GPU.
FP32, TF32off, deterministyczność i batch128 jak w audycie E038.
Zapisać modele GPU, środowisko, seedy, konkretne PID-y, przypisania, logi,
kody zakończenia i liczniki prac. Bez backward/optimizer/zmian uczonych wag
poza ustalonym operatorem kompresji; BN zmienia bufory i korzysta z danych.

Naruszenie integralności, budżetu lub limitu czasu przerywa całość; zachować
częściowe wyniki i status, bez wniosku o sukcesie oraz bez automatycznego
restartu/przedłużenia. Niespełnienie kryterium naukowego zamyka tę próbę
i przywraca redakcję artykułu E035–E039. Żadnego strojenia na obejrzanych
C/dev/test. MAC pozostaje liczbą operacji, nie wynikiem pomiaru latencji.

## 8. Odtworzenie i ewentualna niezależna ocena

Przed realizacją: osobna implementacja E040 w tym repozytorium, kontrola
geometrii i indeksów na CPU, manifest z SHA256 protokołu/kodu/checkpointów/
splitów/widoków A/masek. Następnie jawny launcher4GPU, zapis wyborów z B,
bramka C, jednorazowy etap dev, audyt i decyzja według §6. Polecenia
odtworzenia powstaną wraz z rzeczywistym kodem; ten dokument nie opisuje runu.

Wyłącznie po sukcesie: zamrozić kompletną metodę i potwierdzający protokół,
a po osobnej decyzji o dostępie wykonać jednorazową ocenę na **nieużytej
rezerwie CIFAR-1005000**, z tymi samymi czterema seedami, profilami, limitami
i kontrastami praktycznymi. Rezerwa nie służy ponownej selekcji. Jest to
niezależność danych oceny, nie nowe seedy treningowe. Dla C10 potrzebny
nowy zewnętrzny holdout, np. CIFAR-10.1 z wersją i regułą oceny zamrożonymi
przed dostępem. E035 nie stanie się ponownie niewidzianą walidacją.

Wynik pozytywny uzasadni osobny plan rozszerzenia na istniejący PreActResNet18
i porównanie z silniejszymi pełnymi pipeline'ami, w tym GRAIL/BALF, przed
jakimkolwiek twierdzeniem SOTA. Nowy trening wymaga odrębnej wyraźnej zgody.
