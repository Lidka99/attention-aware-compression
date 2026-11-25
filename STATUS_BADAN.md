# Status badań — 2026-11-24

**Kontynuacja w nowym czacie:** [prompt](PROMPT_NOWY_CZAT.md) · [notatka](NOTATKA_KONTYNUACJA.md).

**Aktualny priorytet po decyzji użytkowniczki: mocniejsze modele na CIFAR-100 i Tiny ImageNet.**
[E048](experiments/2026-11-24_E048_mocniejsze_modele_bazowe/WYNIKI.md) trwa:
`runs/E048/train_20261124T232822Z_89dfc621`, cztery GPU i cztery seedy na zbiór,
najpierw CIFAR-100, potem Tiny. CPU check i smoke obu fal PASS; pełna trafność
jeszcze nieznana. Cel dev: ≥77,50% / ≥59,00%. Na Tiny stem 3×3/s2 bez maxpool
zwiększa rozdzielczość pierwszego etapu, ale podnosi koszt modelu pełnego z
148,15 do 555,52 mln MAC/obraz. Mierzymy te koszty jawnie.

[E049](experiments/2026-11-24_E049_folding_mocniejszych_baz/WYNIKI.md) jest w kolejce
`runs/E049/eval_20261124T234559Z_b4ca081b` po technicznym audycie E048, bez zależności
od tego, czy E048 przekroczy cele jakości. CPU potwierdził 20/20 konstrukcji na
checkpointach smoke. Główny kontrast: zamrożony L06/b10−U przy dwóch pułapach
na obu zbiorach; kontrole T-proj i powtórzenie porównań U/T przy b30/b50/b70.
[Teza i warunki jej obrony](PLAN_PUBLIKACJI.md#aktualny-kierunek-po-decyzji-2026-11-24).

**E047 ukończony — L06/b10 potwierdzony na zewnętrznej próbie ImageNetV2.**
2 000 obrazów, 200 klas Tiny, **+1,3500 pp (4/4)** vs U, sparowany 95%
bootstrap **[+0,700; +2,025] pp** warunkowo na modelach. Pełny/U/L06:
**30,3375/26,9875/28,3375%**; strata U/L06 **3,3500/2,0000 pp**. Te same
modele i pułapy co w E046. `runs/E047/20261122T053804Z_ae2dec73`: 4GPU/exit0,
56,25 s, 12/12 stanów E046 sprawdzone przed punktacją, audyt PASS.
[Wyniki](experiments/2026-11-22_E047_l06_b10_imagenetv2/WYNIKI.md) ·
[figura b10](experiments/2026-11-22_E047_l06_b10_imagenetv2/wykresy/20261122T053804Z_ae2dec73/README.md) ·
[szkic 0.9 do recenzji wewnętrznej](SZKIC_ARTYKULU.md). ImageNetV2 E047 odtąd użyty.

**E046 ukończony — L06 daje +1,1425 pp (4/4) przy b10 na Tiny dev.**
Top‑1 **47,5050%**, sparowana strata **3,7150 pp** wobec 4,8575 pp dla U.
**10 104 855 P / 132 302 192 MAC** pod oboma pułapami E045; geometria z
jawną granicą `k ≤ n`. `runs/E046/20261122T045655Z_0b0c53b7`:
4GPU/exit0, 62,10 s, 8/8 odtworzeń E045 i 4 oceny L06, audyt PASS.
[Wyniki](experiments/2026-11-22_E046_l06_b10_tiny/WYNIKI.md) ·
[hierarchia dowodów](PLAN_PUBLIKACJI.md) · [szkic 0.7](SZKIC_ARTYKULU.md).

**E045 ukończony — łagodny UNIFORM na Tiny dev.**
[Wyniki i figura](experiments/2026-11-22_E045_lagodny_uniform_tiny/WYNIKI.md):
pełny **51,2200%**, b10 **46,3625%** (strata **4,8575 pp** przy redukcji
**10,35% P / 10,67% MAC**), b20 **40,5800%** (strata **10,6400 pp** przy
**20,33% P / 19,97% MAC**). Porównywalne U/b30 i U/b50 z E035 **dev**.
`runs/E045/20261122T031350Z_a58b56a3`: 12/12 nowych ocen, 4GPU/exit0, 62,78 s,
4/4 pełne modele zgodne z E035, audyt PASS. [Szkic 0.6](SZKIC_ARTYKULU.md).
Priorytet: zachowanie jakości; 50% redukcji P nie jest wymogiem.

**E044 ukończony — dodatni transfer L06 na Tiny w b50.**
[Wyniki](experiments/2026-11-22_E044_l06_tiny/WYNIKI.md): L06−U
**+2,100 pp (4/4)**, bootstrap po obrazach [+1,67; +2,52] pp, warunkowo na
czterech modelach. W b30 **+0,255 pp (3/4)**. L06 b50 ma 21,740% wobec
50,7775% pełnego modelu. Tiny poza wyborem L06, val użyty wcześniej w E035.
`runs/E044/20261122T022955Z_6c1d2fc5`: 28/28, 4GPU/exit0, 124,97 s, audyt PASS.
[Szkic](SZKIC_ARTYKULU.md): E043 potwierdzenie na rezerwie, E044 transfer.

**E043 ukończony — L06 potwierdzony w b50, próg transferu b30 niespełniony.**
[Wyniki](experiments/2026-11-21_E043_l06_potwierdzenie_transfer/WYNIKI.md):
rezerwa CIFAR-100, L06−U po CMA **+1,215 pp (3/4)** przy ~50% i
**+0,945 pp (4/4)** przy ~30% redukcji P (próg +1,00 pp). L06−T-proj
**+1,065/+0,950 pp**, w obu 4/4. Strata L06 do nieskompresowanego:
**19,970/7,420 pp**. `runs/E043/20261122T000341Z_9e23e96f`, 28/28 ocen,
4GPU/exit0, 108,08 s, audyt PASS, 16/16 stanów b50/teacher zgodnych z E042.
**Rezerwa CIFAR-100 jest użyta.** L06 to referencja przy b50; ewentualny
test selektora wymaga osobnego protokołu i niewidzianych danych.

**E042 ukończony — jest zapas pod pułapami UNIFORM.** [Wyniki](experiments/2026-11-21_E042_zapas_pod_uniform/WYNIKI.md):
L06 vs U po CMA na dev **C100 +1,175 pp (3/4, próg spełniony)**,
**C10 +1,475 pp (4/4, kontrola)**. Rzeczywiste P/MAC w [99,9%, 100%] U.
Na C100 tylko 5/24 profili z dodatnią średnią; mediana −1,265 pp.
Run `runs/E042/20261121T212257Z_207ae03f`: 216 ocen, 723,49 s, 4GPU/exit0,
audyt PASS, bez treningu, testu i rezerwy. To najlepszy profil z ustalonej
puli wybrany na dev. Zamrożony profil oceniono następnie w E043;
[plan](PLAN_PUBLIKACJI.md), [nowe figury i szkic](SZKIC_ARTYKULU.md).

**E041 ukończony — wynik mieszany przy dopasowanych P i MAC.**
[Wyniki](experiments/2026-11-15_E041_kontrola_p_mac/WYNIKI.md):
TILT−średnia4kontroli po CMA: C10 **+0,11250pp,2/4**, C100
**+2,10375pp,4/4**. Łączne primary niespełnione przez CIFAR-10;
repair analogicznie+0,12750/+1,96375pp. Kontrola104 naC10 lepsza
od TILT o0,3850pp,4/4. **Tiny nie był objęty tym pilotem.**
Run `runs/E041/20261116T040008Z_9e34f1e9`:4GPU/exit0,227,054s,
168ocen, w tym96nowych kontrolnych; audyt i przeliczenie30kontrastów PASS.
Bez treningu, testu/rezerwy i nowych plików wag. Próbę zamknięto zgodnie
z protokołem; ewentualny Tiny to osobne rozszerzenie, nie zmiana bramki.

E041 jest kontrolą w [polskim szkicu artykułu](SZKIC_ARTYKULU.md). ATP (2025) i
Linear Decay (2020) są precedensami liniowego profilu: TILT to lokalna
nazwa/adaptacja, nie nowa ogólna zasada alokacji. Szczegóły i źródła:
[aktualizacja oceny wkładu](OCENA_WKLADU_I_ILUSTRACJI.md).

Ograniczona próba mocniejszej metody E040 została rozstrzygnięta negatywnie.
[Wyniki E040](experiments/2026-11-09_E040_interakcje_po_bn/WYNIKI.md):
model par po BN ma1,765/2,332×MAE modelu addytywnego na C10/C100,
na każdym zbiorze0/4seedów z poprawą; bramka predykcji niespełniona.
Run `runs/E040/20261111T032100Z_07ac54f7`,4GPU/exit0,3032,72s,
1558konstrukcji/1550CMA, audyt PASS. **Stop przed dev**, bez treningu,
testu/rezerwy i nowych plików wag; stany RAM zwolnione, GPU wolne.
Nie badano przewagi top1. Próba zamknięta bez kolejnych wariantów.
[Plan artykułu](PLAN_PUBLIKACJI.md),
[historyczny projekt i related work](NOTATKA_PROJEKTOWA_2026-11-09.md).

## Cel

Znaleźć metodę kompresji opartą na foldingu, która przy tym samym rozmiarze
sieci zachowuje więcej jakości **bez fine-tuningu**, i pokazać ją na tle
*Cut Less, Fold More* oraz MAG1/MAG2.

## Najważniejszy wynik

**E039 ukończony — uporządkowany TILT bije średnią4 ustalonych przestawień
w9/9 komórek, każda4/4 seedy.** CMA30/50/70: C10+1,69/+7,67/+21,16pp,
C100+1,25/+7,40/+10,67pp, Tiny+6,29/+12,78/+10,00pp. Repair także9/9.
Parametry dopasowane, ale SHUFFLED ma0,433–0,910×MAC(T): **nie izoluje
kolejności od compute**. Run `runs/E039/20261108T223038Z_9d2d039e`,
4GPU/exit0,684,24s;432 oceny+72kontrole, audyt PASS, bez testu/treningu.
[Wyniki E039](experiments/2026-11-08_E039_przestawienia_tilt/WYNIKI.md).

**E038 ukończony — TILT pomaga także pruningowi, brak ogólnej specyficzności
dla foldingu.** Przy identycznych profilach FOLD/MAG2 i50% parametrów:
I=ΔF−ΔMAG **−2,80/−4,375/−3,025/−1,97/+3,84pp** (R/C10,C100,Tiny,P/C10,C100);
primary tylko1/5. Średni ΔMAG dodatni15/15 CMA; FOLD/T nadal lepszy
od MAG2/T15/15. Run `runs/E038/20261108T215451Z_a5912565`,4GPU/exit0,
340,86s;360MAG+120FOLD+60teacher, audyt PASS, bitowa kontrola MAG120/120.
[Wyniki E038](experiments/2026-11-08_E038_operator_alokacja/WYNIKI.md).

**E037 ukończony — brak ogólnej przewagi TILT przy zbliżonym MAC.** Kontrola
na dev, bez treningu i testu. UNIFORM dopasowany do MAC TILT (U') ma
**1,10–1,82× więcej parametrów**; U' bije U we wszystkich15 komórkach.
T−U' po CMA jest dodatni tylko **5/15**, a bramkę przechodzi tylko
R/C10/30 (+0,37pp). Na CIFAR-100/Tiny U' wygrywa do+23,69pp; na CIFAR-10
TILT pozostaje nieco lepszy przy30/50%, ale ResNet/C10/70 przegrywa
−2,01pp. Run `runs/E037/20261108T031759Z_855c0b33`:4GPU/exit0,292,28s,
180 ocen+60kontroli, audyt PASS. **E034/E035 pozostają ważne dla stałego
budżetu parametrów. U' ma więcej parametrów, więc E037 nie izoluje
wpływu compute od pojemności modelu i nie dowodzi przyczyny zysku TILT.**
[Wyniki E037](experiments/2026-11-08_E037_kontrola_mac/WYNIKI.md).
[Wykresy w folderach eksperymentów i audyt zapisów](reports/2026-11-08_wykresy/README.md).

**E036 ukończony — przewaga TILT nie jest darmowa.** Audyt kosztu na
istniejących wagach, bez treningu i danych. Przy dopasowanych parametrach
TILT ma **1,105–1,823× MAC** UNIFORM (+10,5% do+82,3%) i **0,702–0,948×
przepustowości batch128** (−5,2% do−29,8%); średnie ilorazy latencji batch1
0,965–1,046, SD0,019–0,100, bez dowodu równoważności. Peak batch128 na
Tiny/50–70% wyższy o68–82%. Run
`runs/E036/20261107T231027Z_27bac186`:4GPU/exit0,789,50s,840 plików czasów,
audyt PASS. **Nie wolno pisać „matched FLOPs/compute”**: budżet parametrów
nie jest budżetem obliczeń. To ogranicza zakres twierdzenia, nie unieważnia
wyników jakości. [Wyniki E036](experiments/2026-11-07_E036_audyt_kosztu/WYNIKI.md).

**Ponowny audyt zapisów:** kompletność runów E034–E037 i przeliczone
kontrasty potwierdzone; checkpointy E0348/8 zgodne z SHA256. Poprawiono
błędne SD latencji i parametry w tabeli E036 oraz nadmiernie przyczynową
interpretację E037. Surowych wyników nie zmieniano. Nieaktualne `running`
w historycznych przerwanych runach nie oznacza aktywnych procesów.
[Dowody i odtworzenie audytu](reports/2026-11-08_wykresy/AUDYT_WYKONANIA.md).

**E035 ukończony — niezależny test, primary CMA15/15; każda komórka4/4.**
Pięć par,4seedy,30/50/70%, U/TILT, bez treningu. Oficjalne testy CIFAR10/100
i val Tiny po10k. Dev360+60 oraz test360+60, po4GPU/exit0,1018,04s;
dev top1 dokładnie odtworzone przed testem, audyty PASS.
TILT−U CMA30/50/70 [pp]: ResNet/C10 **+0,8100/+3,4125/+10,9875**,
ResNet/C100 **+0,8275/+1,3025/+2,0600**, Tiny **+3,5375/+9,3600/+7,4825**;
PreAct/C10 **+0,9600/+4,9500/+13,5775**, PreAct/C100 **+4,4875/+7,6425/+22,7375**.
**Osobny próg≥2pp przy50/70:9/10**, wyjątek ResNet/C100/50. Repair14/15:
ResNet/C100/70 nadal tylko2/4 dodatnie; NLL CMA gorsze TILT przyC100/50/70.
Przy70 T top1 ResNet/C10032,68%, Tiny15,88%, PreAct/C10045,7425% — duże
straty bezwzględne. Kierunek potwierdzony, nie uniwersalność/optimum.
Protokół f42c0b8, kod5a7ec92; run `runs/E035/20261107T183831Z_8b0b14e0`.
**Testy CIFAR i val Tiny odtąd użyte; E035 nie użył rezerwy CIFAR-100.**
Rezerwę wykorzystano później w E043. Tiny wcześniej otwierano do kształtu
E005/E007, bez oceny modeli. Wpisy „test/rezerwa nietknięte” są historyczne.
[Wyniki E035](experiments/2026-11-07_E035_ocena_niezalezna/WYNIKI.md).
Następny krok: opracowanie zakresu twierdzeń i materiału publikacyjnego;
nie dostrajać na obejrzanym teście. Nowy trening wymaga osobnej zgody.

**E034 Phase 1 ukończony — primary spełniony na obu CIFAR.** PreActResNet18,
4seedy, baseline95,215/76,905%; trening8/8 po200 epok,4GPU/exit0.
Folding144 ocen+24kontrole,4GPU/exit0,179,07s; audyt PASS.
TILT−UNIFORM przy30/50/70%: CIFAR-10 **+0,985/+4,880/+13,190pp** CMA,
CIFAR-100 **+4,435/+7,225/+23,095pp** CMA; każda komórka4/4 dodatnie.
Repair_bn: +1,020/+4,920/+13,230 oraz +4,600/+7,175/+23,310pp, także4/4.
Primary6/6 i secondary6/6. Transfer na drugą architekturę potwierdzony;
nie dowód uniwersalności ani mechanizmu. CIFAR-100/70 TILT46,52% wobec
baseline76,905%. [Wyniki E034](experiments/2026-11-05_E034_preactresnet18/WYNIKI.md).

**E033 — prosta hipoteza późnych grup niepotwierdzona.** 432 oceny
+36kontroli,4GPU/exit0,539,07s, bez treningu. EP (TILT na grupach0–8,
UNIFORM na9–11) vs T na CIFAR-100/b70/CMA: **+0,135pp, 2/4** — primary
niespełniony; repair_bn +0,950pp,3/4. LC−T −1,855/−1,395pp, eksploracyjne.
EP−U +1,755/+1,630pp. Na CIFAR-10/Tiny EP−T **−7,135/−5,065pp**,0/4;
EP nadal bije U. Zgodność LEGACY/UP24/24 (wagi max0, ARI1).
**Usunięcie dodatkowej kompresji późnych grup przy dopasowanym budżecie
nie odzyskuje zysku TILT; prosta narracja „CIFAR-100 przez layer4”
zamknięta jako niewystarczająca.** Profile hybrydowe są sprzężone budżetem
(EP dociska grupy6–8), więc to nie czysta izolacja.
[Wyniki E033](experiments/2026-11-05_E033_interakcja_pozne_grupy/WYNIKI.md).

**E032 — replikacja późnej wrażliwości ukończona, H1/H2 spełnione.**
288 ocen+24kontrole,4GPU/exit0,149,79s, bez treningu. CIFAR-100 po CMA:
rho głębokość–szkoda0,622–0,629, L−E+16,48 do+17,54pp,4/4seedów;
kontrolny CIFAR-10 rho−0,224 do+0,077, L−E ujemne4/4.
Kontrast rho średnio+0,677. Późne grupy tracą20,64pp vs0,96pp.
**Nie wyjaśnia to rozrzutu zysku TILT:** profile CIFAR-100 bardzo podobne
(rho między seedami0,958–0,986), L−E vs zysk E031/b70 ma rho0,00 CMA
i+0,20 repair_bn. Replikacja profilu, nie pełne wyjaśnienie przyczynowe.
[Wyniki E032](experiments/2026-11-04_E032_replikacja_wrazliwosci/WYNIKI.md).

**E031 — audyt30/70% ukończony, wynik mieszany.** Trzy zbiory,4seedy,
504 oceny+36kontroli; finalny run `runs/E031/20261104T212951Z_6c3d7356`,
4GPU/exit0,399+33 reuse i105+3 nowe. LEGACY ma identyczne wagi/partycje
z upstream; wszystkie24 bramki jakości LEGACY/LOCAL spełnione.
TILT−UNIFORM przy30/70% **redukcji parametrów** (CMA): CIFAR-10
**+1,025/+10,910pp**, CIFAR-100 **+0,620/+1,620pp**, Tiny **+3,780/+7,7475pp**.
Przy repair_bn: **+1,035/+9,260**, **+0,925/+0,680**, **+3,6925/+7,335pp**.
**Kierunek11/12: CIFAR-100/70%/repair_bn tylko2/4 dodatnie — łączne
kryterium E031 niespełnione.** CMA przechodzi6/6, ale CIFAR-100 jest słaby;
przy70% TILT osiąga33,02% CIFAR-100 i16,18% Tiny. Profile TILT bieżącego
solvera różnią się od historycznych, szczególnie przy30%.
[Wyniki E031](experiments/2026-11-02_E031_oficjalny_30_70/WYNIKI.md).

**E030 — oficjalny operator na Tiny ukończony, kryteria spełnione.**
Niezmieniony operator upstream na naszym torchvision ResNet18/Tiny64
(upstream nie ma wariantu64×64),4seedy,~50%,84 oceny+12 kontroli,exit0,2,0min.
LEGACY E012/E014 **identyczny** z upstream (wagi, ARI1); LOCAL E024 zgodny
jakościowo (max0,36pp), ale nie bitowo (ARI0,66–0,76). TILT−UNIFORM na
operatorze autorów: **+9,408pp** (CMA) i **+9,178pp** (ich repair_bn),4/4.
Bias FC nieistotny (−0,025pp). [Wyniki E030](experiments/2026-11-02_E030_oficjalny_tiny/WYNIKI.md).

**E029 — audyt oficjalnego operatora ukończony, dodatni wynik.**
Niezmieniony upstream CLFM commit6946915 na naszych checkpointach
CIFAR-10/100,4 seedy,~50% parametrów. LEGACY E012/E014 odtwarza wagi
upstream po zachowaniu biasu; LEGACY/LOCAL mają identyczny top1 przy
wspólnych profilach/BN. TILT−UNIFORM: **+3,995/+1,695pp** (CMA) oraz
**+3,745/+1,460pp** (oryginalne repair_bn), zawsze4/4 na każdym zbiorze.
168 ocen+24kontrole,4GPU/exit0,3,13min. To kontrola operatora, **nie pełna
reprodukcja artykułu1:1**: nasze wagi/dev i wspólne A4096 crop/flip.
[Wyniki E029](experiments/2026-11-01_E029_oficjalny_operator/WYNIKI.md).

**E028 — transfer reguły na CIFAR-10 ukończony, przewagi brak.**
176 ocen,4GPU/4seedy,exit0,19,1min, bez treningu. Ta sama zamrożona reguła
co E027 (wybór po disagreement), ten sam operator i BN.
NEW_SEARCH **93,650%**, TILT93,430%, UNIFORM89,435%:
NEW−TILT **+0,220pp (2/4)** — próg+1pp niespełniony; NEW−UNIFORM+4,215pp.
**Pułap najlepszego z50 kandydatów−TILT to tylko+0,545pp**: na CIFAR-10
TILT jest blisko granicy osiągalnej alokacji. Kontrola BN: teacher
95,46→95,44%. **Linia „nowy optymalizator search” nie daje stabilnej
przewagi nad TILT na dwóch zbiorach.** Stabilny wkład pozostaje po stronie
alokacji per warstwa i prostej reguły TILT.
[Wyniki E028](experiments/2026-11-01_E028_transfer_cifar10/WYNIKI.md).

**E027 — diagnostyka funkcji celu ukończona, primary niespełniony o0,065pp.**
CIFAR-100/~50%,200 ocen,4GPU,19,3min. Zmiana: wybór po **disagreement**
zamiast KL. NEW_SEARCH **56,485%** vs TILT55,550%: **+0,935pp, 4/4**
(próg+1pp); NEW−UNIFORM+2,630pp (4/4). Zgodność miar z dev top1:
disagreement0,634 > KL0,546 > teacher_ce0,544, na każdym seedzie.
Najlepszy z50 kandydatów bije TILT o1,66pp (4/4) — headroom istnieje na
CIFAR-100, ale selektor go nie domyka.
[Wyniki E027](experiments/2026-11-01_E027_diagnostyka_celu/WYNIKI.md).

**E026 — search z poprawioną BN ukończony, primary negatywny.**
CIFAR-100/~50%,4seedy,200 ocen na4GPU,exit0,9,65min. Jedyna zmiana
wobec E024: augmentacja A do BN. NEW_SEARCH **55,155%**, TILT55,550%,
UNIFORM53,855%, stary SEARCH pod tą samą BN54,715%.
NEW−TILT **−0,395pp (1/4)**, NEW−UNIFORM+1,300pp (3/4),
NEW−OLD+0,440pp (3/4). Profile zmienione4/4. KL_C lepsze niż TILT4/4,
ale top1 tylko1/4. Naprawa BN nie wystarcza: następna hipoteza dotyczy
zgodności funkcji celu z decyzjami, nie zwiększania limitu search.
[Wyniki E026](experiments/2026-11-01_E026_alokacja_poprawione_bn/WYNIKI.md).

**E025 — diagnostyka BN ukończona, przyczyna spadku E024 zidentyfikowana.**
176 nowych ocen+16 reuse,4GPU/4seedy, exit0. Crop/flip przy4096 obrazach
i CMA podnosi teacher **69,405→75,140%** (wyjściowo75,235%):+5,735pp.
Samo zwiększenie liczby próbek do35904 daje tylko+0,135pp. Krótka EMA0.1
po resecie załamuje modele do1%; audyt potwierdził pozostały wpływ
inicjalizacji wariancji. Przy A/aug/CMA stary SEARCH nadal przegrywa
z TILT o**0,835pp**, choć ma lepsze KL/NLL. Następny protokół: odizolować
wpływ poprawionej BN na sam search, bez zmiany celu i limitu prób.
[Wyniki E025](experiments/2026-11-01_E025_diagnostyka_bn/WYNIKI.md).

**E024 — samodzielna optymalizacja alokacji ukończona, primary negatywny.**
Cel: otwarty problem budżetu per warstwa; pełnosieciowy search KL na
nieetykietowanym train, CIFAR-100/~50%,4 seedy,200 ocen na4GPU, bez treningu.
SEARCH−UNIFORM **+1,235pp (4/4)**, SEARCH−TILT **−0,175pp (1/4)**.
KL na holdout train i NLL poprawione; kryterium≥+1pp vs TILT niespełnione.
Kontrola BN obniża teacher o**5,83pp**: przed dalszym search trzeba
zweryfikować nową procedurę BN. Kod niezależny od wcześniejszych skryptów;
metoda z kalibracją, nie calibration-free. E023 anulowany przed runem.
[Wyniki E024](experiments/2026-10-31_E024_optymalizacja_alokacji/WYNIKI.md).

**E022 (mechanizm) ukończony:** 147/147 pomiarów grupowych z kontrolami,
36/36 ablacji ANTI-TILT; wykorzystano E016 i pominięty wcześniej E020,
uzupełniając tylko8 brakujących zadań na4GPU (exit0), bez treningu.
BN przy75% kompresji grupy odzyskuje79,3/67,1/38,1% szkody
(CIFAR-10/100/Tiny). Wrażliwość post-BN rośnie z głębokością tylko wyraźnie
na CIFAR-100 (ρ=0,643); TILT **nie** silnie chroni grup o dużej szkodzie
post-BN. ANTI−UNIFORM przy globalnym50%: **−32,52/−29,29/−15,72pp**,
4/4 ujemnych na każdym zbiorze. Rekonstrukcja przewiduje rangę szkody
przy stałych25–75%, słabiej przy90%; nie potwierdza AUTO50.
[Wyniki E022](experiments/2026-10-31_E022_mechanizm_alokacji/WYNIKI.md).

**E019 (Tiny ImageNet) ukończony: kryterium spełnione.** TILT+0.30 vs
UNIFORM: **+3,71 / +10,32 / +7,60 pp** przy 30/50/70% parametrów;
**12/12** par seed×budżet dodatnich. Baseline dev 50,82%; po foldingu
TILT 38,12/30,08/16,14%, UNIFORM 34,41/19,77/8,54%; MAG2 znacznie gorszy.
To trzeci zbiór potwierdzający kierunek E014. Bezwzględna jakość po
foldingu jest niska — ograniczenie aplikacyjne.
[Wyniki E019](experiments/2026-10-29_E019_tinyimagenet_alokacja/WYNIKI.md).

**E021 (metryki grupowania) ukończony: wynik negatywny.** 72/72 zadania,
3 zbiory × 8 wariantów × 3 budżety. **Żaden** wariant (L1, L2, cosine,
korelacja, Ward, standaryzacja) nie spełnił progu +0,5 pp vs reference
i vs random przy 50/70% na wszystkich zbiorach; L1 był wyraźnie gorszy.
[Wyniki E021](experiments/2026-10-31_E021_metryki_grupowania/OPIS.md).

**E018 (replikacja cosine) ukończony: wynik negatywny.** Na seedach 43–45
CIFAR-10/50%: średnio **+0,47 pp**, dodatnie 2/3 (próg +2,0 pp i 3/3).
Przy 30% cosine gorsza; CIFAR-100/50% dodatnia, ale sprzeczna z seed42
(−3,72 pp). **Trop cosine zamknięty.**
[Wyniki E018](experiments/2026-10-29_E018_replikacja_cosine/WYNIKI.md).

**E017 (miary grupowania) ukończony: kryterium nie spełnione.** Przy
identycznym operatorze i budżecie, `standardized` ≈ reference, `bn_effective`
wyraźnie gorsza, `cosine` gorsza na CIFAR-100. Wyjątek: **+8,14 pp** dla
cosine przy 50% kanałów na CIFAR-10 (78,62 vs 70,48) — obserwacja **post hoc**,
jeden seed; wymaga replikacji na seedach 43–45. 42/42 zadania, 4 GPU.
[Wyniki E017](experiments/2026-10-29_E017_miary_grupowania/WYNIKI.md).

**E016 S2 ukończony: AUTO50 nie spełniło kryterium sukcesu.** Przy ~50%
redukcji parametrów, seed42: AUTO−UNIFORM **+2,58 pp (CIFAR-10), −10,48 pp
(CIFAR-100)**; względem najlepszego TILT **−1,66/−13,88 pp**.
8/8 zadań, cztery GPU, integralność OK. Brak aktywnych obliczeń.
[Wyniki S2](experiments/2026-10-27_E016_wrazliwosc_foldingu/WYNIKI_S2.md).

**E016 S1 (diagnostyka, nie nowy trening) ukończony:** na CIFAR-10 i
CIFAR-100 seed42 pojedyncze grupy foldingu mają bardzo różną wrażliwość.
Szkoda post-BN przy 75% wynosi średnio **2,82 pp** (CIFAR-10) i **9,54 pp**
(CIFAR-100); BN odzyskuje większość szkody przed-BN, ale nie całą.
Predyktor primary `weight_reconstruction_rel_error` ma Spearmana **0,800 /
0,702** ze szkodą post-BN. `depth_norm` sam praktycznie nie przewiduje
szkody na CIFAR-10 (ρ≈0). Bramki C1–C5 spełnione; **S2 (automatyczna
alokacja 50%) ukończony z wynikiem negatywnym opisanym powyżej**.
[WYNIKI E016](experiments/2026-10-27_E016_wrazliwosc_foldingu/WYNIKI.md).

Wcześniej: **E015 (CIFAR-100) nie spełnił zamrożonego kryterium
replikacji.** TILT+0.30 vs UNIFORM: **+0,81 / +1,96 / +1,60 pp** przy
30/50/70%; 50/70% poniżej progu +2,0 pp. E014 na CIFAR-10 pozostaje mocny:
**+0,87 / +3,77 / +11,27 pp**, 12/12 dodatnich.

## Eksperymenty — skrót

| ID | Pytanie | Wynik | Status |
|---|---|---|---|
| E001 | baseline EfficientNet+SE / CIFAR-100 | dev 66,84% | completed |
| E002 | czy podobieństwo kanałów przetrwa SE | obserwacja, bez dowodu korzyści | completed |
| E003 | kompresja 10% bez naprawy | POST−PRE +0,30 pp (CI 0); duża szkoda | completed |
| E004 | równe dostrojenie 120 epok | różnice znikają; RANDOM dorównuje | completed |
| E005 | baseline Tiny ImageNet 96×96 | dev 51,53% | completed |
| E006 | bramki SE jako predyktor szkody | brak wartości ponad `s_post` | completed |
| E007 | baseline Tiny ImageNet 224×224 | dev 58,49% | completed |
| E008 | kompresja 30% + 60 epok naprawy | POST−PRE +0,51 pp (CI 0) | completed |
| E009 | naprawa 120 epok | POST−PRE +0,28 pp (CI 0); różnice znikają | completed |
| E010 | 9 kryteriów selekcji | tylko TAYLOR, bez replikacji | completed |
| E011 | 4 checkpointy ResNet18/CIFAR-10 | dev 95,28–95,60%; średnia 95,46% | completed |
| E012 | folding bez fine-tuningu | po BN 94,20/89,30/70,50% (10/30/50%) | completed |
| E013 | alokacja budżetu, 1 seed | TILT+0.30: +0,58/+4,34/+10,70 pp | completed |
| E014 | replikacja na 4 seedach | +0,87/+3,77/+11,27 pp; 12/12 dodatnich | completed + analiza |
| E015 | alokacja na CIFAR-100 | +0,81/+1,96/+1,60 pp; 11/12; kryterium nie spełnione | completed |
| E016 | diagnostyka i AUTO50 | S1 98/98; S2 8/8, AUTO przegrywa z TILT na obu zbiorach | completed, S2 negatywny |
| E017 | miary grupowania kanałów | 42/42; kryterium nie spełnione; cosine +8,14 pp tylko CIFAR-10/50% (post hoc) | completed |
| E018 | replikacja cosine, seedy 43–45 | 24/24; +0,47 pp CIFAR-10/50%, 2/3; 30% gorsza; trop zamknięty | completed, negatywny |
| E019 | Tiny ImageNet 64×64, TILT | +3,71/+10,32/+7,60 pp; 12/12; kryterium spełnione | completed |
| E021 | metryki L1/L2/cosine/korelacja/Ward | 72/72; żaden nie spełnia progu; L1 gorszy | completed, negatywny |
| E022 | mechanizm, BN, ochrona i ANTI-TILT | kierunek ważny; prosta ochrona szkody post niepotwierdzona; rekonstrukcja działa przy stałych25–75% | completed |
| E023 | łagodne budżety — projekt na starych helperach | anulowany na życzenie użytkowniczki przed uruchomieniem | cancelled, brak wyników |
| E024 | niezależny search alokacji pełnej sieci | +1,235pp vs UNIFORM, −0,175pp vs TILT; KL/NLL lepsze, kontrola BN−5,83pp | completed, primary negatywny |
| E025 | diagnostyka BN na stałych profilach | augmentacja odzyskuje5,735pp teacher; większe A tylko0,135pp; krótka EMA reset zawodzi | completed |
| E026 | search KL z poprawioną BN | −0,395pp vs TILT; +1,300pp vs UNIFORM; +0,440pp vs stary SEARCH | completed, primary negatywny |
| E027 | cel: disagreement vs KL vs teacher CE | +0,935pp vs TILT (4/4), próg+1pp niespełniony; disagreement najlepiej zgodny z dev; pułap+1,66pp | completed, primary niespełniony |
| E028 | transfer reguły E027 na CIFAR-10 | +0,220pp vs TILT (2/4); +4,215pp vs UNIFORM; pułap tylko+0,545pp | completed, transfer nieudany |
| E029 | oficjalny operator i rozdzielenie BN/biasu | zgodność top1; TILT dodatni na upstream i ich repair_bn,2 zbiory/4seedy/~50% | completed, dodatni audyt |
| E030 | oficjalny operator na Tiny64 | +9,408/+9,178pp TILT−UNIFORM (CMA/repair_bn),4/4; LEGACY identyczny, LOCAL max0,36pp | completed, dodatni audyt |
| E031 | oficjalny operator30/70%,3zbiory | zgodność24/24; kierunek11/12, CIFAR-100/70%/repair_bn2/4 dodatnie | completed, wynik mieszany |
| E032 | profile75% pojedynczej grupy, CIFAR-100 vs CIFAR-10,4seedy | późna wrażliwość replikuje się4/4; nie wyjaśnia seed45/zysku TILT | completed, H1/H2 spełnione |
| E033 | interakcja wczesne/środkowe vs późne grupy, EP/LC | EP−T CIFAR-100/b70 +0,135pp CMA (2/4), repair_bn+0,95 (3/4); CIFAR-10/Tiny EP−T ujemne4/4 | completed, primary negatywny |
| E034 | PreActResNet18, CIFAR-10/100,4seedy | CMA +0,985/+4,880/+13,190 i +4,435/+7,225/+23,095pp; wszystkie4/4 | completed, primary6/6 |
| E035 | niezależne testy CIFAR10/100 i val Tiny | CMA kierunek15/15, każda4/4; próg≥2pp9/10 | completed, primary spełniony |
| E036 | koszt MAC/czas/pamięć przy matched params | MAC T/U1,105–1,823; przepustowość b128 0,702–0,948 | completed, audyt opisowy |
| E037 | UNIFORM dopasowany MAC (U' z większą liczbą parametrów) | T−U' CMA tylko5/15 dodatnich; bramka1/15; U' wygrywa C100/Tiny | completed, primary negatywny |
| E038 | FOLD/MAG2 × U/T, identyczne szerokości i BN | TILT pomaga MAG15/15; I przy50% dodatni1/5, brak ogólnej specyficzności foldingu | completed, primary negatywny |
| E039 | cztery stałe przestawienia odchyleń TILT | T−mean4 CMA9/9 i4/4; MAC przestawień0,433–0,910×T | completed, primary spełniony |
| E040 | model par szkody folding+BN pod P i MAC | MAE par/add1,765/2,332 na C10/C100,0/4poprawy; stop przed dev | completed, primary negatywny, kierunek zamknięty |
| E041 | cztery kontrole profilu, dopasowane P i MAC≤0,1% | T−mean4 CMA C10+0,11250pp(2/4), C100+2,10375pp(4/4); Tiny niebadany | completed, primary niespełnione łącznie |
| E042 | zapas jakości przy jednoczesnych pułapach UNIFORM, po 24 profile | najlepszy−U C100 +1,175 pp (3/4); kontrola C10 +1,475 pp (4/4) | completed, dodatni pilot dev |
| E043 | zamrożony L06 na rezerwie CIFAR-100, potwierdzenie b50 i transfer b30 | b50 +1,215 pp (3/4); b30 +0,945 pp (4/4), poniżej +1 pp | completed, główne kryterium spełnione |
| E044 | zamrożony L06 na istniejącym Tiny ImageNet | b50 +2,100 pp (4/4); b30 +0,255 pp (3/4) | completed, główne kryterium spełnione |
| E045 | łagodny UNIFORM, Tiny dev i krzywa jakość–koszt | b10 46,3625% (strata 4,8575 pp), b20 40,5800% (strata 10,6400 pp) | completed, opisowy |
| E046 | zamrożony L06 przy b10 pod dwoma pułapami, Tiny dev | 47,5050%; zysk +1,1425 pp (4/4), strata do pełnego 3,7150 pp | completed, dodatni pilot dev |
| E047 | replikacja L06/b10 na ImageNetV2 (200 klas Tiny) | +1,3500 pp (4/4), CI [+0,700; +2,025] pp; strata do pełnego 2,0000 pp | completed, kryterium potwierdzenia spełnione |

## E016 — wynik S1 (2026-10-27)

- Run: `runs/E016/s1_20261027T215948Z_cb9ffaf_d98f0ed9`; 98/98 itemów
  (CIFAR-10 49/49, CIFAR-100 49/49, kontrole 2/2), 4 GPU, 4 niezależne
  shardy na falę. Bez nowego treningu, bez testu i bez rezerwy.
- Średnia szkoda post-BN przy 25/50/75/90%: CIFAR-10
  **0,15 / 0,68 / 2,82 / 9,87 pp**; CIFAR-100
  **0,82 / 2,98 / 9,54 / 22,54 pp**.
- Najbardziej wrażliwe przy 75%: CIFAR-10 `layer3.0.relu2` (11,68 pp),
  `layer2.0.relu2` (9,38), `relu_conv` (4,56); CIFAR-100
  `layer4.0.relu2` (30,14), `layer3.0.relu2` (22,10),
  `layer4.1.relu1` (19,66).
- Predyktory vs szkoda post-BN: `weight_reconstruction_rel_error`
  **0,800 / 0,702**; `calib_logit_rmse_pre_bn` 0,786 / 0,883;
  `calib_logit_rmse_post_bn` 0,790 / 0,974; `depth_norm` **−0,011 / 0,443**.
- Bramki C1–C5 **TAK**; nie są dowodem skuteczności AUTO50. Korelacje
  łączne mieszają wpływ grupy i poziomu kompresji.
- Infrastruktura: dwa przerwane/poprawione biegi zachowane; pierwszy abort
  po 20 itemach, drugi ukończył CIFAR-10 i został wznowiony po poprawce
  `expected_controls=1`. Poprawka E015 `import os` sprawdzona.
- Analiza: `runs/E016/analizy/20261027T230435Z_s1/analysis.md`.

## Następny krok

**Aktualizacja po E038/E039 (pierwszeństwo nad planami poniżej):** obie
kontrole wzmacniają materiał empiryczny, ale zawężają nowość. TILT jest
użytecznym profilem także dla pruningu; kolejność/kształt ma znaczenie
w porównaniu parameter-matched, nadal sprzężone z MAC. Następny krok:
redakcja metody i artykułu z jawnymi E035–E039 oraz najbliższym related work,
bez twierdzenia o specyficznym mechanizmie foldingu, optimum lub przewadze
compute-matched. Testów nie używano ponownie; nowy trening wymaga zgody.

**Aktualizacja po E037 (ma pierwszeństwo nad planem historycznym poniżej):**
E034/E035 pokazują przewagę TILT przy stałej liczbie parametrów na dev
i teście, ale E036/E037 pokazują, że TILT ma1,1–1,8× MAC, a UNIFORM
z większą liczbą parametrów przy tym samym MAC wygrywa na CIFAR-100/Tiny.
**Twierdzenie artykułu trzeba ograniczyć do budżetu parametrów; nie ma
podstaw do „compute-matched” ani „lepszej alokacji na FLOP”.** Kolejny
krok: decyzja o ramie publikacji — uczciwe pokazanie obu porównań
(parameter-matched i MAC-matched) oraz kosztu, bez search i bez strojenia
na użytym teście. Alternatywa to nowy, jawnie kosztowy cel alokacji, ale
to nowy protokół i wymaga zgody; search pozostaje zamknięty.
Testy po E035 są użyte, rezerwa CIFAR-100 użyta po E043. Nowy trening
tylko po wyraźnej zgodzie. E031 mieszany, E033 negatywny, layer4 zamknięty.

**Audyt luki alokacji:** [literature/AUDYT_ALOKACJA_FOLDING.md](literature/AUDYT_ALOKACJA_FOLDING.md)
— Wang i in. (2025) wprost zostawiają optymalizację sparsity per warstwa jako
future work; *Cut Less, Fold More* używa jednolitych ratio.
**Pełny plan na nowy czat:** [PLAN_2026-10-30_dalsze_kroki.md](PLAN_2026-10-30_dalsze_kroki.md)
(mechanizm → łagodne budżety → oficjalny baseline → druga architektura).

1. **E022 ukończony.** Mechanizm częściowy: konflikt wczesnej ochrony
   i późnej wrażliwości CIFAR-100, BN zmniejszający przewagę nad UNIFORM;
   same profile pojedynczych grup nie wyjaśniają całej alokacji.
2. **Aktualna decyzja po E027/E028:** linia „nowy optymalizator search”
   **zatrzymana jako kierunek główny**. Na CIFAR-100 reguła disagreement
   dała najlepszy wynik search (+0,935pp, 4/4) poniżej progu; na CIFAR-10
   tylko wyrównała TILT (+0,220pp), a pułap puli był+0,545pp. Nie zwiększać
   limitu, nie stroić progu na użytym dev i nie skalować search na kolejne
   zbiory. Stabilny wkład: alokacja per warstwa + TILT + kontrole negatywne
   + rola BN. **E029 zweryfikował oficjalny operator** na CIFAR-10/100
   przy~50%; E030 dodał Tiny, a E031 rozszerzył zgodność na30/70%.
   Kierunek E031 nie przechodzi łącznie: CIFAR-100/70%/repair_bn2/4.
   Nie jest to pełna reprodukcja CLFM. Następny krok: koszt, przepis
   i kryteria drugiej architektury; nowy trening tylko za wyraźną zgodą.
   E023/łagodne budżety odłożone.
   E031, E032 i E033 zakończone. E032 potwierdził późną wrażliwość CIFAR-100
   na4seedach, ale nie jej związek z rozrzutem zysku TILT. E033 sprawdził
   interakcję przy profilach E031: EP−T na CIFAR-100/b70/CMA tylko+0,135pp
   (2/4), a na CIFAR-10/Tiny ujemne4/4. **Prosta hipoteza „chroń layer4”
   niepotwierdzona; nie budować na niej nowego alokatora.** Następny krok
   to koszt, przepis i kryteria drugiej architektury; nowy trening tylko
   po wyraźnej zgodzie. Nowego treningu nie uruchomiono.
3. Druga architektura i nowy trening wyłącznie po zgodzie użytkowniczki.
   Decyzja publikacyjna po dalszych kontrolach; nie na podstawie samych tabel.
4. E016 AUTO50 i miary E017/E018/E021 zamknięte. E015 pozostaje negatywny
   względem zamrożonego progu. Profile E022 mają jeden seed; E032 rozszerzył
   diagnostykę75% grupy na4seedy obu CIFAR z poprawioną BN, nadal na dev;
   oficjalny test i rezerwa pozostają nietknięte.

## Jak czytać dokumentację

- **Ten plik** — aktualny stan, jedna strona.
- **`PODSUMOWANIE.md`** w folderze eksperymentu — cel, wynik, wniosek.
- **`WYNIKI.md` / `OPIS.md`** — pełne tabele, protokoły i ograniczenia.
- Pełne raporty zostają jako archiwum; nie są potrzebne do codziennego czytania.
