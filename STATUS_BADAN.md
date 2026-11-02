# Status badań — 2026-11-02

**Kontynuacja w nowym czacie:** [NOTATKA_KONTYNUACJA.md](NOTATKA_KONTYNUACJA.md).

## Cel

Znaleźć metodę kompresji opartą na foldingu, która przy tym samym rozmiarze
sieci zachowuje więcej jakości **bez fine-tuningu**, i pokazać ją na tle
*Cut Less, Fold More* oraz MAG1/MAG2.

## Najważniejszy wynik

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
   przy~50%; nie stanowi pełnej reprodukcji CLFM ani audytu Tiny/30/70%.
   Następny krok: protokół drugiej architektury; nowy trening tylko za zgodą.
   E023/łagodne budżety odłożone.
   Nie uruchomiono żadnego kolejnego eksperymentu.
3. Druga architektura i nowy trening wyłącznie po zgodzie użytkowniczki.
   Decyzja publikacyjna po dalszych kontrolach; nie na podstawie samych tabel.
4. E016 AUTO50 i miary E017/E018/E021 zamknięte. E015 pozostaje negatywny
   względem zamrożonego progu. Profile E022 mają jeden seed i ocenę na dev;
   oficjalny test i rezerwa pozostają nietknięte.

## Jak czytać dokumentację

- **Ten plik** — aktualny stan, jedna strona.
- **`PODSUMOWANIE.md`** w folderze eksperymentu — cel, wynik, wniosek.
- **`WYNIKI.md` / `OPIS.md`** — pełne tabele, protokoły i ograniczenia.
- Pełne raporty zostają jako archiwum; nie są potrzebne do codziennego czytania.
