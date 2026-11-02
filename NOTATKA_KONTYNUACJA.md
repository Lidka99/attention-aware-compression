# Kontynuacja — 2026-11-02

## Najnowsze: E030 — trzeci zbiór na operatorze autorów

**Completed, kryteria spełnione.** Run `runs/E030/20261102T013740Z_609b7737`,
analiza `analysis_20261102T014235Z/`. Tiny64,4seedy,~50%,
84 oceny+12 kontroli,exit0,2,0min. Upstream nie ma klasy modelu Tiny,
więc użyto jego **niezmienionego operatora** na naszym torchvision
ResNet18 (stem7×7+maxpool); to jawna granica twierdzenia.
LEGACY identyczny z upstream (ARI1, wagi0); LOCAL max0,36pp, ARI0,66–0,76
— zgodność empiryczna, nie bitowa.
TILT−UNIFORM na operatorze autorów **+9,408pp** (CMA) i **+9,178pp**
(ich repair_bn),4/4. Bias FC −0,025pp.
[WYNIKI](experiments/2026-11-02_E030_oficjalny_tiny/WYNIKI.md).
Razem z E029: **+4,00/+1,70/+9,41pp** na CIFAR-10/100/Tiny.
Dwa przerwane starty E030 to błędy kodu uruchomienia, bez ocen modeli.

## E029 — oficjalny operator sprawdzony (wpis poprzedni)

**Completed, dodatni audyt.** Run `runs/E029/20261102T005941Z_064da5c2`,
analiza `analysis_20261102T010349Z/`.4GPU/4seedy,168 ocen+24kontrole,
exit0,3,13min. [WYNIKI](experiments/2026-11-01_E029_oficjalny_operator/WYNIKI.md).

Upstream `osaukh/folding_as_projection`, commit
`694691556e8314a835178a741938dd3d4d80c171`, sparse clone
`runs/E029/upstream_6946915`, pliki niezmienione. Stub wyłącznie brakującego
opcjonalnego importu CLIP, żadnej zmiany matematyki CNN.
LEGACY E012/E014 ma identyczne wagi po zachowaniu biasu FC. LEGACY i LOCAL
E024 mają identyczny top1 względem upstream przy wspólnych szerokościach/BN.
LOCAL nie jest bitowo identyczny: drobne logit differences i wyjątki klastrów
opisane w raporcie. Natywny losowy bias FC mierzony osobno, ma mały efekt.

TILT−UNIFORM na operatorze upstream: CIFAR-10/100 **+3,995/+1,695pp**
z CMA oraz **+3,745/+1,460pp** z oficjalnym repair_bn; wszystkie4/4 dodatnie.
Nie jest to pełny pipeline/paper1:1: nasze wagi, dev,2zbiory/1budżet,
BN na wspólnym A4096crop/flip bez rotation15. Nie twierdzić, że audyt
obejmuje Tiny i30/70%. TILT per grupa to nasz jawny adapter upstream.

**Następny krok:** protokół drugiej architektury (np.PreActResNet18);
nowy trening wymaga wyraźnej zgody. Można również rozszerzyć audyt
upstream na istniejące Tiny/30/70% bez treningu. Nic nowego nie uruchomiono.
4GPU wolne,dysk około4,2GB; test/rezerwa nietknięte. Starsze wpisy poniżej
zachowano historycznie; „baseline nieuruchomiony” nie dotyczy już operatora.

## Najnowsze: E027 + E028 — linia search zatrzymana

**E027** (`runs/E027/20261101T143927Z_ce68f402`, analiza
`analysis_20261101T145857Z/`): CIFAR-100,200 ocen,4GPU,19,3min.
Zmiana celu KL→**disagreement**. NEW_SEARCH56,485% vs TILT55,550%
→**+0,935pp, 4/4**, ale **próg+1,0pp niespełniony o0,065pp**.
NEW−UNIFORM+2,630pp. Zgodność z dev: disagreement0,634>KL0,546>CE0,544
na każdym seedzie. Najlepszy z50 kandydatów bije TILT o1,66pp (4/4):
na CIFAR-100 headroom istnieje, selektor go nie domyka.

**E028** (`runs/E028/20261101T154700Z_086b311e`, analiza
`analysis_20261101T160617Z/`): transfer tej samej reguły na CIFAR-10,
176 ocen,19,1min. NEW93,650% vs TILT93,430% →**+0,220pp, 2/4**;
NEW−UNIFORM+4,215pp. **Pułap puli−TILT tylko+0,545pp**: TILT jest blisko
granicy osiągalnej alokacji. Kontrola BN teacher95,46→95,44%.

**Decyzja:** nie rozwijamy dalej „nowego optymalizatora” jako kierunku
głównego. Dwa zbiory, dwa cele, brak stabilnej przewagi nad TILT.
Stabilny wkład: alokacja per warstwa, TILT, kontrole negatywne, rola BN.
Następne kroki: **oficjalny baseline1:1**, potem **druga architektura**.
Nie zwiększać limitu search, nie stroić progu na użytym dev.
[WYNIKI E027](experiments/2026-11-01_E027_diagnostyka_celu/WYNIKI.md),
[WYNIKI E028](experiments/2026-11-01_E028_transfer_cifar10/WYNIKI.md).
Brak aktywnych zadań;4GPU wolne,dysk około4,3GB. Test/rezerwa nietknięte.
Nowy trening wymaga osobnej zgody. Poniższe wpisy są historyczne.

## E026 ukończony — wpis historyczny

Run `runs/E026/20261101T133903Z_bbd75e05`, analiza
`analysis_20261101T135139Z/`.4GPU/4seedy,200 ocen,exit0,9,65min.
[WYNIKI E026](experiments/2026-11-01_E026_alokacja_poprawione_bn/WYNIKI.md).

Jedyna zmiana wobec E024: A4096 do BN z crop/flip, CMA. Ten sam KL,
limit50, reguła i przestrzeń search; CPU replay potwierdził zgodność.
Wyniki U/T/OLD_SEARCH pod poprawioną BN użyte z E025, bez powtórek dev.

**Primary negatywny:** NEW55,155%, TILT55,550%, U53,855%, OLD54,715%.
NEW−TILT−0,395pp (1/4), NEW−U+1,300pp (3/4), NEW−OLD+0,440pp (3/4).
Profile zmienione4/4; KL_C lepsze vs TILT4/4, top1 tylko1/4. Naprawa BN
pomaga, ale nie usuwa rozbieżności celu KL i trafności.

**Następna hipoteza:** kryterium zachowania decyzji teacher na
nieetykietowanym train zamiast KL, osobny zamrożony protokół, ta sama BN
i koszt. Nie zwiększać automatycznie liczby prób, nie skalować obecnego
KL-search i nie wracać do miar grupowania. Hipoteza, nie obietnica poprawy.
Brak aktywnych zadań;4GPU wolne,dysk około4,4GB. Test/rezerwa nietknięte,
nowy trening wymaga osobnej zgody. Poniższe plany przed E026 są historyczne.

## Najnowsze: E025 ukończony — czytaj przed E024

Diagnostyka BN: `runs/E025/20261101T121011Z_9c3d4231`, analiza
`analysis_20261101T121334Z/`, audyt EMA post hoc `ema_audit/`.
176/176 nowych ocen+16 reuse z E024,4GPU/4seedy,4×exit0,2,84min.
[WYNIKI E025](experiments/2026-11-01_E025_diagnostyka_bn/WYNIKI.md).

- Przy tym samym A4096 i CMA crop/flip podnosi teacher69,405→75,140%
  (wyjściowo75,235%):+5,735pp. Większa próbka bez augmentacji tylko+0,135pp.
- EMA0.1 reset po32 batchach zachowuje0,0343 wariancji początkowej1;
  dla niektórych warstw to ponad100× wariancji retain. Modele dają1%.
  Zależność reset−retain potwierdzona na1280 statystykach, błąd≤4,26e−8.
- A/aug/CMA: U53,855%, T55,550%, stary SEARCH54,715%. SEARCH−TILT−0,835pp,
  1/4 dodatnich; lepsze KL/NLL nie gwarantują przewagi top1.
- To diagnostyka na już używanym dev, bez nowej alokacji. E024 primary
  pozostaje negatywny; nie nadpisywać historycznych wyników.

**Następny proponowany eksperyment:** zamrozić A4096+aug+CMA jako minimalną
poprawkę i sprawdzić search przy tym samym celu KL/limicie50 ocen, aby
odizolować wpływ BN. Nie wybierać wariantu BN według największej trafności
z tabeli, nie zmieniać równocześnie celu i przestrzeni szukania.
Brak aktywnych zadań,4GPU wolne,dysk około4,4GB. Test/rezerwa nietknięte;
nowy trening nadal wymaga osobnej zgody. Starsze plany poniżej są historyczne.

## Najnowsza decyzja i E024 — czytaj przed E022

Użytkowniczka wybrała **samodzielną implementację optymalizacji budżetu
per warstwa**, bez importowania starych skryptów. Zgoda na nieetykietowany
train do wyboru alokacji; to calibration-based, bez fine-tuningu.
E023 zachowany, anulowany przed jakimkolwiek runem.

**E024 completed, primary negatywny.**
Run `runs/E024/20261031T141558Z_971ba86e`, analiza w `analysis/`.
CIFAR-100/~50% parametrów,4 checkpointy,200 ocen pełnej sieci,4GPU,
exit0,9,77min. SEARCH−UNIFORM+1,235pp (4/4), SEARCH−TILT−0,175pp (1/4).
KL na osobnym train C poprawione na4/4; NLL lepsze, ale primary top1
nie spełnia zamrożonego progu. **Kontrola samego BN szkodzi teacher
o5,83pp**, więc nie skalować obecnego protokołu search na inne zbiory.
[Pełne wyniki](experiments/2026-10-31_E024_optymalizacja_alokacji/WYNIKI.md).

Następny proponowany krok: zamrożona osobna weryfikacja BN na teacher
i stałych profilach; dopiero potem decyzja o celu optymalizacji.
Nie uruchomiono kolejnego eksperymentu; GPU wolne, dysk około4,4GB.
Nowy trening nadal wymaga osobnej zgody. Test/rezerwa nietknięte.

Poniższe rekomendacje łagodnych budżetów to stan sprzed zmiany kierunku;
bieżąca decyzja powyżej ma pierwszeństwo.

## Aktualizacja E022 — czytaj najpierw

**Mechanizm ukończony, brak aktywnych zadań.**
[WYNIKI E022](experiments/2026-10-31_E022_mechanizm_alokacji/WYNIKI.md).
Run `runs/E022/20261031T111804_ccc634a5`, finalna analiza
`analysis_20261031T114947Z/`. 8/8 nowych zadań, GPU0–3, exit0;
147/147 pomiarów grupowych i36/36 ablacji łącznie z ponownym użyciem źródeł.

Ważne odkrycie inwentaryzacji: istnieje E020, wcześniej pominięty w statusie.
`runs/E020/20261031T055920Z_cb9ffaf_sens_e650f55f` ma49/49 Tiny,
`runs/E020/20261031T061615Z_cb9ffaf_abl_bbb35b29` ma28 ukończonych ablacji
i niezamknięty historyczny manifest. Nie wznawiać ani nie nadpisywać go:
8 brakujących pomiarów zapisano już w E022. Nie powtarzać eksperymentów.

Wnioski (CIFAR-10 / CIFAR-100 / Tiny):
- Przy75% kompresji pojedynczej grupy BN odzyskuje79,3/67,1/38,1% szkody.
- Rho głębokość–szkoda post:−0,028/+0,643/+0,091; wczesna wrażliwość
  nie jest uniwersalnym trendem post-BN.
- TILT nie spełnia silnej ochrony grup o dużej szkodzie post; przy globalnym
  50% rho ochrony:−0,039/−0,553/−0,099. CIFAR-100 ma krytyczne późne grupy.
- ANTI−UNIFORM przy50%:−32,52/−29,29/−15,72pp; 4/4 ujemnych na każdym.
  Odwrócenie kierunku nie pomaga CIFAR-100, a profile nie są addytywne.
- Rekonstrukcja przewiduje rangę szkody przy stałych25–75% (rho≥0,5),
  słabnie przy90%. AUTO50 pozostaje negatywny.

**Następna decyzja:** łagodne budżety10/20% (osobny protokół), następnie
oficjalny baseline. Nie uruchomiono kolejnego eksperymentu. Nowy trening
wyłącznie za zgodą użytkowniczki. Test/rezerwa nietknięte; nie wracać do miar.

Poniżej zachowana historia; bieżący stan i kolejność powyżej mają pierwszeństwo.

**Plan dalszych kroków:** [PLAN_2026-10-30_dalsze_kroki.md](PLAN_2026-10-30_dalsze_kroki.md).

## Jeden cel
Sprawdzić, czy przy tej samej liczbie parametrów **automatyczny rozdział
foldingu** zachowuje więcej jakości niż UNIFORM i stały TILT, bez treningu wag.

## Co rzeczywiście mamy
- E014: na CIFAR-10 TILT+0.30 poprawia wynik o +0,87/+3,77/+11,27 pp.
- E015: na CIFAR-100 poprawa +0,81/+1,96/+1,60 pp; zamrożone kryterium
  replikacji **niespełnione**. TILT+0.15 lepszy jako wariant wtórny.
- E016 S1: diagnostyka 98/98 przypadków ukończona, seed42 na dwóch zbiorach.
  Korelacje łączne mieszają różnice grup i poziomów kompresji; nie dowodzą
  jeszcze przydatności predyktora do alokacji całego modelu.
- E016 S2: **completed, wynik negatywny**, 8/8 na 4 GPU. AUTO−UNIFORM
  +2,58/−10,48 pp (CIFAR-10/100), AUTO−najlepszy TILT −1,66/−13,88 pp.
  G1/G2 NIE, G3 TAK. [Raport](experiments/2026-10-27_E016_wrazliwosc_foldingu/WYNIKI_S2.md).
- **E017: benchmark miar grupowania ukończony.** 42/42, 4 GPU. Kryterium
  niespełnione: standardized ≈ reference, bn_effective gorsza, cosine gorsza
  na CIFAR-100. Wyjątek post hoc: cosine **+8,14 pp** przy 50% na CIFAR-10.
  [Wyniki](experiments/2026-10-29_E017_miary_grupowania/WYNIKI.md).

## Co sprawdzamy teraz
Kierunek: **alokacja budżetu w foldingu** — najmocniejszy wynik projektu.
E014 (CIFAR-10) i E019 (Tiny ImageNet): TILT+0.30 bije UNIFORM, 12/12 par
dodatnich na obu zbiorach. E015 (CIFAR-100) słabszy (+0,81/+1,96/+1,60 pp).
Miary grupowania (E017/E018) zamknięte jako negatywne — nie stroić dalej.
Kolejny krok: mechanizm i druga architektura.

## Naprawa i kontrola wykonania
Prowadzący przejął zadanie bez podagentów. Nie uruchamiać równolegle
kolejnego launchera. Nadzór: `run_s2_supervised.py`, konkretny PID,
limit czasu i `.status.json`; bez `pgrep -f` i bez automatycznych restartów.
Szczegóły i wynik próby: `experiments/2026-10-27_E016_wrazliwosc_foldingu/NAPRAWA_S2.md`.
Aktualny stan naukowy i następny krok: `STATUS_BADAN.md`.

## Kolejność dalszej pracy
1. Naprawa i smoke obu zbiorów **ukończone: 8/8, 4 GPU, exit 0**;
   run `s2_smoke_20261028T211120Z_cb9ffaf_f1e27a35`. Brak aktywnych obliczeń.
2. E017 Etap 1 ukończony; pełny raport w `WYNIKI.md`. Run
   `runs/E017/20261029T022532Z_cb9ffaf_f50f95ea`, analiza
   `runs/E017/analizy/20261029T225633Z_bfd26c4d/`; exit 0, 457,6 s.
3. E018 ukończony: 24/24, exit 0, 262,4 s; **wynik negatywny**.
   Run `runs/E018/20261029T233216Z_cb9ffaf_3740f8f4`, analiza
   `runs/E018/analizy/20261029T233800Z_9ac7cdcc/`. Trop cosine zamknięty.
4. **E019 completed:** Tiny ImageNet, 4 seedy, folding 48/48.
   TILT+0.30 vs UNIFORM **+3,71/+10,32/+7,60 pp** przy 30/50/70%
   parametrów; 12/12 dodatnich; **kryterium spełnione**.
   Trening `runs/E019/20261030T000950Z_cb9ffaf_seeds42-45_4ef97df7`,
   folding `runs/E019/fold_20261030T220139Z_cb9ffaf_51bc405f`,
   analiza `runs/E019/analizy/20261030T222400Z_final/`.
   Baseline dev 50,82%; po foldingu TILT 38,12/30,08/16,14%,
   UNIFORM 34,41/19,77/8,54%. Bezwzględna jakość niska — ograniczenie.
5. **E021 completed:** 72/72, 3 zbiory × 8 metryk × 3 budżety.
   Żadna metryka (L1, L2, cosine, korelacja, Ward, standaryzacja) nie
   spełniła progu; L1 gorszy. Run `runs/E021/20261031T083247Z_cb9ffaf_d7f7cff9`,
   analiza `runs/E021/analizy/20261031T091111Z_6947373b/`.
6. **Audyt literatury:** `literature/AUDYT_ALOKACJA_FOLDING.md` — Wang i in.
   (2025) wprost zostawiają optymalizację sparsity per warstwa jako future work;
   *Cut Less, Fold More* używa jednolitych ratio.
7. Następny krok: mechanizm (bez treningu) i druga architektura.
   Żaden eksperyment nie jest uruchomiony; GPU wolne.

## Nienaruszalne wyniki
- E015 trening: `runs/E015/20261027T190911Z_cb9ffaf_seeds42-45_82c98b12`.
- E015 folding: `runs/E015/fold_20261027T210324Z_cb9ffaf_36a01249`.
- E016 S1: `runs/E016/s1_20261027T215948Z_cb9ffaf_d98f0ed9`.
- E016 S1 analiza: `runs/E016/analizy/20261027T230435Z_s1/`.
- Nieudany S2 smoke: `runs/E016/s2_smoke_20261028T203618Z_cb9ffaf_bc9f84f7`.
Oficjalny test i rezerwa pozostają nietknięte. Historycznych runów nie nadpisywać.
