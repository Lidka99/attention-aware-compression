# Rejestr decyzji

## D001 — 2026-09-25: oddzielne repo i dokumentacja przed obliczeniami

Utworzono lokalne repo w istniejącym folderze `badania_atencja_cnn`. Zachowano wcześniejszy plan. Prace dzielimy na E000–E003. Wszystkie pozostają `draft`; danych i architektury nie wybrano. Nowość jest warunkiem przejścia do badania własnego kryterium.

Każdy późniejszy wpis: data UTC, decyzja, uzasadnienie, dostępne wtedy wyniki, alternatywy, zmienione wersje protokołów. Decyzji po obejrzeniu wyników nie przedstawiać jako wcześniejszych założeń.

## Niezależny projekt — ustalenie 2026-09-26

Eksperymenty piszemy od nowa w tym repozytorium. Nie kopiujemy ani nie importujemy kodu, konfiguracji, wag czy wyników z attention_v2 ani attentionv3. Nie tworzymy zależności od tych folderów. Wcześniejsza propozycja użycia lokalnego checkpointu SE jest wycofana.

Dobór modelu i danych przeprowadzamy od początku. Możemy korzystać z udokumentowanych bibliotek i znanych architektur. Aktualna decyzja: trenujemy od losowych wag na 4 GPU w środowisku Conda attention, bez publicznych checkpointów.

## 2026-09-26 — uruchomienie pełnego pilota

Użytkowniczka wybrała CIFAR-100 i zleciła uruchomienie pełnego kodu. Zamrażamy konfigurację E001: 200 epok, 96×96, EfficientNet-B0 od losowych wag, seed=42, 4 GPU, Conda attention. Parametry config.json pozostają niezmienione. Wyniki trafiają do lokalnego runs/E001/; zewnętrzna kopia nie jest skonfigurowana. Start w odłączonym procesie, logi i checkpointy zapisuje istniejący launcher.

## 2026-10-11 — cel publikacyjny i kierunek po E004

Kontekst (dostępne wyniki): E003/E004 pokazały, że przy 10% redukcji
parametrów na przeuczonym baseline CIFAR-100 różnice PRE/POST/RANDOM
znikają po równym dostrajaniu; wybór losowy dorównał atencyjnemu.
E005 (Tiny ImageNet od zera, 96×96, 100 epok) powstaje jako nowy baseline.

Ustalenia użytkowniczki:
1. Celem ogólnym jest **publikowalny wynik** — obecne eksperymenty to
   faza przygotowawcza, nie materiał na artykuł.
2. **Później przejdziemy na 224×224**, żeby zbliżyć się do protokołów
   literatury i oczekiwanej jakości (~60% scratch vs nasze ~51–52% przy 96).
3. Kolejny baseline powinien mieć **mocniejszą recepturę przeciw
   przeuczeniu**: mixup/cutmix, random erasing, label smoothing,
   ewentualnie silniejszy weight decay — lista do ustalenia w protokole.

Konsekwencje:
- E005 pozostaje uczciwie opisany jako baseline przy 96×96 z ograniczeniem
  przeuczenia (train ~98% vs dev ~51%).
- Decyzja o kolejności (najpierw szybki test 30% na obecnym baseline, czy
  od razu baseline 224 z mocniejszą recepturą) zostaje otwarta i wymaga
  wyboru użytkowniczki.
- Nie zmieniamy zamrożonego E005 w trakcie jego trwania.

## 2026-10-11 — E006: poprawka losowania par po nieudanym biegu 1.0

Kontekst: bieg `runs/E006/20261011T051120Z_dc7b47f_seed42_tl2bv200`
(protokół 1.0, commit dc7b47f) zakończył się niepowodzeniem na rank 0:
`sample_disjoint_pairs: niedobor par w kubelkach [3, 4]; osiagnieto
[30, 30, 30, 10, 4]`. Żadna szkoda nie została policzona — bieg padł przed
pomiarem, więc zmiana nie może być zależna od wyników.

Dowód z zapisanych aktywacji tego biegu (1000 obrazów A, ten sam seed):
`sample_disjoint_pairs` używał kubełków **równych szerokości**, a rozkład
`s_post` jest skrajnie skupiony (mediana 0,0076; 110 216 z 114 960 par
w jednym kubełku). Najwyższy kubełek miał 276 par na 24 kanałach — maks.
12 par rozłącznych przy wymaganych 30. Schemat był niewykonalny, nie jest
to problem danych ani modelu.

Decyzja (za zgodą użytkowniczki, rozmowa 2026-10-11): kubełki
**kwantylowe** (równej liczności, 5×30 par), K=150 i kryterium oceny bez
zmian; `pairs.binning = "quantile"` w configu, protokół 1.1. Na tych samych
aktywacjach każdy kubełek ma ~22 992 kandydatów i pojemność ~230 par
rozłącznych; symulacja algorytmu daje 5×30 = 150 par i 300 kanałów.
Skrócono też timeout NCCL (30→10 min) — tylko szybsze sprzątanie po
błędzie, bez wpływu na wynik.

Bieg 1.0 pozostaje udokumentowany jako failed. Ponowny pomiar: nowy katalog
w `runs/E006`, ten sam baseline i te same seedy.

## 2026-10-11 — E006 zakończony: wynik negatywny dla kryterium bramek SE

Bieg `runs/E006/20261011T053842Z_eb6b660_seed42_c5gen8pp` (protokół 1.1,
commit eb6b660, 150 par / 300 kanałów, bez dev/rezerwy/testu): kontrola
dodatnia ρ(`s_post`, szkoda) = −0,155 (kierunek zgodny), ρ(`g_pert`) =
+0,044, Δρ = +0,199 z CI [−0,020; +0,411] → `primary_passed = false`;
korelacja cząstkowa `g_pert` po `s_post` = +0,047. Sygnały wtórne na
poziomie szumu (kontrola FWER p = 0,081).

Konsekwencja zgodna z protokołem: **wątek bramek SE jako sygnału ponad
`s_post` zostaje zamknięty**; brak podstaw do pełnej kompresji E007 z
kryterium bramkowym. Dalszy kierunek — (a) szybki test 30% redukcji na
obecnym baseline vs (b) baseline 224×224 pod publikację — pozostaje
decyzją użytkowniczki (zapisana w AKTUALNY_PLAN.md).

Dokumentacja: [WYNIKI.md](../experiments/2026-10-11_E006_kryterium_bramkowe_predykcja/WYNIKI.md).
Przebieg: nieudany bieg 1.0 `...051120Z`, przerwany start techniczny
`...053110Z` (oba zachowane), bieg końcowy `...053842Z`.

## 2026-10-12 — E007 zakończony: baseline 224×224

Kontekst (dostępne wyniki): E005 dał baseline 96×96 na 51,53% dev, z
przeuczeniem (train ~98,5%). Zgodnie z celem publikacyjnym (uzgodnienia
2026-10-11) uruchomiono E007 — mocniejszy baseline 224×224, receptura
oparta na Shahriar 2026 (arXiv:2607.01984; ich scratch 100 ep = 60,54%),
zaadaptowana do 4 GPU.

Wynik biegu `runs/E007/20261011T061825Z_4de51b3_seed42_24mzpz80`
(protokół 1.0, commit `4de51b3`, status completed, exit 0; start
2026-10-11T06:18:25Z, koniec 10:55:32Z, ~4h37m, ~165 s/epokę): 100/100
epok, wynik główny = `last.pt` (epoka 100) dev top-1 **58,49%**
(5849/10000); najlepszy dev 58,58% (epoka 98); train top-1 99,50%
(89531/89984); dev NLL 2,0001. Powyżej E005 (51,53%), poniżej kotwicy
Shahriar 2026 (60,54%; porównanie jakościowe — ich ocena na oficjalnym
val, nasza na dev).

Decyzja/konsekwencje:
- Otrzymaliśmy docelowy **baseline 224×224 pod publikację** i punkt
  odniesienia do badań kompresji; E005 pozostaje jako baseline pomocniczy
  96×96.
- **Decyzja o kolejności dalszych prac nadal należy do użytkowniczki**:
  (a) szybki test 30% redukcji na tym baseline (kandydat E008) vs
  (b) inne warianty (mocniejsza receptura, dodatkowe seedy).
- Zamrożone protokoły E001–E006 pozostają bez zmian; rezerwy i testu nie
  używamy.

Dokumentacja: [WYNIKI.md](../experiments/2026-10-11_E007_baseline_224/WYNIKI.md).

## 2026-10-13 — E008 zakończony; E009 (120 epok naprawy) w toku

Bieg runs/E008/20261012T042932Z_0fd9962_seed42_tn1zlyim (protokół 1.1, commit 0fd9962) zakończony 2026-10-12 15:57 UTC (exit 0). Etap 1 bez naprawy: 10% → PRE 47,42 / POST 47,07 / RANDOM 40,79; 30% → PRE 9,67 / POST 6,65 / RANDOM 0,99 (BASE 58,49, bramka odtworzona); selekcja > losowa, ale POST < PRE (odwrotnie niż na CIFAR-100). Etap 2 po 60 epokach równej naprawy: BASE 59,45 / POST 58,32 / PRE 57,81 / RANDOM 57,55; POST−PRE +0,51 pp, CI [−0,18; +1,19] → kryterium pierwszorzędne nieprzekroczone; POST−RANDOM +0,77 pp, CI [+0,03; +1,50] — jedyna istotna przewaga nad losowaniem, na granicy; wszystkie skompresowane poniżej BASE (o 1,13–1,90 pp).

Decyzja użytkowniczki (2026-10-12): dłuższy budżet naprawy jako osobny eksperyment — E009: 120 epok, jeden ciągły cosine, te same modele startowe (etap 1 E008), cztery warianty równo. Bieg runs/E009/20261012T155859Z_fef2836_seed42_jsagnut2 wystartował automatycznie 2026-10-12 15:59 UTC (launcher wymaga ukończonego E008); koniec ~13.10 ~14:00.

Dokumentacja: [WYNIKI.md E008](../experiments/2026-10-12_E008_kompresja_224/WYNIKI.md); [PROTOKOL_v1.md E009](../experiments/2026-10-12_E009_naprawa_120/PROTOKOL_v1.md).

## 2026-10-13 — E009 przerwany awarią maszyny; pełne powtórzenie

Bieg `runs/E009/20261012T155859Z_fef2836_seed42_jsagnut2` (protokół 1.0, commit
`fef2836`) został przerwany awaryjnym wyłączeniem maszyny 2026-10-13 ok. 09:18 UTC
(sygnał 1) w trakcie ostatniego wariantu RANDOM (epoka 11/120). Stan dowodowy:
BASE 60,04 / PRE 59,18 / POST 59,46 po 120/120 epok (kompletne metryki i predykcje
dev), RANDOM nieukończony, brak `summary.json`. Manifest biegu oznaczono jako
`failed`; żadne liczby z tego biegu nie wchodzą do analizy.

Decyzja użytkowniczki (2026-10-13, rozmowa): pełne powtórzenie E009 w nowym katalogu,
ten sam zamrożony protokół 1.0, ten sam kod i wejścia; bez zmian protokołu, rezerwy
i testu nie używano. Powtórzenie: bieg
`runs/E009/20261013T172339Z_954ece0_seed42_nh02g2l8` (commit 954ece0) wystartował
2026-10-13 17:23 UTC; pierwsza epoka BASE odtworzyła wartości przerwanego biegu
(train NLL, dev NLL i dev top-1 identyczne); oczekiwany koniec ~2026-10-14 ~17:00 UTC.

Dokumentacja: [PROTOKOL_v1.md](../experiments/2026-10-12_E009_naprawa_120/PROTOKOL_v1.md).
