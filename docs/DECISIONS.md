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
