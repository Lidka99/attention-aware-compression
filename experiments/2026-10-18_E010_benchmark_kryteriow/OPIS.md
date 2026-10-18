# E010 — benchmark kryteriów selekcji przy równych budżetach naprawy

Eksperyment: `2026-10-18_E010_benchmark_kryteriow`.
Data utworzenia protokołu: 2026-10-18 UTC.
Status: **frozen** (protokół 1.0, nieuruchomiony). Status wykonania podaje
manifest biegu; testy CPU: [TESTY_CPU_v1.json](TESTY_CPU_v1.json) — passed.

## Cel i pytanie

Czy któreś z prostych, standardowych kryteriów wyboru kanałów do scalenia
daje po **krótkiej naprawie** (1–20 epok) istotnie lepszą jakość niż losowy
wybór — i czy efekt utrzymuje się na drugim datasecie?

E010 jest **przesiewem**, nie dowodem publikacyjnym. Ma rozstrzygnąć, która
miara w ogóle ma sygnał po naprawie, zanim zainwestujemy w pełną metodę,
trzy seedy i dłuższe budżety.

## Kontekst

- E008 (60 epok naprawy, Tiny 30%): POST−RANDOM +0,77 pp, CI [+0,03; +1,50] —
  na granicy istotności; POST−PRE +0,51 pp, CI obejmuje zero.
- E009 (120 epok naprawy, Tiny 30%): POST−PRE +0,28 pp [−0,41; +0,96],
  POST−RANDOM +0,25 pp [−0,48; +1,02] — różnice między PRE/POST/RANDOM
  znikają. Selekcja ma znaczenie tylko bez naprawy (etap 1 E008).
- E003/E004 (CIFAR-100 10%): po 120 epokach SGD różnice między selekcjami
  znikają; wybór losowy dorównuje.
- Literatura porównuje kryteria zero-shot albo po „wystarczającym"
  fine-tuningu; brak kontrolowanego porównania wielu kryteriów przy
  **tych samych krótkich budżetach** i wspólnym operatorze scalania.

## Hipotezy (ustalone przed biegiem)

- **H1 (pozytywna):** co najmniej jedno nowe kryterium (najbardziej
  kandydują: REC — rekonstrukcja LS, oraz TAYLOR) ma przy T=5 epok na
  Tiny 30% przewagę ≥ 1,0 pp nad RANDOM, istotną po korekcie Holma,
  i powtarza ją na CIFAR-100 10%.
- **H0:** żadne kryterium nie utrzymuje przewagi w oknie T ≤ 20 na
  Tiny ani na CIFAR-100.
- **Oba wyniki są informacyjne.** H1 otwiera ścieżkę metody (E011);
  H0 definitywnie zamyka wątek selekcji w badanym reżimie i pozwala
  opublikować benchmark + wynik negatywny oraz przejść do naprawy/operatora.

## Co porównujemy

Wspólny operator fizycznego foldingu E008/E003 (α do fc1 SE, β do projekcji;
E010 dopuszcza też, by reprezentant wchłonął wiele kanałów — dla k-means).
Dziewięć kryteriów:

| Kryterium | Typ | Źródło |
|---|---|---|
| PRE | podobieństwo aktywacji przed SE | modele etapu 1 E008/E003 (reużyte) |
| POST | podobieństwo aktywacji po SE | modele etapu 1 E008/E003 (reużyte) |
| RANDOM | kontrola losowa | modele etapu 1 E008/E003 (reużyte) |
| L1 | ważność kanału (norma L1 filtra) | nowe |
| BN-γ | ważność kanału (skala BN) | nowe |
| TAYLOR | wrażliwość loss (gradient × aktywacja) | nowe |
| APOZ | częstość zer aktywacji | nowe |
| REC | rekonstrukcja LS (reszta z momentów Grama) | nowe |
| KMEANS | grupowanie k-means wag po BN, medoid | nowe |

Przypisanie „usuwany → reprezentant" dla kryteriów per-kanał jest **stałe
dla wszystkich** (cosine filtrów po BN), żeby porównywać kryterium, a nie
sposób dopasowania. Współczynniki α/β zawsze z dopasowania LS E003/E008.

## Datasety i siatka

| Dataset | Baseline (od zera) | Ratio | Kryteria | Budżety T |
|---|---|---|---|---|
| Tiny ImageNet 224 | E007, dev 58,49% | 30% (primary) | 9 | 1, 5, 10, 20 |
| Tiny ImageNet 224 | E007 | 10% (walidacja) | PRE, POST, RANDOM, L1, REC, KMEANS | 5 |
| CIFAR-100 96 | E001, dev 66,84% | 10% (secondary) | 9 | 1, 5, 10, 20 |

Oba datasety trenujemy **od zera** na Waszych baseline'ach; nie używamy
publicznych wag. ImageNet-1k pozostaje poza zakresem (dysk 5,2 GB wolne,
~150 GB danych, ~59 h/100 epok jednego modelu na 4×2080 Ti); ewentualny
wariant pretrained to osobna decyzja w `docs/DECISIONS.md`.

## Procedura (skrót)

1. **Wejścia:** baseline'y, modele etapu 1, plany struktur, momenty Grama,
   kalibracja A (Tiny 1000, CIFAR 500) — wszystko pinowane hashami.
2. **Statystyki nowych kryteriów:** jeden przebieg 4 GPU na A (APoZ, Taylor);
   L1/BN-γ/k-means/REC liczone bez nowego przebiegu.
3. **Scalanie:** 6 nowych kryteriów na każdym ratio; weryfikacja liczby
   parametrów, rozłączności, determinizmu; zero-shot dev.
4. **Naprawa:** recipe E009 (AdamW 3e-4, cosine per T, warmup min(3,T),
   batch 32/GPU, 4 GPU DDP, seed 42), wynik = ostatnia epoka, bez early
   stoppingu. CIFAR-100 używa augmentacji E001 (RandomCrop 32+flip+Resize 96)
   i normalizacji 0,5; nie odtwarzamy receptury SGD z E004.
5. **Ewaluacja:** dev (Tiny 10 000, CIFAR 5000), sparowany bootstrap
   10 000, seed 20261018, stratyfikacja po klasach; Holm dla 6 nowych
   kryteriów przy T=5 (Tiny 30%).

## Kryterium decyzyjne (pre-registered)

**Primary:** Tiny 30%, T=5, nowe kryterium X przechodzi, jeśli
`dev_top1(X) − dev_top1(RANDOM) ≥ 1,0 pp` oraz jednostronne p z bootstrapu
po korekcie Holma < 0,05 (rodzina 6 kryteriów).
**Replikacja:** wynik dodatni musi powtórzyć się na CIFAR-100 10%, T=5
(CI 95%, dolna granica > 0); dla kryteriów obecnych w siatce Tiny 10%
także tam. Pozostałe T i pary raportowane jako eksploracyjne.

## Czego ten eksperyment nie rozstrzyga

- Jeden seed (42) — screening; publikacja wymaga ≥3 seedów (E011).
- Jedna architektura (EfficientNet-B0+SE) i jeden operator; ratio 10/30.
- KMEANS to grupowanie z kompensacją LS, nie średnia klastra z literatury.
- Reprezentant w E010 może wchłonąć wiele kanałów; to rozszerzenie operatora
  o udokumentowanej równoważności z E008/E003 dla par rozłącznych.
- Statystyki APoZ/Taylor z jednego zbioru kalibracyjnego A.
- CIFAR-100 używa recipe AdamW E009, nie SGD z E004; porównania są w obrębie
  E010, nie z historycznymi liczbami E004.
- Dev wybierał epokę E007/E001, więc nie jest niezależnym testem.
- Rezerwa i oficjalny test pozostają nietknięte.

## Koszt

- Implementacja + testy CPU: ~3–5 dni pracy.
- GPU: Tiny 30% ≈ 16,2 h; Tiny 10% ≈ 1,5 h; CIFAR-100 10% ≈ 2,5–3,5 h;
  statystyki/zero-shot/summary ≈ 1 h → **~21–22 h ciągłej pracy 4 GPU**.
- Dysk: ~2,2 GB pod `runs/E010/` (wolne ~5,2 GB).
- Ścienna skala: ~1 tydzień do pierwszego wyniku po zamrożeniu.

## Powiązane dokumenty

- [PROTOKOL_v1.md](PROTOKOL_v1.md) — warunki zamrożone i plan techniczny.
- [config.draft.json](config.draft.json) — pinowane wejścia i siatka.
- [E009 — naprawa 120](../2026-10-12_E009_naprawa_120/PROTOKOL_v1.md).
- [E008 — kompresja 224](../2026-10-12_E008_kompresja_224/PROTOKOL_v1.md).
- [E003 — kompresja CIFAR-100](../2026-10-09_E003_kompresja_10proc_parametrow/PROTOKOL_v1.md).
