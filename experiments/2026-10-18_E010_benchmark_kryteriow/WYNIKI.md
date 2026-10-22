# E010 — wynik benchmarku kryteriów selekcji przy krótkiej naprawie

Data biegu: 2026-10-19 UTC. Dokument uzupełniony: 2026-10-21 UTC.
**Status wykonania: completed (exit 0; 2026-10-19 20:48 UTC).**
Protokół przed uruchomieniem: [PROTOKOL_v1.md](PROTOKOL_v1.md), v1.1 (frozen).
Bieg: **`runs/E010/20261018T235741Z_007be2d_seed42_824m539e/`**,
commit **`007be2d`**, dirty=false.

## Decyzja protokołu

**`survivor_not_replicated`.** Jedynym kryterium, które przeszło próg
pierwszorzędny, jest **TAYLOR**; jego przewaga **nie powtórzyła się** na
CIFAR-100 10%. Żadne kryterium nie uzyskało statusu potwierdzonego
„survivora". E010 pozostaje tanim screeningiem, nie dowodem publikacyjnym.

Kryterium pierwszorzędne (zamrożone): Tiny ImageNet 30%, T=5,
`top1(X) − top1(RANDOM) ≥ 1,0 pp` oraz jednostronne p po korekcie Holma
< 0,05 w rodzinie 6 nowych kryteriów.

## Primary — Tiny ImageNet 224, 30% redukcji, T=5

| Kryterium | dev top-1 | dev NLL | Δ vs RANDOM | 95% CI | p (jednostronne) | Holm |
|---|---:|---:|---:|---:|---:|---|
| **TAYLOR** | **55,61%** | 2,0375 | **+1,03 pp** | **[+0,26; +1,79]** | **0,0047** | **tak** |
| APOZ | 54,91% | 2,0301 | +0,33 pp | [−0,48; +1,13] | 0,2217 | nie |
| KMEANS | 54,64% | 2,0401 | +0,06 pp | [−0,72; +0,82] | 0,4438 | nie |
| RANDOM | 54,58% | 2,0466 | — | — | — | — |
| L1 | 54,38% | 2,0384 | −0,20 pp | [−1,00; +0,59] | 0,6970 | nie |
| BN_GAMMA | 54,35% | 2,0711 | −0,23 pp | [−1,00; +0,54] | 0,7264 | nie |
| REC | 54,45% | 2,0695 | −0,13 pp | [−0,90; +0,63] | 0,6382 | nie |
| POST | 54,98% | 2,0357 | +0,40 pp | [−0,37; +1,18] | — | — |
| PRE | 54,78% | 2,0417 | +0,20 pp | [−0,58; +0,98] | — | — |

PRE/POST/RANDOM to kryteria reużyte z E008; nie wchodzą do rodziny Holma.
Pozostałe wiersze to nowe kryteria E010. Tylko TAYLOR przekracza próg
+1,0 pp i przechodzi korektę wielokrotności.

## Krzywe naprawy — Tiny ImageNet 224, 30% redukcji

Dev top-1 [%] po T epokach naprawy (T=0 to wynik zero-shot):

| Kryterium | T=0 | T=1 | T=5 | T=10 | T=20 |
|---|---:|---:|---:|---:|---:|
| TAYLOR | 2,51 | 54,36 | 55,61 | 56,58 | 57,20 |
| POST | 6,65 | 52,65 | 54,98 | 56,16 | 57,04 |
| PRE | 9,67 | 52,95 | 54,78 | 56,30 | 56,81 |
| REC | 4,81 | 53,13 | 54,45 | 55,40 | 56,59 |
| KMEANS | 1,14 | 52,29 | 54,64 | 55,54 | 56,49 |
| APOZ | 1,50 | 52,90 | 54,91 | 56,00 | 56,35 |
| L1 | 1,70 | 52,21 | 54,38 | 55,29 | 56,17 |
| BN_GAMMA | 0,96 | 51,75 | 54,35 | 55,47 | 56,16 |
| RANDOM | 0,99 | 51,89 | 54,58 | 55,18 | 56,15 |

TAYLOR jest ponad RANDOM na każdym budżecie: T=1 **+2,47 pp**
(CI [+1,66; +3,29]), T=5 **+1,03 pp** ([+0,26; +1,79]), T=10 **+1,40 pp**
([+0,65; +2,16]), T=20 **+1,05 pp** ([+0,30; +1,80]). To jedyny tak
spójny sygnał w E010, ale wyłącznie na Tiny ImageNet.

## Replikacja — CIFAR-100 96, 10% redukcji, T=5

| Kryterium | dev top-1 | dev NLL | Δ vs RANDOM | 95% CI |
|---|---:|---:|---:|---:|
| TAYLOR | 65,96% | 1,4503 | +0,36 pp | [−0,42; +1,12] |
| RANDOM | 65,60% | 1,4711 | — | — |
| L1 | 65,20% | 1,4755 | −0,40 pp | [−1,20; +0,42] |
| BN_GAMMA | 65,18% | 1,4749 | −0,42 pp | [−1,22; +0,36] |
| REC | 65,16% | 1,4802 | −0,44 pp | [−1,28; +0,40] |
| PRE | 64,70% | 1,4929 | −0,90 pp | [−1,76; −0,04] |
| KMEANS | 64,68% | 1,4774 | −0,92 pp | [−1,76; −0,08] |
| POST | 64,56% | 1,4847 | −1,04 pp | [−1,86; −0,22] |
| APOZ | 64,56% | 1,4839 | −1,04 pp | [−1,92; −0,18] |

**Replikacja nieudana:** TAYLOR +0,36 pp, CI obejmuje zero i nie osiąga
progu +1,0 pp. Na CIFAR-100 TAYLOR ma bardzo mocny wynik zero-shot
(65,02% vs RANDOM 49,48%, +15,54 pp), ale po pierwszej epoce naprawy
przewaga spada do +0,84 pp, a przy T=5 jest już w granicach szumu.
PRE, POST, APOZ i KMEANS wypadają przy T=5 **istotnie poniżej** RANDOM.

## Walidacja — Tiny ImageNet 224, 10% redukcji, T=5

| Kryterium | dev top-1 | Δ vs RANDOM | 95% CI |
|---|---:|---:|---:|
| RANDOM | 56,39% | — | — |
| POST | 56,14% | −0,25 pp | [−0,93; +0,42] |
| REC | 56,10% | −0,29 pp | [−0,95; +0,39] |
| L1 | 56,07% | −0,32 pp | [−1,01; +0,36] |
| KMEANS | 56,05% | −0,34 pp | [−1,02; +0,34] |
| PRE | 56,03% | −0,36 pp | [−1,04; +0,31] |

Przy 10% redukcji wszystkie kryteria są nierozróżnialne od losowania
(±0,4 pp, wszystkie CI obejmują zero). TAYLOR nie był w tej siatce
(`not_in_grid`).

## Interpretacja

- **TAYLOR to kandydat, nie metoda.** Ma spójny sygnał na Tiny 30%
  (zero-shot i T=1–20), ale nie replikuje się na CIFAR-100; jego przewaga
  znika po naprawie. Nie spełnia kryterium „survivor + replikacja".
- **Pozostałe kryteria nie biją losowania.** Na CIFAR-100 część z nich
  wypada istotnie gorzej od RANDOM; na Tiny 10% wszystkie są sobie równe.
- **Potwierdzenie E008/E009:** po naprawie różnice między selekcjami są
  małe. Sygnał wyboru, jeśli istnieje, jest wąski (dataset, ratio, krótki
  budżet) i nie przekłada się na ogólną przewagę metody attention-aware.
- E010 jest screeningiem: jeden seed, jedna architektura, jeden operator.
  Nie uprawnia do twierdzeń publikacyjnych ani do zamknięcia wątku
  atencji poza badanym reżimem.

## Dane i wykonanie

- **Siatki:** Tiny 224 30% — 9 kryteriów × T ∈ {0,1,5,10,20} (primary);
  CIFAR-100 96 10% — 9 kryteriów × T ∈ {0,1,5,10,20} (secondary);
  Tiny 224 10% — 6 kryteriów × T ∈ {0,5} (walidacja).
- **Operator:** wspólny folding E008/E003; α/β z least-squares momentów
  Grama; E010 dopuszcza reprezentanta wchłaniającego wiele kanałów
  (potrzebne dla KMEANS). Dla par rozłącznych wynik identyczny z E008/E003.
- **Kryteria:** PRE, POST, RANDOM (reużyte modele etapu 1), L1, BN_GAMMA,
  TAYLOR, APOZ, REC, KMEANS. Statystyki nowych kryteriów liczone na
  kalibracji A (Tiny 1000, CIFAR 500), bez augmentacji.
- **Naprawa:** AdamW 3e-4, wd 0,01, warmup `min(3,T)`, cosine do 3e-6,
  LS 0,1, clip 1,0, AMP FP16, batch 32/GPU (globalny 128), 4 GPU DDP,
  seed 42, wynik = ostatnia epoka, dev co epokę, bez early stoppingu.
  Tiny: RRC(224) + ColorJitter + hflip; CIFAR: RandomCrop(32, pad 4) +
  hflip + Resize(96), normalizacja 0,5 (recipe AdamW E009, nie SGD E004).
- **Ewaluacja:** dev Tiny 10 000, CIFAR 5 000; sparowany bootstrap 10 000,
  seed 20261018, stratyfikacja po klasach; Holm dla 6 nowych kryteriów
  przy primary.
- **Bramki techniczne:** odtworzenie 9/9 modeli E008/E003 bit w bit;
  dokładne budżety parametrów (2 981 828 / 3 837 364 / 3 721 424);
  modele reużyte i BASE odtworzyły top-1 dokładnie, NLL z |Δ| ≤ 1e-5.
- **Czas:** start 2026-10-18 23:57 UTC, koniec 2026-10-19 20:48 UTC;
  **20 h 51 min** na 4 GPU. Rezerwa i oficjalny test **nietknięte**.

### Przebiegi techniczne wyłączone z analizy

- `20261018T225828Z_338dba1_seed42_y97ml4hj` — wersja 1.0, zatrzymana po
  pierwszej komórce merge (zawieszenie NCCL po zapisaniu wyniku); brak
  danych naukowych.
- `20261018T233427Z_1f7fbc5_seed42_ixkbidqa` — wersja 1.1, `IndexError`
  przy scalaniu `tiny_BASE`; brak danych naukowych.
- Bieg końcowy: `20261018T235741Z_007be2d_seed42_824m539e`, status
  completed, exit 0. Manifest ma techniczne pole `protocol_version: 1.0`;
  `config.json` i protokół są w wersji **1.1**.

## Artefakty i hashe

- `summary.json` — SHA-256 `fbe919022ced28675d87b1bb8ef11b172c6440c7a1c1cbd18c38f6f807bbb43b`;
  `artifacts.sha256` — 764 wpisy.
- `stats/<dataset>.npz|json`, `groups/`, `merge/`, `repair/<dataset>_<ratio>_<kryterium>_T<T>/`,
  predykcje dev per komórka, modele po scaleniu i naprawie.

## Ograniczenia

Jeden seed (42); jedna architektura (EfficientNet-B0+SE) i jeden operator;
ratio 10/30, T ≤ 20; KMEANS to grupowanie z kompensacją LS, nie średnia
klastra z literatury; statystyki APoZ/Taylor z jednego zbioru
kalibracyjnego; CIFAR używa recipe AdamW, więc porównania są w obrębie
E010, nie z historycznym E004; dev wybierał epokę baseline'u, więc nie jest
niezależnym testem; brak wniosków publikacyjnych bez E011.

## Następny krok

Protokół dla wyniku mieszanego przewiduje powtórkę screeningu na drugim
seedzie. Praktyczny wniosek z [PODSUMOWANIE_MIAR.md](PODSUMOWANIE_MIAR.md):
**E011 jako wąskie, 3-seedowe potwierdzenie TAYLORA** z kontrolami
PRE/POST/RANDOM i wiernymi baseline'ami z literatury. Decyzja o kierunku
publikacyjnym (wynik negatywny/analiza vs pełna ścieżka E011) należy do
użytkowniczki.
