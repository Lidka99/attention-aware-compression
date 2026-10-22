# E009 — wynik naprawy 120 epok po kompresji 30% na baseline 224

Data biegu: 2026-10-14 UTC. Dokument uzupełniony: 2026-10-21 UTC.
**Status wykonania: completed (exit 0; 2026-10-14 15:03 UTC).**
Protokół przed uruchomieniem: [PROTOKOL_v1.md](PROTOKOL_v1.md), v1.0 (frozen).
Bieg: **`runs/E009/20261013T172339Z_954ece0_seed42_nh02g2l8/`**,
commit **`954ece0`**, dirty=false.

Bieg `runs/E009/20261012T155859Z_fef2836_seed42_jsagnut2` został przerwany
awarią maszyny 2026-10-13 ok. 09:18 UTC w trakcie RANDOM (11/120 epok).
Jego liczby **nie wchodzą do analizy**; poniżej raportowany jest wyłącznie
pełny bieg powtórzony od zera.

## Wynik główny — 120 epok równej naprawy (30% redukcji)

| Model | dev top-1 | dev NLL | parametry |
|---|---:|---:|---:|
| BASE (dostrajanie baseline'u) | **60,04%** | 1,9978 | 4 263 748 |
| POST (kryterium po SE) | **59,46%** | 1,9970 | 2 981 828 |
| PRE (kryterium przed SE) | **59,18%** | 1,9857 | 2 981 828 |
| RANDOM | **59,21%** | 1,9964 | 2 981 828 |

Porównania sparowane po obrazach (bootstrap 10 000, seed 20261017,
stratyfikacja po klasach; `summary.json`):

- **POST−PRE = +0,28 pp, CI [−0,41; +0,96]** → kryterium pierwszorzędne
  (dolna granica > 0) **nieprzekroczone**.
- POST−RANDOM = +0,25 pp, CI [−0,48; +1,02] → brak różnicy.
- PRE−RANDOM = −0,03 pp, CI [−0,76; +0,72] → brak różnicy.
- BASE−POST = +0,58 pp, CI [−0,09; +1,26] → różnica nieistotna.
- BASE−PRE = +0,86 pp, CI [+0,15; +1,57]; BASE−RANDOM = +0,83 pp,
  CI [+0,14; +1,54] → PRE i RANDOM pozostają istotnie poniżej BASE.

## Porównanie budżetów naprawy: 60 epok (E008) vs 120 epok (E009)

| Model | E008, 60 epok | E009, 120 epok | Zmiana |
|---|---:|---:|---:|
| BASE | 59,45% | 60,04% | +0,59 pp |
| POST | 58,32% | 59,46% | +1,14 pp |
| PRE | 57,81% | 59,18% | +1,37 pp |
| RANDOM | 57,55% | 59,21% | +1,66 pp |

| Porównanie | E008 (60 epok) | E009 (120 epok) |
|---|---:|---:|
| POST−PRE | +0,51 pp [−0,18; +1,19] | +0,28 pp [−0,41; +0,96] |
| POST−RANDOM | +0,77 pp [+0,03; +1,50] | +0,25 pp [−0,48; +1,02] |

## Interpretacja

- Dłuższa naprawa podnosi **wszystkie** warianty o 0,6–1,7 pp, ale najbardziej
  PRE i RANDOM. Pozostała przewaga POST nad PRE i RANDOM znika.
- Po 120 epokach POST jest tylko 0,58 pp poniżej BASE i przedział obejmuje
  zero; PRE i RANDOM pozostają istotnie niżej. To poprawa względem E008,
  ale nie jest dowodem działania kryterium atencyjnego — to jeden seed
  i jeden checkpoint.
- Łącznie E008 + E009: przy 30% redukcji różnice między selekcjami są małe
  i zanikają wraz z wydłużaniem naprawy. Nie ma podstaw do twierdzenia,
  że atencja kanałowa daje trwałą przewagę w tym reżimie.
- Rezerwa i oficjalny test pozostają **nietknięte**.

## Dane i wykonanie

- Wejścia: modele etapu 1 E008 (BASE/PRE/POST/RANDOM po 30% redukcji),
  baseline E007 (`718bbb26…`, dev 58,49%), Tiny ImageNet 224; dev 10 000
  (50/klasę); 30% redukcji = 3168 usuniętych kanałów, 2 981 828 parametrów.
- Receptura: AdamW lr 3e-4, wd 0,01, warmup 3 epoki, cosine do 3e-6,
  LS 0,1, clip 1,0; augmentacje jak w E008; batch 32/GPU (globalny 128),
  AMP FP16, 120 epok, wynik = ostatnia epoka, dev co epokę, bez early
  stoppingu i wyboru epoki po fakcie.
- Cztery warianty sekwencyjnie na 4 GPU (DDP) — nie są to cztery niezależne
  powtórzenia. Start 2026-10-13 17:23 UTC, koniec 2026-10-14 15:03 UTC;
  **21 h 40 min**.
- Kontrole wejść: [TESTY_CPU_v1.json](TESTY_CPU_v1.json) — passed; hashe
  modeli i danych zgodne z configiem; liczba parametrów zgodna.

## Artefakty i hashe

- `summary.json` — SHA-256 `c37cc97b5e90368ea58d68e7a5e661e204475fb44bf093c27d44d8325722c7b9`;
  `artifacts.sha256` — 59 wpisów.
- `dev_final_{BASE,PRE,POST,RANDOM}.npz` oraz `dev_final_*_rank{0..3}.npz`;
  `last_*.pt`; `metrics.jsonl`; `command.sh`; `config.resolved.json`.

## Ograniczenia

Jeden seed i jeden bieg; jeden baseline i jedna architektura; wyłącznie
30% redukcji; dev nie jest niezależnym testem (wybierał epokę w E007);
wynik dotyczy 120 epok i tego operatora scalania. Brak wniosków
publikacyjnych bez powtórzeń międzyseedowych.

## Następny krok

Po E009 uruchomiono E010 — przesiew dziewięciu kryteriów przy krótkiej
naprawie na Tiny ImageNet i CIFAR-100. Wynik:
[E010/WYNIKI.md](../2026-10-18_E010_benchmark_kryteriow/WYNIKI.md)
(`survivor_not_replicated`). Dalszy krok — decyzja o E011 — należy do
użytkowniczki.
