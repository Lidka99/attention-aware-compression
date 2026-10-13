# E008 — wynik kompresji 10%/30% na baseline 224 i równego dostrajania (60 epok)

Data: 2026-10-13 UTC. **Status wykonania: completed (exit 0; 2026-10-12 15:57 UTC).**
Protokół przed uruchomieniem: [PROTOKOL_v1.md](PROTOKOL_v1.md), v1.1.
Bieg: **`runs/E008/20261012T042932Z_0fd9962_seed42_tn1zlyim/`**,
commit **`0fd9962`**, dirty=false.

## Wynik główny — etap 2: równe dostrajanie (60 epok, 30% redukcji)

| Model | dev top-1 | dev NLL | parametry |
|---|---:|---:|---:|
| BASE (dostrajanie baseline'u) | **59,45%** | 1,9930 | 4 263 748 |
| POST (kryterium po SE) | **58,32%** | 1,9957 | 2 981 828 |
| PRE (kryterium przed SE) | **57,81%** | 1,9889 | 2 981 828 |
| RANDOM | **57,55%** | 2,0040 | 2 981 828 |

Porównania sparowane po obrazach (bootstrap 10 000, stratyfikacja po klasach,
`summary.json`):

- **POST−PRE = +0,51 pp, CI [−0,18; +1,19]** → kryterium pierwszorzędne
  (dolna granica > 0) **nieprzekroczone**.
- POST−RANDOM = +0,77 pp, CI **[+0,03; +1,50]** → jedyna istotna przewaga
  nad losowaniem, **na granicy istotności**.
- PRE−RANDOM = +0,26 pp, CI [−0,49; +0,99] → brak różnicy.
- BASE−POST = +1,13 pp [0,45; 1,82]; BASE−PRE = +1,64 pp [0,95; 2,33];
  BASE−RANDOM = +1,90 pp [1,20; 2,59] — kompresja 30% kosztuje istotnie
  także po naprawie.

## Etap 1 — szkoda bez naprawy

| Model | 10% redukcji | 30% redukcji |
|---|---:|---:|
| PRE | 47,42% | 9,67% |
| POST | 47,07% | 6,65% |
| RANDOM | 40,79% | 0,99% |
| BASE | 58,49% (bramka odtworzona: 5849/10000) | — |

Surowo: selekcja (PRE/POST) bije losową o **+5,7 do +8,7 pp**, ale **POST < PRE**
przy obu budżetach — odwrotnie niż na CIFAR-100 (E003). Przy 30% bez naprawy
model praktycznie się rozpada (1–10% top-1).

## Interpretacja

- Po równym dostrojeniu na mocniejszym baseline różnice między selekcjami są
  **małe i w większości nieistotne**; „attention-aware" (POST) nie wygrywa
  z PRE w teście głównym.
- POST utrzymuje najwyższą pozycję wśród metod wyboru i jako jedyny jest
  (o włos) istotnie powyżej RANDOM — sygnał wyboru istnieje, ale jest słaby.
- Kompresja 30% nadal kosztuje istotnie (~1,1–1,9 pp) po 60 epokach naprawy.
- **Kontynuacja: E009** — naprawa **120 epok** (te same modele startowe,
  jeden ciągły cosine), uruchomiony automatycznie 2026-10-12 15:59 UTC;
  stan 13.10 07:10: BASE **60,04%**, PRE **59,18%** (po 120 epokach),
  POST w trakcie (~59% przy ep. 86/120), RANDOM w kolejce; koniec ~13.10 ~14:00.
  E009 rozstrzygnie, czy dłuższy budżet zmienia obraz różnic.

## Dane i wykonanie

- Baseline E007 (`718bbb26…`, dev 58,49%); plany strukturalne 10%/30%
  (1048 / 3168 usuniętych kanałów); kalibracja A=1000 (5/klasę) do par i α/β,
  B=1000 do diagnostyki, seed 20261015; dev 10 000 (50/klasę); rezerwa
  i test **nietknięte**.
- Etap 1: 4 rangi, momenty FP64, rank0 wybrał grupy (przed oceną), dev
  (batch 64, FP32, deterministycznie), diagnostyka B. Etap 2: warianty
  sekwencyjnie na 4 GPU: 60 epok AdamW lr 3e-4 (warmup 3, cosine → 3e-6),
  LS 0,1, clip 1,0, augmentacje jak w treningu bazowym, batch 32/GPU.
- Czas: etap 1 ~10 min; etap 2 ~11,5 h. `artifacts.sha256`: 107 wpisów.

## Artefakty i hashe

- `summary.json` (`bd3faa05df8e51c1…`), `stage1/stage1_metrics.json`,
  `stage2/stage2_metrics.json`, `dev_final_{BASE,PRE,POST,RANDOM}.npz`,
  modele `models/*.pt` (etap 1) i `stage2/last_*.pt`.

## Ograniczenia

Jeden seed; dev nie jest niezależnym testem (wybierał epokę w E007);
60 epok naprawy mogło nie w pełni zbiec — adresuje to E009; wyniki obowiązują
dla tego baseline i tych budżetów.

## Następny krok

E009 (120 epok) → porównanie budżetów 60 vs 120 epok i decyzja o kierunku
publikacyjnym (zamknięcie wątku selekcji vs pełna ścieżka na baseline 224).
