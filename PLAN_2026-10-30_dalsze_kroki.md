# Plan po 2026-10-30 — dla nowego czatu

## Stan wyjściowy (fakty, nie plany)

- **E014 (CIFAR-10):** TILT+0.30 vs UNIFORM **+0,87 / +3,77 / +11,27 pp**
  przy 30/50/70% parametrów; 12/12 par seed×budżet dodatnich; 4 seedy.
- **E015 (CIFAR-100):** **+0,81 / +1,96 / +1,60 pp**; 11/12 dodatnich,
  ale zamrożone kryterium (+2,0 pp) **niespełnione**.
- **E019 (Tiny ImageNet 64×64):** **+3,71 / +10,32 / +7,60 pp**;
  12/12 dodatnich; **kryterium spełnione**. Baseline dev 50,82%;
  po foldingu TILT 38,12/30,08/16,14%, UNIFORM 34,41/19,77/8,54%.
- **E016 AUTO50:** negatywny (przegrywa z TILT). **E017/E018 miary
  grupowania:** negatywne (cosine nie replikuje się).
- Brak aktywnych zadań. Cztery GPU wolne. Dysk: **5,7 GB wolne (91%)**.
  Środowisko: Conda `attention`.

## Cel najbliższego etapu

Zamienić wynik alokacji w spójny, weryfikowalny wkład: **mechanizm +
użyteczny reżim + wiarygodny baseline**. Nie szukać kolejnych miar.

## Aktualizacja 2026-10-31

- **Audyt luki:** [literature/AUDYT_ALOKACJA_FOLDING.md](../literature/AUDYT_ALOKACJA_FOLDING.md)
  — Wang i in. (2025) wprost piszą, że nie optymalizują sparsity per layer
  i zostawiają to na przyszłość; *Cut Less, Fold More* używa jednolitych ratio.
  Luka jest realna, ale nie twierdzimy, że pomysł alokacji jest nowy w kompresji.
- **E021 (L1/L2/cosine/korelacja/Ward):** 72/72, **wynik negatywny**.
  Żadna metryka nie bije referencji; L1 gorszy. Wątek miar zamknięty.
- Priorytet pozostaje: mechanizm → łagodne budżety → oficjalny baseline →
  druga architektura.

## Krok 1 — mechanizm (bez treningu, priorytet) — następny eksperyment **E022**

Wejście: checkpointy E011/E015/E019 i istniejące summary foldingu.
Pytania:
1. Jak szkoda zależy od głębokości grupy (pre i post BN)?
2. Ile BN odzyskuje i które grupy pozostają krytyczne?
3. Dlaczego CIFAR-100 wypada słabiej niż CIFAR-10 i Tiny ImageNet?
4. Czy TILT rzeczywiście chroni grupy o wysokiej zmierzonej szkodzie?
5. Czy błąd rekonstrukcji wag przewiduje rzeczywistą szkodę *przy stałym
   poziomie kompresji*?

Wyjście: nowy folder `experiments/<data>_E022_mechanizm_alokacji/` z OPIS,
skryptami CPU/GPU (ocena istniejących modeli), WYNIKI. Bez nowego treningu.
Koszt: 1–2 dni, 1 GPU. Kryterium: ilościowy związek + ablacja, bez
twierdzenia o uniwersalności.

## Krok 2 — łagodniejsze budżety (tylko folding)

Dodać budżety **10% i 20% parametrów** (obok 30/50/70) na trzech zbiorach
× 4 seedy × UNIFORM/TILT+0.30 (opcjonalnie +0.15). To pytanie, czy metoda
jest użyteczna, gdy kompresja nie niszczy sieci. 48–72 zadania, ok. 20–30 min.
Protokół i kryterium zamrozić **przed** uruchomieniem; wspólny target =
faktyczny UNIFORM; ten sam operator i BN.

## Krok 3 — oficjalny baseline 1:1

Pobrać `github.com/osaukh/folding_as_projection` i uruchomić oficjalny
folding ResNet18 na naszych checkpointach przy dopasowanych budżetach.
Porównać z naszą referencją E012. Cel: upewnić się, że nie porównujemy się
do słabszej adaptacji. Udokumentować różnice (BN repair, fc bias, puste
klastry). Koszt: 1–2 dni.

## Krok 4 — druga architektura (dopiero po 1–3)

Np. PreActResNet18 (obsługiwany przez oficjalny kod) albo ResNet18 przy
96×96 z mocniejszym przepisem. Trenować 4 seedy, potem TILT. Koszt 2–5 h;
najpierw sprawdzić dysk (5,7 GB).

## Krok 5 — decyzja publikacyjna

Minimum: 2 architektury, 3 zbiory, mechanizm, oficjalny baseline, uczciwe
pokazanie słabego CIFAR-100 i załamania jakości przy 50–70%. Dopiero wtedy
decyzja. Jeśli tego nie ma: warsztat/benchmark negatywny albo zmiana kierunku.

## Czego nie robić

- Nie wracać do miar grupowania bez nowego mechanizmu.
- Nie stroić AUTO50 ani progów po wynikach.
- Nie usuwać historycznych runów i nie nadpisywać wyników.
- Nie ogłaszać publikowalności na podstawie samych tabel.
- Nie trenować bez sprawdzenia 4 GPU i miejsca na dysku.

## Gdzie wszystko jest

- `STATUS_BADAN.md` — stan i skróty.
- `NOTATKA_KONTYNUACJA.md` — szybki start nowego czatu.
- E014: `experiments/2026-10-27_E014_replikacja_alokacji/WYNIKI.md`.
- E015: `experiments/2026-10-27_E015_cifar100_alokacja/WYNIKI.md`.
- E019: `experiments/2026-10-29_E019_tinyimagenet_alokacja/WYNIKI.md`.
- E016–E018: wyniki negatywne w ich folderach.
