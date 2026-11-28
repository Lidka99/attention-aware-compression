# E049 — figura b10: wspólne bramki na CIFAR-100 i Tiny ImageNet

[PDF wektorowy](e049_b10_joint_caps_en.pdf) · [PNG](e049_b10_joint_caps_en.png) ·
[Generator](make_figure.py) · [Wyniki runu](../../../../runs/E049/eval_20261124T234559Z_b4ca081b/WYNIKI.md)

Figura do manuskryptu (angielskie etykiety, szerokość druku 170 mm),
zbudowana **wyłącznie** z `runs/E049/eval_20261124T234559Z_b4ca081b/`
(`per_seed.csv` i `audit.json`, `split=benchmark`). Bez nowych punktacji,
bez GPU, bez treningu i bez zmian w `paper/`. Cztery seedy 42–45.

## Nazewnictwo

- **Frozen Width-Offset Profile** — pełna nazwa profilu; na wykresie
  występuje wyłącznie skrót **Frozen offsets**.
- Krótka etykieta **Frozen offsets** odpowiada identyfikatorowi **L06**
  w plikach runu: wierszom `b10_L06`, kluczom `loss_to_full_pp/b10_L06`
  i `contrasts/b10_L06-b10_*`. Identyfikator i wartości danych pozostają
  bez zmian — usunięto tylko widoczny napis z paneli.
- **UNIFORM** = `b10_U`, **T-proj** = `b10_Tproj`.

## Układ

- **A — Paired top-1 gains at b10.** Cztery wiersze: Frozen offsets
  względem UNIFORM i względem T-proj, każda para CIFAR-100 (góra)
  i Tiny ImageNet (dół). Puste kółka to seedy, romb to średnia z trzema
  miejscami (+0,615 / +0,675 / +0,670 / +1,800). Pionowa kreska przerywana
  w zerze. **Próg +1 pp zaznaczono wyłącznie przy dwóch wierszach
  Frozen offsets−UNIFORM** (krótkie znaczniki „I” tylko w tych wierszach);
  nie stosuje się do wierszy Frozen offsets−T-proj.
- **B — Loss to the full model.** Po jednym wierszu na zbiór: strata
  top-1 Frozen offsets do pełnego modelu po CMA, per seed i średnia
  (liczba obok rombu). Osie od zera; przerywana linia to próg ≤3 pp.

Seedy mają deterministyczne przesunięcie pionowe: w A ±0,09 jednostki,
w B dwa tory ±0,34 (sąsiednie wartości trafiają na przeciwne tory), aby
bliskie kółka i znacznik średniej nie zasłaniały się wzajemnie; nie jest
to niepewność ani CI. Żadnych przedziałów ufności nie rysujemy —
`audit.json` ich nie zawiera, a E049 nie zakłada testu istotności.

## Wyniki na figurze (benchmark, średnia 4 seedów)

| Wiersz | CIFAR-100 | Tiny ImageNet |
|---|---:|---:|
| Frozen offsets−UNIFORM | **+0,6150 pp** (4/4) | **+0,6750 pp** (4/4) |
| Frozen offsets−T-proj | **+0,6700 pp** (4/4) | **+1,8000 pp** (4/4) |
| strata Frozen offsets do pełnego | **1,6575 pp** | **4,5700 pp** |

Na Tiny strata Frozen offsets (4,57 pp) przekracza próg 3 pp, więc
wspólna bramka b10 nie jest spełniona (`joint_gate_met: false`); panel B
pokazuje ten fakt bez ukrywania go. Wartości ujemne nie występują, ale
oś A zawiera zero, żeby odległość od zera była widoczna.

## Weryfikacja

`make_figure.py` przelicza z `per_seed.csv` top-1, różnice sparowane
i stratę `full − b10_L06` dla każdego seeda, a potem wymaga zgodności
z `audit.json` (tolerancja 1e-9): średnie top-1, `mean_pp`, `per_seed_pp`,
`sd_pp`, `positive_pairs = 4`, `loss_to_full_pp` dla L06 oraz
`full == 0`. Sprawdza też `n = 10000`, komplet wariantów i to, że L06
oraz T-proj mieszczą się w 99,9–100% kosztu U w P i MAC. Dodatkowo
skrypt zbiera wszystkie widoczne napisy figury i przerywa pracę, gdy
którykolwiek zawiera `L06`; identyfikator występuje tylko w kluczach
danych i w tym pliku. Uruchomienie przerywa się też przy każdej innej
niezgodności. Zapis jest deterministyczny (identyczne bajty przy
powtórzeniu); PDF ma 170,0 × 88,9 mm, osadzone fonty DejaVu Sans
(tekst wektorowy), PNG 400 dpi.

Wejścia (SHA-256):

- `per_seed.csv`: `c58a1fe5c477cdaeefdfef24a5375b2fa020fc47f254971ec172772075cf5411`
- `audit.json`: `2361839e91cb1d56cc7f7a5e5817575a1e81f2f48a5ff749cb2bdb4df1705810`

## Odtworzenie

Z katalogu głównego repozytorium, Conda `attention`, Matplotlib 3.10.9,
bez GPU:

```bash
CUDA_VISIBLE_DEVICES='' /home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-11-24_E049_folding_mocniejszych_baz/wykresy/20261124T234559Z_b4ca081b/make_figure.py
```

SHA-256 wyników:

- `e049_b10_joint_caps_en.png`: `3936078c2847341a8dc2eeb7d66ec94045d678a23dc9ad0c67b6c9ebb5bc4c51`
- `e049_b10_joint_caps_en.pdf`: `e7e64ef596274b74ccaae75e20598d485449f0c7108ddaca73ccda32821c5a17`
