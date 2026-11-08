# E036 — koszt alokacji TILT przy dopasowanych parametrach

Status: **completed, audyt opisowy**, 2026-11-08. Bez treningu, danych
rzeczywistych i nowej oceny jakości. **TILT i UNIFORM mają niemal identyczną
liczbę parametrów, ale TILT nie jest dopasowany pod liczbę operacji.**

## Wykonanie i integralność

- Protokół [OPIS.md](OPIS.md) zamrożony w `71d90f3` przed pomiarami;
  poprawka księgowania pamięci `32c0920` przed finalnym runem.
- Run `runs/E036/20261107T231027Z_27bac186`: **completed**,4×exit0,
  **789,50s (13,16min)**. Cztery GPU, seed42–45→GPU0–3, jeden proces/seed.
- Pomiary:5par×7modeli (teacher+6profili)×2batche×3serie×200forwardów.
  Po140 modeli i210 pomiarów na seed; **840 plików czasów**.
- CPU check35 modeli: ręczna geometria i hooki dały **identyczne MAC**
  dla każdej warstwy conv/linear; 13 grup/plany i stany PRE zgodne z E035.
- `analysis/verification.json` **PASS**:420 kontroli,140 zgodności hashy
  stanu, pełna siatka, czasy dodatnie/skończone, brak dostępu do `data/`,
  brak nowych metryk jakości. `analysis/analysis.json` i CSV w tym katalogu.
- Conda attention, Python3.11.15, Torch2.12.0+cu130, CUDA13.0,
  4×RTX2080Ti. FP32, TF32 off, cudnn benchmark off/deterministic on,
  inference_mode, BLAS/OMP2. Wejścia syntetyczne `randn`, hash zapisany.
- Jakość w tabelach jest **wyłącznie reuse E035**; E036 nie otwierał
  testów/val i nie kalibrował BN. Profili nie zmieniano po pomiarach.

## Główny wynik

Przy tym samym budżecie parametrów TILT ma **więcej MAC** i niższą
przepustowość batch128. Pełne wartości: [TABLES.md](TABLES.md).

| Para | Red. | MAC T/U | Przepust. b128 T/U | Latencja b1 T/U | Peak b128 U/T [MB] | Jakość T−U [pp] |
|---|---:|---:|---:|---:|---:|---:|
| R/C10 | 30% | 1,105 | 0,948 | 1,018 | 195,9/205,8 | +0,81 |
| R/C10 | 50% | 1,364 | 0,782 | 0,997 | 168,2/197,9 | +3,41 |
| R/C10 | 70% | 1,789 | 0,885 | 1,046 | 207,4/188,8 | +10,99 |
| R/C100 | 30% | 1,105 | 0,948 | 1,027 | 196,0/206,0 | +0,83 |
| R/C100 | 50% | 1,366 | 0,780 | 1,012 | 167,4/198,0 | +1,30 |
| R/C100 | 70% | 1,823 | 0,875 | 0,981 | 204,4/188,8 | +2,06 |
| R/Tiny | 30% | 1,109 | 0,946 | 1,018 | 174,5/184,5 | +3,54 |
| R/Tiny | 50% | 1,357 | 0,702 | 1,009 | 104,5/176,0 | +9,36 |
| R/Tiny | 70% | 1,806 | 0,742 | 1,038 | 92,0/167,4 | +7,48 |
| P/C10 | 30% | 1,105 | 0,947 | 0,965 | 195,8/205,8 | +0,96 |
| P/C10 | 50% | 1,358 | 0,789 | 0,985 | 168,2/197,9 | +4,95 |
| P/C10 | 70% | 1,748 | 0,883 | 1,015 | 207,4/187,3 | +13,58 |
| P/C100 | 30% | 1,105 | 0,947 | 0,994 | 196,0/206,0 | +4,49 |
| P/C100 | 50% | 1,361 | 0,787 | 1,006 | 167,3/198,0 | +7,64 |
| P/C100 | 70% | 1,790 | 0,885 | 1,018 | 204,4/187,3 | +22,74 |

Zakresy: **MAC T/U 1,105–1,823** (+10,5% do+82,3%); przepustowość b128
**0,702–0,948** (−5,2% do−29,8%); latencja b1 **0,965–1,046**, czyli
w granicach narzutu uruchomienia. Parametry T/U **0,9994–1,0000**.

## Interpretacja

TILT+0.30 nie zmniejsza szerokości wczesnych grup tak mocno jak UNIFORM.
Wczesne warstwy mają wysoką rozdzielczość przestrzenną, więc dodatkowe
kanały kosztują dużo MAC, mimo że liczba parametrów pozostaje dopasowana.
To wyjaśnia rosnący koszt względny przy50/70%: alokacja przenosi budżet
parametrów w region o większym koszcie obliczeń na kanał.

Pamięć nie daje prostej kompensacji. Na CIFAR przy30/50% TILT ma wyższy
peak batch128, przy70% niższy; na Tiny przy50/70% peak TILT jest wyraźnie
wyższy (**176,0 vs104,5MB** i **167,4 vs92,0MB**). To wynik kształtów
aktywacji, nie liczby parametrów; nie interpretować jako uniwersalnej reguły.

**Wniosek dla artykułu:** przewaga jakości TILT z E034/E035 jest realna
przy dopasowanym budżecie **parametrów**, ale nie jest darmowa: wymaga
10–82% więcej MAC i obniża przepustowość b128 o5–30%. Nie wolno pisać
„matched compute” ani sugerować, że TILT jest równie szybki jak UNIFORM.
To ograniczenie zakresu twierdzenia, nie unieważnienie wyników jakości.

## Ograniczenia

- Jeden serwer i cztery przypisania seed/GPU; nie jest to niezależna
  replikacja sprzętowa ani podstawa wnioskowania o populacji GPU.
- Steady-state device-forward bez loadera, transferów host–device i aplikacji;
  batch1 jest zdominowany przez narzut. Zegary nieblokowane, możliwy dryf.
- Syntetyczne wejście `randn` jest wystarczające dla kosztu sprzętowego,
  ale nie reprezentuje rozkładu danych; nie mierzono wpływu rzadkości/sparsity.
- Dwie pokrewne architektury, istniejące checkpointy, jeden przepis treningu.
  Brak pomiaru zużycia energii i wydajności na innych GPU/CPU.
- Macierz alokacji jest zamrożona; E036 nie testuje, czy inny profil
  poprawiłby jakość na jednostkę MAC. To audyt kosztu, nie optymalizacja.

## Błąd wykonania

Pierwszy run `runs/E036/20261107T225229Z_1f5b7f3a` failed: dwa workery
zatrzymały się na `memory_allocated()==0` po pierwszym modelu. cuDNN
zostawia trwały blok alokatora ok.8,5MB; po `model.cpu()` nie było tensorów
CUDA. Poprawiono wyłącznie księgowanie: baseline po warmupie, tolerancja32MB
na trwały workspace. Nie zmieniono modeli, profili, czasów, liczby próbek,
kolejności ani kryteriów. Run zachowany bez wyników do agregacji.

## Odtworzenie

```bash
export CONDA_DEFAULT_ENV=attention
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/users/s224574/miniconda3/envs/attention/bin/python
E=experiments/2026-11-07_E036_audyt_kosztu
R=runs/E036/20261107T231027Z_27bac186
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/benchmark.py" freeze
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/benchmark.py" check
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$E/benchmark.py" run
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/analyze.py" "$R"
CUDA_VISIBLE_DEVICES='' "$PY" -B "$E/audit.py" "$R"
```

Pełny run tworzy nowy katalog; analyze/audit należy skierować do wybranego runu.
