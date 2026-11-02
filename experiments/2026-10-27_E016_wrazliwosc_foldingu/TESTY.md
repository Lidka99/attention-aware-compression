# E016 — wykonane testy i kontrole

## 1. CPU preflight (bez GPU)

Komenda:

```bash
CUDA_VISIBLE_DEVICES='' /home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-27_E016_wrazliwosc_foldingu/check_cpu_e016.py
```

Wynik: **`E016_CPU_PREFLIGHT_OK`**.

Sprawdzone:

- środowisko Conda `attention` i wersje bibliotek,
- hashe 5 importowanych źródeł (E011/E012/E014/E015) zgodne z configiem,
- brak importów `attention_v2`/`attention_v3`,
- 98 itemów S1, 8 smoke, 4 rozłączne shardy na dataset,
- poprawka launchera E015 (`import os`, `LAUNCHER_REVISION`) obecna,
- hashe checkpointów i splitów oraz liczba parametrów (11 173 962 /
  11 220 132),
- train/dev: 45 000/5 000 i 40 000/5 000, brak nachodzenia,
- kalibracja 2 048 próbek: deterministyczna, posortowana, bez duplikatów,
  sha256 `6e9656c4…` (CIFAR-10) i `ed8a3a3b…` (CIFAR-100),
- kontrola ratio 0: predyktory zerowe, folding tożsamościowy poza
  `num_batches_tracked`,
- rzeczywisty folding `layer1.0.relu1` 25%: 64→48 kanałów,
  params 11 173 962→11 155 498 (CIFAR-10) i 11 220 132→11 201 668
  (CIFAR-100), recon 0,2357 / 0,2953.

## 2. Smoke na 4 GPU

Komenda:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-27_E016_wrazliwosc_foldingu/launch_sensitivity.py --smoke
```

Run: `runs/E016/smoke_20261027T213755Z_cb9ffaf_a61fab01`.
Wynik: **8/8 completed**, 4 procesy-workerów na falę, GPU0–3, exit 0.
Smoke ma pełny BN i pełny dev, ale nie ma znaczenia naukowego.

## 3. Pełny S1 na 4 GPU

Komenda (launcher):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-27_E016_wrazliwosc_foldingu/launch_sensitivity.py
```

Run: `runs/E016/s1_20261027T215948Z_cb9ffaf_d98f0ed9`.
Wynik: **98/98 completed**, 8 procesów-workerów (4 na falę), GPU0–3,
exit 0. CIFAR-10 49/49, CIFAR-100 49/49, kontrole 2/2.

Kontrole po biegu: dev count 5 000 w pre/post-BN każego itemu, baseline
top1/NLL zgodne z configiem, `official_test_used=false`,
`reserve_used=false`, kontrola ratio 0 bez szkody pre-BN, agreement 1,0.

## 4. Analiza CPU

Komenda:

```bash
/home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-27_E016_wrazliwosc_foldingu/analyze_e016.py \
  --run-dir runs/E016/s1_20261027T215948Z_cb9ffaf_d98f0ed9 \
  --output-dir runs/E016/analizy/20261027T230435Z_s1
```

Wynik: `analysis.json`, `analysis.md`, `group_level.csv`;
**`E016_S2_RECOMMENDED=YES`** (C1–C5 spełnione).

## 5. Znane zdarzenia infrastrukturalne

- Pierwszy pełny launcher został przerwany przez limit czasu powłoki po
  20 itemach CIFAR-10; bieg zachowany jako nieukończony
  (`s1_20261027T213945Z_cb9ffaf_d066bae8`, `ABORTED.md`).
- Drugi launcher ukończył CIFAR-10 49/49, ale `verify_summaries` oczekiwał
  2 kontroli na dataset zamiast 1. Poprawka: `expected_controls=1`.
  CIFAR-10 zachowano; CIFAR-100 uruchomiono w tym samym biegu przez
  udokumentowany resume. Finalny run jest kompletny.
- Poprawka launchera E015 (`import os`) jest obecna; E016 jej nie powiela.
