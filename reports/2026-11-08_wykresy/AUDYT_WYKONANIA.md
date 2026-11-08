# Audyt wykonania E034–E037 (2026-11-08)

CPU-only, bez otwierania danych/testów i bez uruchamiania modeli. Sprawdzono zapisy
w `runs/`. Skrypt: `reports/2026-11-08_wykresy/audit_zapisow.py`; dowody:
`reports/2026-11-08_wykresy/audit_evidence.json`. Wszystkie czytane artefakty
zahashowano (checkpointy strumieniowo), a wejścia sprawdzono przed/po audycie.

## Werdykt

Lokalne artefakty są spójne i silnie potwierdzają, że E034–E037 wykonano do końca:
manifesty `completed`, 4×exit0, komplet plików, checkpointy zgodne z summary/logami,
a niezależne przeliczenia zgadzają się z analizami. To **nie** jest jednak zewnętrzny,
podpisany dowód fizycznej pracy GPU; hash pokazuje spójność, nie czas.

## E034

- `runs/E034/train_20261105T014209Z_21b44f52`: manifest `completed`, 4×exit0, wall **8503,09 s**
  (2026-11-05T01:42:09→2026-11-05T04:03:52). 8 summary (4 seedy×2 zbiory),
  **1600** epok w 4 logach po 400 linii, 8×200 linii `metrics.jsonl`.
  8 checkpointów `last.pt`: SHA256 strumieniowo zgodny z `summary.json`, końcową
  linią `metrics.jsonl` i końcową linią logu.
- `runs/E034/fold_20261106T003625Z_9ec4f2ea`: manifest `completed`, 4×exit0, wall **179,07 s**.
  **24** summary, **144** metryki (2 profile×3 tryby), **8** plików controls
  (**24** metryki kontrolne), 4 logi×6 zadań = **24**.
- Baseline PRE i **6** kontrastów CMA (T−U) przeliczone z 24 summary: zgodne z
  `analysis.json`/`verification.json`. CMA: CIFAR-10 b70 **+13,190 pp**,
  CIFAR-100 b70 **+23,095 pp**; wszystkie 6/6 dodatnie.
  Baseline CIFAR-10 **95,215%**, CIFAR-100 **76,905%**.

## E035

- `runs/E035/20261107T183831Z_8b0b14e0`: manifest `completed`; dev i test po 4×exit0 i 4 logi po 15 zadań.
  Dev i test: po **60** summary / **360** metryk / **20** controls (**60** metryk).
  Przeliczenia z zapisów, bez ponownej oceny modeli. `dev_gate.json` passed: 360 ocen, 60 kontroli,
  max|Δ| top1 = 0, nll = 1,19e-7.
- Czasy: dev **490,39 s**, test
  **515,72 s**, całość
  **1018,04 s** (2026-11-07T18:38:31→2026-11-07T18:55:30).
- **15 dev i15 test** kontrastów CMA (T−U) przeliczono z surowych summary;
  mean/SD/liczba dodatnich zgodne z `analysis.json`. Modele i kalibracje
  mają zgodne hashe stanu w dev/test; nie uruchamiano ponownej oceny jakości.

## E036

- `runs/E036/20261107T231027Z_27bac186`: manifest `completed`, 4×exit0, **789,50 s**
  (2026-11-07T23:10:27→2026-11-07T23:23:37). **20** plików `costs.json`
  = **140** stanów; **840** plików czasów ×200 = **168 000** zdarzeń; 4 statusy po 210
  zadań; 4 logi po 240 linii; monitor kończy `[0,0,0,0]`.
- **140** `state_sha256` zgodnych z `frozen.json`; liczebności analizy:
  280 rows / 120 ratios / 120 averages / 280 memory / 140 costs.
  Rows czasów i ilorazy batch1 przeliczono z surowych plików.
- SD ilorazu latencji T/U dla batch1 z surowych 60 ratios: **0,0192–0,1002**
  (min ResNet/Tiny b70, max ResNet/CIFAR-10 b50).

## E037

- `runs/E037/20261108T031759Z_855c0b33`: manifest `completed`, 4×exit0, **292,28 s**
  (2026-11-08T03:17:59→2026-11-08T03:22:52). **60** summary (**180** metryk),
  **20** controls (**60** metryk), 4 statusy po 15 zadań, 4 logi po 30 linii.
- **101/101** hashów w `frozen.json` zgodnych z plikami (w tym 20 checkpointów
  E011/E015/E019/E034); plany U′ mają MAC(U′)≤MAC(T).
- **135** kontrastów, **180** rows i **135** averages przeliczone niezależnie
  z E035 DEV + E037 summary: zgodne z `analysis.json` (max |Δ| rows
  0.0e+00, allocation 1.8e-15, averages 5.6e-17).
  20 bloków controls zgodnych z E035 dev (120/120), wszystkie wartości finite.
  Primary 15 CMA: **1/15** gate, `primary_pass=false`.
- Stosunek parametrów U′/T: **1,099–1,821×**.

## Stare runy (rozpoznane, nie nadpisane)

- `runs/E037/20261108T030702Z_a7428da3`: manifest i 4 statusy `running` — **interrupted/running
  historical status**; PID 999264, 999305–999308 nieaktywne (`os.kill(0)`=ENOENT,
  `ps` brak), logi puste, 0 summary / 0 pomiarów / 0 costs.
- `runs/E036/20261107T225229Z_1f5b7f3a`: manifest `failed`, seed 42/43 `running`, seed 44/45
  `failed` — **failed historical status ze starymi statusami running**; PID 979331,
  979385–979388 nieaktywne, 4 częściowe `costs.json`, 0 plików pomiarowych.

## Ustalenia i korekty dokumentacji

1. **E036 — SD batch1:** raportować SD z surowych ratios (**0,0192–0,1002**),
   nie tylko średnie ~1; średnia bliska 1 nie dowodzi równoważności metod.
2. **E037 — interpretacja przyczynowa:** U′ ma **1,099–1,821×** parametrów T
   przy zbliżonym MAC. Nie rozdzielono wpływu obliczeń i pojemności;
   zdanie „zysk TILT wynika z compute” było zbyt mocne i zostało skorygowane.
3. **Snapshot vs bieżący analyze:** snapshot `source/experiment.py` (SHA `07d4656cd16f…`)
   nie ma `averages` i używa `m['wall_seconds']`; bieżący (SHA `0479d3159af3…`) dodaje
   `averages` i poprawia na `m['elapsed_seconds']`. `analysis.json` w runie ma `averages`
   i `wall_seconds=292,2807=manifest.elapsed_seconds`, więc powstał
   z poprawionej wersji; snapshot nie odtwarza analizy 1:1. (E036: `source/` nie zawiera
   `analyze.py`; bieżący `analyze.py` powstał po runie.)

## Granica dowodu

Spójność lokalnych artefaktów (manifesty, logi, summary, checkpointy, SHA256) silnie
potwierdza wykonanie, lecz nie stanowi zewnętrznego ani podpisanego dowodu fizycznej
pracy GPU. Hash dowodzi spójności plików, nie czasu ich powstania.

_Audyt: 2026-11-08T17:21:06.891235+00:00 → 2026-11-08T17:21:15.378359+00:00; artefakty: 1251._
