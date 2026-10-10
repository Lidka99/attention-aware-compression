# E004 — wersja wykonania 1.0
Eksperyment: `2026-10-10_E004_wspolne_dostrajanie`.
Data zamrożenia: 2026-10-10 UTC. **Status protokołu: frozen**.
Status wykonania podaje manifest biegu, nie sam status konfiguracji.

Przyjmujemy pytanie, budżet, dane, kontrole i ograniczenia z zachowanego
szkicu [OPIS.md](OPIS.md), v0.1-draft. Ta wersja uzupełnia go kodem
i dokładną komendą. [config.json](config.json) ma te same parametry
naukowe co szkic (poza statusem i numerem wersji).

## Pytanie, warunki i wynik główny

Po **jednakowym** dostrojeniu BASE/PRE/POST/RANDOM (120 epok) porównujemy:
(a) POST−PRE na tych samych 5000 obrazów dev, (b) zmianę każdego wariantu
względem wyniku E003 przed naprawą, (c) wynik wobec BASE po tym samym
treningu. Wynik główny: epoka 120 (ostatnia); epoka najlepsza na dev jest
wyłącznie diagnostyką z jedną regułą dla wszystkich wariantów. Bez early
stopping i bez wybierania po fakcie.

Budżet: adaptacja opublikowanego protokołu CIFAR z GReg (Wang i in.,
ICLR 2021, arXiv:2012.09243, Tab.5): SGD (momentum0,9, wd5e-4),
LR0,01 z redukcją po60→0,001 i po90→0,0001, batch256, 120 epok.
Adaptacja do EfficientNet-B0/96×96/100klas — jawnie opisana w OPIS.md.

## Kod i kontrole

- `support_e004.py` — ścieżki, hashe, zależności.
- `train_finetune.py` — cztery warianty kolejno, każdy DDP na 4 GPU,
  AMP FP16, SyncBatchNorm, deterministyczne algorytmy.
- `summarize_finetune.py` — CPU: walidacja metryk/predykcji, sparowane
  bootstrapy (POST−PRE, POST−RANDOM, PRE−RANDOM) i 3 wykresy.
- `check_pipeline_cpu.py` — kontrole przed biegiem: składnia, zgodność
  config-draft, harmonogram LR, wczytanie wszystkich czterech modeli
  E003 (strict, liczniki, hashe), forward/backward na sztucznych danych,
  niezmienność modelu bazowego, sanity bootstrapu.
- `launch_finetune.py` — nowy bieg, commit/środowisko/komendy/logi/hashe.
- Dowód kontroli: [TESTY_CPU_v1.json](TESTY_CPU_v1.json). Testy są na
  sztucznych danych CPU i nie oznaczają wykonania dostrajania.

## Wejścia

Modele E003 (inference-only, hashe w config.json), train/dev z E001
(splits.json `ecb8804e…`, plik train `735e79b0…`), zero testu i rezerwy.
Każdy wariant startuje z tych samych wag, świeżego SGD i identycznych
seedy; próbki dev każdego wariantu identyczne i w tej samej kolejności.

## Wykonanie na 4 GPU

Z głównego katalogu repo, przy czystym drzewie po commicie kodu/protokołu
i dowodu kontroli:

```bash
conda run --no-capture-output -n attention python experiments/2026-10-10_E004_wspolne_dostrajanie/launch_finetune.py
```

Launcher tworzy nowy `runs/E004/<UTC>_<git>_seed42_<unikalny>/`, zapisuje
commit, konfigurację, środowisko/Conda/pip, komendę, logi, a po treningu
uruchamia analizę CPU i zapisuje wykresy w `wykresy/<run-ID>/` tego
eksperymentu. `artifacts.sha256` po zamknięciu logów. Nie nadpisujemy
biegów; powtórzenie ma nowy katalog.

## Kontrole techniczne biegu

Dokładnie120 epok/wariant; ten sam batch i pokrycie danych; finite loss;
hashe wejściowych modeli niezmienione po treningu; dokładnie5000 unikalnych
ID dev na wariant; zgodność macierzy/argmax/metrics; hashe modeli zgodne
między4GPU; brak użycia testu i rezerwy. Wynik jakościowy — także brak
różnic między metodami — nie jest błędem technicznym.

## Ograniczenia z góry

Jeden model bazowy (seed42), dev używany wcześniej, budżet zaadaptowany
z innej architektury, jeden poziom kompresji, brak niezależnego testu.
Cztery warianty to jeden przebieg sekwencyjny na4GPU, nie cztery
niezależne treningi. E004 nie dowodzi nowości metody; rozstrzyga, czy
po równym budżecie różnice PRE/POST/RANDOM w ogóle się utrzymują.
