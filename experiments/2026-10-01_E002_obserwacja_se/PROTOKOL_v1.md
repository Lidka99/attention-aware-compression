# E002 — zamrożony protokół obserwacji SE, v1.0

Data: 2026-10-01 UTC. Status protokołu: **frozen**. Status wykonania:
manifest konkretnego biegu; samo zamrożenie nie oznacza pomiaru.
Eksperyment: `2026-10-01_E002_obserwacja_se`.

## Pytanie, cel, model i pomiary

Przyjmujemy zakres, pytanie, miary, wybór par, kryteria i ograniczenia
z [OPIS.md](OPIS.md), wersja 0.1-draft, **sekcje 1–7 i 9**. Zachowujemy
szkic, a ta wersja uzupełnia jego sekcję 8 działającym kodem i wejściami.
Nie zmieniono parametrów naukowych szkicu: potwierdza to kontrola CPU
porównująca konfiguracje z wyjątkiem pól status/protocol_version.

- Jedno pytanie: czy podobieństwo kanałów przed SE współwystępuje z
  różnicami względnego ważenia i podobieństwem po SE, na innych obrazach?
- Jeden checkpoint best.pt E001 i SE `features.4.1.block.2`, 480 kanałów.
- 1000 obrazów train, A=500 i B=500, po 5 obrazów/klasę w każdej połowie.
- 114960 par na każdej połowie, 100 par wybranych wyłącznie według
  najwyższego s_h na A; remisy według c,d. Nie zastępujemy par na B.
- Zapis h=input SE, a=wyjście Sigmoid, z=output SE=a*h. Hooki tylko odczytują.
- Główna kontrola: porównanie s_h i s_z oraz d_h i d_z na B dla tych samych
  par, z jednoczesnym opisem std log-stosunku wag. Pełne tabele również
  dla pozostałych par. To wynik opisowy, nie test skuteczności kompresji.
- Nie dobieramy progu „sukcesu metody” po wyniku. Bliskość zera i
  nieokreślone wartości raportujemy; nie wnioskujemy o praktycznej
  korzyści z samego wystąpienia niezerowych różnic.

Kontrole wymagane przed uznaniem pomiaru za ukończony: hashe wejść,
1000 unikalnych ID z poprawnymi etykietami i rozłączność A/B, dokładnie
250 obrazów/rank, poprawne kształty, finite h/a/z, a w [0,1], jedno
wywołanie każdego hooka/batch, z=a*h z atol=1e-7, rtol=1e-6, identyczny
hash modelu przed/po i między rankami, komplet tabel oraz wykresów.
NaN dla nieokreślonych cosinusów zerowych kanałów jest jawnie dozwolone,
a NaN/Inf w surowych aktywacjach nie. Nie odrzucamy niekorzystnych wyników.

## Utrwalone wejścia

- [config.json](config.json): dokładna konfiguracja, ścieżki względem root repo.
- [SAMPLES.json](SAMPLES.json): pełne ID, klasy i przynależność do A/B,
  utworzone przed inferencją, seed wyboru 20261001.
- SHA-256 SAMPLES.json:
  `338eb31862ab3d35495c22c8ae58109aed280a8218f1c32e234c941c3d1bae1d`.
- [INPUTS.json](INPUTS.json): hashe wszystkich sprawdzonych artefaktów E001,
  plików train/meta/splits i wersji źródłowego kodu treningu.
- Checkpoint SHA-256:
  `f2ac9c3fddd140c0d45f2c244c78b432cbdd164b46dff32d00be956202ec5be8`.
- Źródłowy commit treningu: `7cc3f46667990ff45686f7499c5dfd00d6d57649`.

Nie używamy test/dev/rezerwy do inferencji. Czytamy i hashujemy wspólny
kontener train; tylko wybrane 1000 obrazów trafia do modelu. Listy innych
splitów służą wyłącznie kontroli rozłączności metadanych.

## Środowisko i podział czterech GPU

Conda attention, Python 3.11.15, torch 2.12.0+cu130, torchvision 0.27.0+cu130,
NumPy 2.4.6, cztery RTX 2080 Ti 11 GB. Precision FP32 inferencji i FP64
miar na CPU. Jedna replika istniejącego modelu/proces, cztery procesy,
rank r ocenia records[r::4]. Po 250 obrazów, batch 32 i końcówka 26.
Brak dopełniania, brak odrzucania końcówki. To jedna obserwacja, nie
cztery niezależne powtórzenia modelu.

Seed procesu 42+rank; seed generatora loadera 42+rank+100000; 2 workery
z inicjalizacją RNG, 1 wątek CPU. Algorytmy deterministyczne, bez AMP/TF32.
Launcher ustawia CUDA_VISIBLE_DEVICES=0,1,2,3, CUBLAS_WORKSPACE_CONFIG=:4096:8,
PYTHONHASHSEED=42, OMP_NUM_THREADS=1 oraz OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1
dla powtarzalnej i ograniczonej zasobowo analizy CPU. Stany RNG przed/po
zapisuje osobno każdy rank. Różnice środowiska względem E001 są raportowane.

## Kod, kontrole i dokładne uruchomienie

- `common.py` i `prepare_inputs.py`: kontrola źródeł, odczyt wyłącznie train,
  deterministyczne listy. Przygotowanie odmawia nadpisania INPUTS/SAMPLES.
- `collect.py`: cztery procesy i kontrolowany odczyt hooków.
- `pair_metrics.py`: czyste miary NumPy FP64.
- `analyze.py`: kontrola ID, tabele A/B, wybór na A, opis i wykresy B.
- `launch.py`: nowy katalog biegu, zapis wersji, logów, hashy i statusu.
- `check_cpu.py`: kontrole na małych tablicach i losowo inicjalizowanym
  modelu CPU, bez checkpointu i CIFAR. Wynik: [TESTY_CPU_v1.json](TESTY_CPU_v1.json).
  Wcześniejszy [TESTY_CPU.json](TESTY_CPU.json) pozostaje historycznym dowodem
  dla szkicu. Nie zastępuje to odczytu realnego modelu.

Kontrole obejmują miary względem bezpośrednich wzorów, zerowe kanały,
stałe i zmienne skalowanie, kolejność remisów, kształty hooków, z=a*h,
niezmienność wyjścia/modelu i odrzucenie powtórnego wywołania hooka.
Launcher wymaga zgodności hashy kodu i configu z kontrolami CPU.

Z katalogu głównego repozytorium, przy czystym drzewie po lokalnym commicie:

```bash
conda run --no-capture-output -n attention python experiments/2026-10-01_E002_obserwacja_se/launch.py
```

Dokładny commit kodu zostanie zapisany przez launcher przed pomiarem w
manifest.json. Dokument, config, listy ID, kod i dowód kontroli muszą
być częścią tego commita. Nie uruchamiamy ze zmienionego kodu po teście.

## Wyniki i odtworzenie

Nowy `runs/E002/<UTC>_<git>_seed42_<unikalny>/`: config, manifesty wejść,
źródła, środowisko/eksport Condy/pip, logi, RNG, rank*.npz i kontrole ranków,
pairs_A/B.csv i npz, channels_A/B.csv, selected_pairs.csv, summary.json,
status i artifacts.sha256 po zamknięciu logów. Surowe aktywacje i duże
tabele pozostają poza Git. Brak zewnętrznej kopii zapasowej jest ograniczeniem.

W `wykresy/<run-ID>/` tego eksperymentu: PNG/PDF dla trzech porównań,
rozkładu zmienności log-stosunku i rozkładów wag kanałów; małe summary.json
i selected_pairs.csv. Ich hashe zapisuje manifest biegu. Raport WYNIKI.md
powstanie po analizie i wskaże run-ID, dowody, obserwacje oraz ograniczenia.

Powtórzenie launchera tworzy nowy bieg. Nie nadpisujemy zakończonych wyników.
Sama analiza istniejących aktywacji na CPU (nowy katalog wykresów, a także
świeży katalog z kopiami rank*.npz, config.resolved.json i SAMPLES.json,
bez wcześniejszych tabel/summary) używa:

```bash
conda run -n attention python experiments/2026-10-01_E002_obserwacja_se/analyze.py --run-dir <NOWY_KATALOG_Z_KOPIAMI_WEJSC> --plots-dir <NOWY_KATALOG_WYKRESOW>
```

Weryfikacja: `sha256sum -c artifacts.sha256` w katalogu biegu i hashe
wykresów z manifestu. Na innym środowisku nie gwarantujemy zgodności bitowej;
obecny bieg nie jest dowodem odtworzenia na innym komputerze.

## Granice wnioskowania

Jeden blok, jeden seed treningu, obrazy train, przeuczenie E001. Pary
współdzielą kanały; nie są niezależnymi replikami. A/B sprawdza zachowanie
na innych obrazach, nie generalizację poza treningiem. Nie mierzymy jakości
po scaleniach ani przewagi osobnego kryterium a ponad aktywacje z.
Po wyniku zatrzymujemy się przed etapem 4 i ustalamy kolejny mały krok.
