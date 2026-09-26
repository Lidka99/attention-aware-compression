# Uruchomienie E001 krok po kroku

Na razie przygotowany jest pilot CIFAR-100; pełny trening nie został uruchomiony. Najpierw potwierdzamy dane i zamrażamy [protokół](OPIS.md). Test syntetyczny można wykonać bez pobierania danych.

## 1. Wejdź do repo i aktywuj istniejącą Condę

```bash
cd /home/users/s224574/attention/badania_atencja_cnn
source /home/users/s224574/miniconda3/etc/profile.d/conda.sh
conda activate attention
python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.device_count())'
```

Oczekiwano: `2.12.0+cu130 0.27.0+cu130 4`. Nie używamy `.venv`; przerwana próba przygotowania tego środowiska nie jest częścią protokołu.

Na innej maszynie Linux x86_64, z nową instalacją Condy i odpowiednim sterownikiem, odtwórz środowisko przez:

```bash
conda create -n attention --file configs/environment/conda-linux-64.lock.txt
conda activate attention
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r configs/environment/pip-freeze.txt
```

Nie uruchamiaj `conda create` na istniejącym środowisku attention. Zapis attention.yml służy dodatkowo do przeglądu zależności. Pakiety pip i sterownik nie są objęte samym lockfile Condy. Świeża instalacja z tych plików nie została jeszcze przetestowana; sprawdź wersje i wykonaj test poniżej. Eksport środowiska nie zastępuje magazynu pakietów na wypadek niedostępności źródeł.

## 2. Test techniczny na 4 GPU

```bash
bash experiments/2026-09-26_E001_efficientnet_b0_od_zera/launch.sh --smoke
```

To dwie krótkie epoki na 256 sztucznych przykładach i ocena na 64 sztucznych przykładach, bez pobierania checkpointów ani CIFAR-100. Test dopuszcza niezatwierdzone zmiany Git, zapisując dirty=true; nie jest wynikiem naukowym. Uruchom go ponownie, aby sprawdzić powtarzalność. Każde wywołanie wypisze inny katalog wyników.

## 3. Przygotuj rzeczywiste dane

Po przyjęciu CIFAR-100 jako danych pilota:

```bash
python experiments/2026-09-26_E001_efficientnet_b0_od_zera/prepare_data.py --data-root data/cifar100
```

Skrypt pobiera oficjalny zbiór (nie wagi modelu), sprawdza integralność przez torchvision i zapisuje splits.json z listami ID oraz SHA-256 plików train/meta. Istniejącego innego podziału nie nadpisze. Zachowaj splits.json razem z wynikami; kopia trafia również do manifestu każdego treningu.

## 4. Przed właściwym treningiem

Potwierdź protokół i dane; zapewnij trwałą kopię artefaktów; zapisz kod, konfigurację i protokół w commicie. `git status --short` ma być puste. Kod blokuje właściwy trening przy niezapisanych zmianach. Zapis samego commita nie jest potwierdzeniem jakości modelu.

## 5. Właściwy trening od zera

```bash
bash experiments/2026-09-26_E001_efficientnet_b0_od_zera/launch.sh --data-root data/cifar100
```

Komenda działa na pierwszych czterech widocznych kartach (domyślnie 0,1,2,3). Przy przydziale przez klaster respektuje CUDA_VISIBLE_DEVICES; dokładnie cztery karty muszą być widoczne. Nie nadpisuj przydziału klastra. Trening to 200 epok; uruchom w trwałej sesji terminala lub przez mechanizm zadań swojego serwera. Skrypt nie uruchamia ukrytego procesu w tle.

## 6. Wznowienie po awarii

W poniższej komendzie zastąp RUN_ID dokładną nazwą wcześniejszego katalogu:

```bash
bash experiments/2026-09-26_E001_efficientnet_b0_od_zera/launch.sh \
  --data-root data/cifar100 --resume runs/E001/RUN_ID/last.pt
```

Używaj wyłącznie własnych checkpointów utworzonych tym skryptem. Wznowienie rozpocznie się od kolejnej pełnej epoki. Nie zmieniaj konfiguracji, wersji kodu ani środowiska. Nowy katalog zachowuje powiązanie z rodzicem; zachowaj oba. Na potrzeby testu syntetycznego dodaj --smoke i wskaż checkpoint tego testu.

## 7. Co sprawdzić po zakończeniu

- manifest.json: status=completed, exit_code=0, smoke=false dla rzeczywistego treningu.
- metrics.jsonl: przebieg każdej epoki, liczniki próbek i poprawnych odpowiedzi.
- metrics.json: najlepsze dev top-1 oraz zgodne hashe modelu na 4 GPU.
- best.pt: najlepszy checkpoint według dev; last.pt: pełny stan do wznowienia.
- best_dev_predictions.json: ID, etykieta i przewidziana klasa. Wynik na dev nie jest wynikiem na oficjalnym teście.
- artifacts.sha256: uruchom `sha256sum -c artifacts.sha256` wewnątrz katalogu wyników.

Wyniki są lokalnie w runs/, poza Git. Skopiuj cały katalog do uzgodnionego trwałego magazynu i zapisz lokalizację w WYNIKI.md. Samo .gitignore nie jest kopią zapasową.
