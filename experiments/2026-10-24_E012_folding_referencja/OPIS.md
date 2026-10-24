# 2026-10-24_E012_folding_referencja

Status protokołu: **frozen 1.0 przed biegiem**. Status wykonania w manifeście.
Poniższy szkic zakresu uzupełnia wiążąca specyfikacja wykonania na końcu pliku.

## Cel

Odtworzyć referencyjny folding dla CIFAR-10/ResNet18 bez treningu po
kompresji. Najpierw zgodność operatora i kontrola działania, dopiero później
alternatywne miary w osobnej próbie badawczej.

## Planowane wejście

Seed42, ostatnia epoka 200 z E011. Nie wybieramy najlepszego seeda po dev.
Konkretny hash checkpointu zostanie wpisany do zamrożonego configu po
zakończeniu i audycie E011. Seedy43–45 pozostają do późniejszych powtórzeń.

## Zakres do zamrożenia

- BASE oraz oryginalne grupowanie HKMeans/L2 na powiązanych parametrach.
- Budżety proponowane: 0/10/30/50% kanałów per grupa. Rzeczywista redukcja
  parametrów/FLOPs mierzona osobno; identyczne szerokości dla porównań.
- Wyniki przed korektą BN i po identycznej korekcie BN; BASE także po BN.
- Brak gradientów i optymalizatora. Dane do BN tylko z train E011,
  ocena na dev E011. Test nietknięty.
- Cztery niezależne zadania GPU0–3 dla czterech budżetów; nie są to seedy.
- Bramki: singleton/zero compression, residual shapes, niepuste klastry,
  skończone parametry i logity, zachowanie predykcji po zapisie modelu.

## Aktualny brak

Operator zaimplementowany w `folding.py`; testy CPU i integracja HKMeans
opisane w `TESTY.md`. Nie jest to bitowo dokładna reprodukcja całego pipeline'u.
Audyt opisuje konwencję macierzy sum/średnich i przebudowę BN:
[audyt](../../literature/AUDYT_CUT_LESS_FOLD_MORE.md).
## Wiążąca specyfikacja wykonania 1.0

1. Wejście: E011 seed42 epoka200; hash checkpointu/splitu w `config.json`.
   Odtworzyć na GPU dev top-1 dokładnie 4764/5000 (95,28%) i NLL
   0,16766352424621583 z tolerancją 1e-5. Inaczej przerwać.
2. CPU folding na bieżących wagach, 12 grup w kolejności źródłowej,
   konkatenacja powiązanych parametrów bez running stats, PCA pełny,
   HKMeans0.0.6 n_init1/max_iter10/random_state42/n_jobs1.
3. Cztery procesy: GPU0 ratio0, GPU1 ratio.1, GPU2 ratio.3, GPU3 ratio.5.
   Każdy ocenia pełne 5000 dev. To jeden seed wejściowy, nie cztery powtórzenia.
4. Ratio0 jest bypassem, nie testem ścieżki klastrowania przy k=n.
   Dla pozostałych ratio: odtworzona konwencja conv sum/mean i BN mean.
   Przypadek pustego klastra przerywa bieg, bez zmiany budżetu.
5. Dwa jawne warianty na identycznych grupach: `ref_behavior` odtwarza
   pozostawienie nowego losowego biasu fc; `preserve_bias` kopiuje oryginalny
   bias. Kod autorów gubi bias przy odbudowie Linear. Ten drugi wariant jest
   kontrolą techniczną, nie nową miarą grupowania.
6. Nie obiecujemy bitowej zgodności z autorami: PCA/sklearn lokalnej wersji,
   CPU folding, lokalny seed odbudowy modułów i torchvision zamiast custom
   ResNet. Odtwarzamy badane reguły operatora; nie dokładne liczby artykułu.
7. BN: 1 przejście 45 000 train E011, batch128, shuffle, augmentacje treningowe
   E011, seed20261024 resetowany identycznie dla wariantów. Bez resetowania
   buforów i zmiany momentum (tak działa `repair_bn` w źródle). Bez gradientów
   i optymalizatora. Weryfikacja identycznego hasha parametrów przed/po BN.
8. Eval przed i po BN; zapis NPZ logits/labels/indices, stanu modelu po BN,
   grup, NLL/top1, parametrów i MACs Conv/Linear (FLOPs=2MACs, bez BN/ReLU).
   Kontrola strict reload i identycznych logitów na wejściu syntetycznym.
9. Wynik opisowy dla jednego checkpointu. Nie ustanawiamy zwycięzcy nowej
   metody; brak testów hipotez i optymalizacji na dev. Oficjalny test nietknięty.

## Środowisko i odtworzenie

Conda attention: torch2.12.0+cu130, torchvision0.27.0+cu130, numpy2.4.6,
scikit-learn1.9.1, scipy1.17.1. Pełne pip freeze w każdym biegu.
Hartigan0.0.6: PyPI sdist pomija requirements.txt i nie buduje się na cp311.
Zainstalowano tę samą wersję z oficjalnego taga v0.0.6, commit
`1648be1b61420bf33156683adbd47ba722c5c4ae` repo IBM/hartigan-kmeans:

```bash
python -m pip install --no-deps --no-build-isolation 'git+https://github.com/IBM/hartigan-kmeans.git@1648be1b61420bf33156683adbd47ba722c5c4ae'
```

Nie zmieniono numpy/torch/torchvision E011. Zależność lokalna `train.py` E011
pinowana SHA256 w configu (dataset/preprocessing/model). Komenda z root repo:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/users/s224574/miniconda3/envs/attention/bin/python -B experiments/2026-10-24_E012_folding_referencja/launch.py
```

Launcher wymaga czterech wolnych GPU, 2 GiB wolnego dysku i commitowanego
folderu E012. Każdy bieg nowy `runs/E012/<UTC>_<commit>_<id>`.
