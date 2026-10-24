# 2026-10-22_E011_cifar10_resnet18_checkpoint

Protokół 1.0, frozen 2026-10-22 przed biegiem głównym.
Status wykonania określa manifest pod `runs/E011/`.

## Cel

Wytrenować checkpointy CIFAR-10/ResNet18 potrzebne do późniejszego foldingu
bez fine-tuningu. To trening modeli źródłowych, nie naprawa po kompresji.
Nie jest to jeszcze porównanie miar ani wynik kompresji.

Repo autorów nie udostępnia wag zwykłego ResNet18/CIFAR-10. Wybieramy
recepturę z ich siatki, bez wyboru hiperparametrów na podstawie wyników.
Trening implementujemy samodzielnie z torchvision; nie kopiujemy kodu autorów.

Źródło receptury:
https://github.com/osaukh/folding_as_projection/tree/694691556e8314a835178a741938dd3d4d80c171/train
(`cifar10_learn_configurable.py`, `run.sh`).

## Ustalenia zamrożone

- Model: torchvision ResNet18 bez wag, stem 3×3/stride1/padding1,
  maxpool=Identity, fc=512→10, 11 173 962 parametrów.
- Cztery niezależne treningi: GPU0 seed42, GPU1 seed43, GPU2 seed44,
  GPU3 seed45. Batch 128 na GPU i na model; brak DDP/akumulacji.
  Zachowujemy batch autorów. Jeden główny checkpoint seed42, pozostałe
  do powtórzeń; nie wybieramy najlepszego seeda po wyniku.
- Dane: oficjalne train 50 000; własny podział 45 000 train/5 000 dev,
  stratyfikowany 4500/500 na klasę, seed 20261022, wspólny dla seedów.
  Archiwum źródłowe zawiera również test, ale wypakowujemy i ładujemy
  wyłącznie pliki train i metadane. Testu nie oceniamy.
- SGD lr0.1, momentum0.9, wd0.0005 do wszystkich parametrów,
  bez Nesterov, CE bez label smoothing. 200 epok, cosine do 0,
  krok schedulera po epoce, bez warmupu, FP32, bez AMP.
- Train: RandomCrop(32,pad4), HFlip, Rotation15, ToTensor,
  mean=(.4914,.4822,.4465), std=(.2023,.1994,.2010).
  Dev: CenterCrop32, ta sama normalizacja.
- Wynik główny: epoka 200. Brak early stoppingu i selekcji best.
  Metryki dev co epokę służą diagnostyce. Raportować każdy seed oraz
  średnią i odchylenie standardowe między seedami; nie traktować obrazów
  jako niezależnych powtórzeń treningu.
- `last.pt` nadpisywany atomowo co epokę, tylko wagi i metadane.
  Bez stanu optymalizatora: awaria oznacza nowy bieg, nie wznowienie.
- Nowy katalog run na każde uruchomienie, snapshot źródeł/config,
  hashe danych i podziału, wersje środowiska, przypisanie GPU i logi.

## Ograniczenia i zgodność

To adaptacja receptury autorów, a nie odtworzenie identycznego checkpointu:
wydzielamy dev z train, używamy lokalnych wersji torch/torchvision,
czterech seedów oraz deterministycznych algorytmów. Wynik na dev nie jest
bezpośrednio porównywalny z wynikiem artykułu na oficjalnym teście.

Nie ma progu accuracy dobranego po wyniku. Po zakończeniu: sprawdzić
kompletność 200 epok, skończone metryki, hashe i zgodność checkpointu z
oceną. Słaby wynik raportujemy, nie wybieramy innej receptury po cichu.

## Uruchomienie

Z katalogu głównego repo, środowisko Conda attention:

```bash
CUDA_VISIBLE_DEVICES='' /home/users/s224574/miniconda3/envs/attention/bin/python -B experiments/2026-10-22_E011_cifar10_resnet18_checkpoint/check_cpu.py
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/users/s224574/miniconda3/envs/attention/bin/python -B experiments/2026-10-22_E011_cifar10_resnet18_checkpoint/launch.py --smoke
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/users/s224574/miniconda3/envs/attention/bin/python -B experiments/2026-10-22_E011_cifar10_resnet18_checkpoint/launch.py
```

Smoke: 2 epoki na 1000 train/200 dev, wszystkie cztery GPU, osobny run,
bez znaczenia naukowego. Launcher odmawia uruchomienia przy zajętych GPU
lub braku czterech urządzeń. Szacunkowy zapis: ~180 MB wag na run,
plus ~325 MB danych (archiwum i train), logi i snapshoty.

## Kolejny eksperyment

E012: kontrola referencyjnego foldingu na checkpointcie seed42, potem
porównanie grupowania. Protokół E012 zostanie zamrożony po walidacji wejścia
i operatora; nie uruchamiać automatycznie niezweryfikowanego foldingu.
