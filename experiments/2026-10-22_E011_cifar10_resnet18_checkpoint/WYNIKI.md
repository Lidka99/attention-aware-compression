# E011 — checkpointy CIFAR-10/ResNet18

Status: completed, wszystkie 4 workery exit0, po 200 epok.
Bieg `runs/E011/20261024T192205Z_018f3f2_seeds42-45_b2da2012`, kod `018f3f2`.
Start 2026-10-24 19:22 UTC, koniec 20:37 UTC (~75 minut).

| Seed | Dev top-1, epoka200 |
|---|---:|
|42|95,28%|
|43|95,60%|
|44|95,46%|
|45|95,50%|

Średnia 95,46%, rozstęp 0,32 pp. Cztery niezależne treningi, GPU0–3,
batch128 na model, bez DDP. Dev5000 z oficjalnego train, test nietknięty.
To przygotowanie wag do foldingu, nie wynik kompresji. Seed42 jest wejściem
E012 wybranym przed wynikami, nie najlepszym z czterech.

Kontrola `sha256sum -c artifacts.sha256`: wszystkie 31 wpisów OK.
Seed42 checkpoint SHA256:
`f74d18ef8906b296c7dcaa9184faead6aec22611d2129f6652c4a149f0813596`.
Odtworzenie predykcji dev jest osobną bramką E012.
