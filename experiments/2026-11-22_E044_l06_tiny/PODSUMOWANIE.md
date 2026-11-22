# E044 — Tiny ImageNet: ukończony, kontrola CPU PASS

**[Wyniki](WYNIKI.md):** L06−U po CMA **+2,100 pp (4/4)** w b50 i
**+0,255 pp (3/4)** w b30; główne kryterium spełnione. 28/28 ocen, 4 GPU,
4×exit 0, audyt PASS.

**Próba transferu L06 spoza dwóch CIFAR-ów.** Używamy istniejących wag
i oficjalnego val Tiny. Tiny nie uczestniczył w wyborze L06 w E042, ale
ten val był już użyty przez E035; wynik E044 dotyczy transferu konfiguracji
na wcześniej użytym zbiorze oceny. [Protokół](OPIS.md).

| Redukcja P | Profil | P | MAC/obraz 64×64 |
|---:|---|---:|---:|
| ≈50% | U | 5 623 142 | 75 359 828 |
| ≈50% | T-proj | 5 617 602 | 75 341 332 |
| ≈50% | L06 | 5 618 309 | 75 312 564 |
| ≈30% | U | 7 845 193 | 103 503 916 |
| ≈30% | T-proj | 7 843 030 | 103 495 000 |
| ≈30% | L06 | 7 838 059 | 103 449 808 |

**Wykonalność:** zachowano przesunięcia `k/n` L06 i kolejność 12 grup;
rachunek dostosowano do stema i rozdzielczości Tiny. Wszystkie profile
trafiają w **99,9–100% obu pułapów**. Na seedzie 42 zbudowano wszystkie
sześć modeli i pełny model; rzeczywiste P/MAC, mapa grup, klastry i bias FC
przeszły kontrolę. Pełny model: 11 279 112 P / 148 148 224 MAC. Potwierdzono
hashe 4/4 końcowych checkpointów E019.

**Dane:** 200 klas w kolejności klasyfikatora E019; A4096 z indeksów train
E035, odrębnie od dev; oficjalny val to 10 000 indeksów osobnego cache.
W fazie CPU sprawdzono metadane i oczekiwane hashe, bez otwierania cache
z obrazami. Zgodność obu cache potwierdzono w pomiarze GPU. Bez ImageNetV2.

**Ślad:** [manifest CPU](../../runs/E044/cpu_20261122T020819Z_00821a3f/manifest.json),
74,96 s, exit 0, Conda `attention`; [zamrożone wejścia i szerokości](../../runs/E044/cpu_20261122T020819Z_00821a3f/frozen.json),
SHA-256 `b145cf9d1c0e7e6785963dfe425a4c3df694db6ef8f4422d6ec1bdf70c191908`.
[Instrukcja odtworzenia](URUCHOMIENIE.md). Pomiar jakości wykonano w
`runs/E044/20261122T022955Z_6c1d2fc5` (124,97 s).
