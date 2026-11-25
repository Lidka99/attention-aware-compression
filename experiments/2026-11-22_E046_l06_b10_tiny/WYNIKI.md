# E046 — L06 zmniejsza stratę przy b10 na Tiny dev

**Ukończony 2026-11-22.** Przy **10,41% redukcji P i 10,70% redukcji MAC**
L06 osiąga **47,5050%** top‑1 po CMA. Strata do pełnego modelu spada z
**4,8575 do 3,7150 pp**: poprawa względem UNIFORM wynosi **+1,1425 ± 0,8820 pp**
(4/4 dodatnich par). Oba koszty mieszczą się pod pułapami UNIFORM z E045.

| Model | P | MAC/obraz | Redukcja P / MAC | Top‑1 ± SD [%] | Sparowana strata do pełnego ± SD [pp] |
|---|---:|---:|---:|---:|---:|
| Pełny (odtworzenie E045) | 11 279 112 | 148 148 224 | 0% / 0% | 51,2200 ± 0,1602 | 0 |
| UNIFORM b10 (odtworzenie E045) | 10 111 430 | 132 336 196 | 10,3526% / 10,6731% | 46,3625 ± 0,4687 | 4,8575 ± 0,5309 |
| L06/b10 (E046) | **10 104 855** | **132 302 192** | **10,4109% / 10,6961%** | **47,5050 ± 0,4458** | **3,7150 ± 0,5132** |

### Wynik per seed

| Seed | Pełny [%] | U/b10 [%] | L06/b10 [%] | Pełny−U [pp] | Pełny−L06 [pp] | L06−U [pp] |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 51,28 | 46,90 | 46,95 | 4,38 | 4,33 | +0,05 |
| 43 | 51,04 | 46,19 | 47,96 | 4,85 | 3,08 | +1,77 |
| 44 | 51,15 | 46,55 | 47,36 | 4,60 | 3,79 | +0,81 |
| 45 | 51,41 | 45,81 | 47,75 | 5,60 | 3,66 | +1,94 |

Z góry ustalone kryterium kierunku (dodatnia średnia i co najmniej 3/4
dodatnich par) jest spełnione. SD i zakres opisują zmienność po czterech
seedach na wspólnych 10 000 obrazach. Utrata dla L06 to **3,08–4,33 pp**
wobec 4,38–5,60 pp dla U. E046 ocenia jeden wcześniej zamrożony profil na
używanym **Tiny dev**, przy jawnie szerszej domenie projekcji. Nie jest
niezależnym potwierdzeniem na nowych danych ani wynikiem TILT przy b10.
Val Tiny (E035/E044) i rezerwa CIFAR-100 (E043) to inne fazy danych.

### Bramka geometrii na CPU

L06 wykorzystuje **99,9350% pułapu P** i **99,9743% pułapu MAC** (zapas
6 575 P i 34 004 MAC/obraz), spełniając okno [99,9%, 100%] obu kosztów.
Wektor `k` w kolejności grup protokołu to
`[64, 52, 53, 128, 127, 119, 233, 256, 219, 434, 512, 476]`;
RMS różnic `k/n` względem U to 7,20 punktu procentowego szerokości.
Projekcja zbiegła w 9 iteracjach; po zaokrągleniu naprawa nie była potrzebna.

**Ograniczenie starej reguły:** przy `k ≤ floor(0,9n)` maksymalne koszty to
9 112 788 P i 119 832 480 MAC/obraz. Wymusza to co najmniej **19,21% redukcji P**
i nie pozwala wejść w okno kosztów b10. W E046 dopuszczono `k ≤ n`: osiem
szerokości przekracza stare ograniczenie, w tym cztery zachowują pełną
szerokość grupy. To wynik dla L06 z nową regułą projekcji.

## Ślad wykonania

- [Manifest CPU](../../runs/E046/cpu_20261122T042524Z_aa692377/manifest.json) ·
  [Geometria](../../runs/E046/cpu_20261122T042524Z_aa692377/geometry.json) ·
  [Zamrożony plan](../../runs/E046/cpu_20261122T042524Z_aa692377/frozen.json),
  SHA‑256 `e7b4ffb9b839bb69cf35d35487afd032e1e661ec5ba23f66163d25533a8c7d84`.
- 111,84 s, exit 0, Conda `attention`; SHA‑256 wag zgodne 4/4, kontrola
  konstrukcji i kosztów 12/12 (pełny, U/b10, L06/b10 × seedy 42–45).
  Do liczenia MAC użyto syntetycznego tensora zerowego; w fazie CPU nie
  odczytywano obrazów i nie wykonywano CMA.
- [Run GPU](../../runs/E046/20261122T045655Z_0b0c53b7/manifest.json): **62,10 s,
  4×RTX 2080 Ti, 4×exit 0**. Przed czterema ocenami L06 odtworzono dokładnie
  **8/8** stanów po CMA i liczb trafień pełnego modelu oraz U/b10 z E045.
  Te same seedy E019, A4096/CMA i Tiny dev. Bez treningu, otwierania val
  i rezerwy. [Audyt PASS](../../runs/E046/20261122T045655Z_0b0c53b7/analysis/audit.json) ·
  [Trafienia i koszty per seed](../../runs/E046/20261122T045655Z_0b0c53b7/analysis/per_seed.csv) ·
  [Agregaty](../../runs/E046/20261122T045655Z_0b0c53b7/analysis/summary.json).

[Protokół](OPIS.md) · [Odtworzenie CPU](URUCHOMIENIE.md) · [Odtworzenie GPU](URUCHOMIENIE_GPU.md).
