# E047 — potwierdzenie L06/b10 na ImageNetV2

**Kryterium potwierdzenia spełnione.** Na 2 000 nowych obrazach z
ImageNetV2 MatchedFrequency (200 klas Tiny) zamrożony L06 poprawia top‑1
względem UNIFORM o **+1,3500 ± 0,9704 pp (4/4)** pod oboma pułapami kosztu.
95% sparowany przedział bootstrap po obrazach: **[+0,700; +2,025] pp**,
warunkowo na czterech checkpointach. Strata do pełnego modelu maleje z
**3,3500 do 2,0000 pp**.

| Model po A4096/CMA | Top‑1 ± SD [%] | Sparowana strata do pełnego ± SD [pp] | Parametry | MAC/obraz |
|---|---:|---:|---:|---:|
| Pełny | 30,3375 ± 0,3198 | 0 | 11 279 112 | 148 148 224 |
| UNIFORM b10 | 26,9875 ± 1,0711 | 3,3500 ± 0,7853 | 10 111 430 | 132 336 196 |
| L06/b10 | **28,3375 ± 0,2496** | **2,0000 ± 0,2483** | **10 104 855** | **132 302 192** |

SD opisuje zmienność po seedach. Koszty L06 oznaczają redukcję **10,4109% P**
i **10,6961% MAC** względem pełnego modelu; zapas względem U wynosi 6 575 P
i 34 004 MAC/obraz.

| Seed | Pełny [%] | U/b10 [%] | L06/b10 [%] | Pełny−U [pp] | Pełny−L06 [pp] | L06−U [pp] |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 30,00 | 25,60 | 28,10 | 4,40 | 1,90 | +2,50 |
| 43 | 30,40 | 27,70 | 28,15 | 2,70 | 2,25 | +0,45 |
| 44 | 30,75 | 27,95 | 28,60 | 2,80 | 2,15 | +0,65 |
| 45 | 30,20 | 26,70 | 28,50 | 3,50 | 1,70 | +1,80 |

## Zakres wniosku

Na Tiny dev w E046 kontrast wyniósł +1,1425 pp; E047 potwierdza jego
kierunek i dodatnią średnią na innym źródle obrazów, dla dokładnie tych
samych modeli. Pełny model ma w E047 30,3375%, wobec 51,2200% na Tiny dev;
bezwzględnych trafności nie należy uśredniać między źródłami. To replikacja
na 200 klasach Tiny w ImageNetV2, nie oficjalny test Tiny ani dowód transferu
na nowe checkpointy, klasy czy architektury. Zbiór E047 został już użyty.

## Wykonanie

- [CPU i proweniencja](../../runs/E047/cpu_20261122T053249Z_169cdc5b/manifest.json),
  68,48 s, exit 0. Dwie zgodne mapy klas, 10 obrazów na klasę. [Audyt danych](../../runs/E047/cpu_20261122T053249Z_169cdc5b/audit.json):
  0 wykluczeń i 0 wykrytych duplikatów metodą z protokołu po porównaniu
  z 110 000 obrazów Tiny train/dev/val oraz wewnątrz nowej próby. Val służył
  do audytu rozłączności. Skrót cache: `141584b173a8d7c5898950ef6405227a4494e1f5c12d27282545adde7efd1725`.
- [Run](../../runs/E047/20261122T053804Z_ae2dec73/manifest.json): **56,25 s,
  4×RTX 2080 Ti, 4×exit 0**, seedy 42–45 po jednym na GPU. Zgodność 12/12
  stanów E046 po CMA i P/MAC przed otwarciem testu; 12/12 nowych ocen.
  [Bramka](../../runs/E047/20261122T053804Z_ae2dec73/gate.json) ·
  [Audyt wykonania PASS](../../runs/E047/20261122T053804Z_ae2dec73/analysis/audit.json) ·
  [Trafienia i koszty](../../runs/E047/20261122T053804Z_ae2dec73/analysis/per_seed.csv) ·
  [Agregaty](../../runs/E047/20261122T053804Z_ae2dec73/analysis/summary.json).

[Figura porównawcza b10: Tiny dev i ImageNetV2](wykresy/20261122T053804Z_ae2dec73/README.md) ·
[Zamrożony protokół](OPIS.md) · [Odtworzenie](URUCHOMIENIE.md).
