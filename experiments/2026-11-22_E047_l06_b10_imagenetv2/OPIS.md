# 2026-11-22_E047_l06_b10_imagenetv2 — zewnętrzna replikacja b10

**Pytanie:** czy dokładnie te same modele z E046 zachowują przewagę L06
nad UNIFORM na nowych obrazach tych samych 200 klas? Uzgodniono jedną
ocenę na niewidzianych danych. Hipoteza dotyczy sparowanej straty top‑1
do pełnego modelu przy obu rzeczywistych pułapach E045.

## Źródło i bramka danych — bez predykcji

- Oficjalny test Tiny jest pozbawiony publicznych etykiet, a jego val
  wykorzystano już w E035/E044. Rezerwa CIFAR-100, wykorzystana w E043,
  jest odrębnym zbiorem. W E047 ustalamy z góry jeden zbiór zewnętrzny:
  [ImageNetV2 MatchedFrequency](https://github.com/modestyachts/ImageNetV2),
  rewizja `imagenetv2-b-33` ([Recht i in.](https://arxiv.org/abs/1902.10811)).
  Używamy wszystkich pozycji dla 200 `wnid` Tiny, 10 na klasę przed audytem.
  To niezależne źródło zdjęć i inny rozkład wejść niż oficjalny Tiny.
- Przypinamy archiwum z [wydania autorów](https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/d626240be2538720e83103a0e1178d24aca8b12c)
  (SHA‑256 `f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c`)
  oraz `class_info.json` i `imagenetv2-b-33.json` z commitu upstream
  `0b0c3fe89058e4c7ee2348938b3f4f706a931bf3`. Weryfikujemy komplet 10 000
  identyfikatorów i zgodność etykiet z dwóch źródeł: `cid → wnid` oraz
  bezpośrednio `ID obrazu → wnid`. Etykieta modelu to pozycja w kolejności
  `wnid` E019. Wejście: orientacja EXIF, RGB, cały obraz interpolowany
  biliniowo do 64×64, ta sama normalizacja Tiny co w E046, bez augmentacji.
- Przed udostępnieniem do punktacji porównujemy z **całymi** cache Tiny
  train/dev (100 000) i oficjalnego val (10 000; do audytu, nie do oceny).
  Usuwamy identyczne tensory RGB lub kandydatów o odległości Hamminga ≤4
  w 63-bitowym pHash (DCT 8×8 bez DC, mediana), również dla odbicia poziomego
  i centralnego kwadratowego kadru nowego zdjęcia. Oba warianty podstawowe
  i odbite są też używane w kontroli w obrębie nowej próby; jeżeli para
  jest oznaczona, usuwamy oba jej elementy. To konserwatywny filtr, nie
  dowód braku wszystkich podobnych obrazów. Bramka: ≥1800 i ≥8 w każdej
  z 200 klas. Po odrzuceniu nie dobieramy zastępstw. Utrwalamy listę obrazów,
  powody wykluczeń i skrót ostatecznej próbki przed jakąkolwiek punktacją.

## Jednorazowa punktacja

- **Te same stany co w E046:** jeden pełny model, U/b10 oraz zamrożony
  L06/b10 na każdym z czterech `last.pt` E019 (seedy 42–45). Z E046 bierzemy
  dokładne szerokości, bez ponownej projekcji. Zachowujemy ten sam operator
  CLFM, bias FC, A4096/CMA (train Tiny, 32×128), bez aktualizacji wag.
  Przed odczytem testu przez oceniających wymagamy zgodności **12/12**
  stanów po CMA, kosztów i widoków A z audytowanym E046. L06: 10 104 855 P
  i 132 302 192 MAC/obraz, U: 10 111 430 P i 132 336 196 MAC/obraz.
- Conda `attention`, dokładnie cztery wolne GPU, GPU 0–3 → seedy 42–45,
  po jednym procesie na kartę, deterministyczny FP32, batch 128. Punktacja
  tylko 12 par model–seed; wspólna bariera z PID→UUID i znacznik pierwszego
  użycia testu. Bez ponownej oceny po wyniku ani dostrajania na nowym zbiorze.
- Dla każdego seeda: top‑1, strata `pełny − model` w pp i poprawa straty
  `d = L06 − U` w pp. Podajemy średnie, próbne SD i rzeczywiste P/MAC.
  **Potwierdzenie** wymaga dodatniej średniej, dodatniego `d` na ≥3/4
  checkpointów i dolnej granicy 95% sparowanego przedziału bootstrap
  **>0 pp**. Bootstrap percentylowy: 10 000 losowań, seed 20261122,
  resampling obrazów wewnątrz każdej klasy, wspólne indeksy dla obu profili
  i wszystkich seedów. Przedział warunkowy na czterech modelach; nie
  opisuje zmienności po nowych treningach. Dodatni kierunek bez całej
  bramki będzie raportowany jako niejednoznaczna replikacja.
