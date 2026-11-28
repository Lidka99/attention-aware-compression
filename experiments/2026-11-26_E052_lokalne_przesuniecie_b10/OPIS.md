# 2026-11-26_E052_lokalne_przesuniecie_b10 — mała redystrybucja przy b10

**Pytanie.** Czy oddanie części kanałów z ukrytych grup `layer3` na rzecz
dwóch mocniej zwężonych grup `layer4` zmniejsza stratę top‑1 po foldingu?
Punktem odniesienia jest `runs/E051/dev_20261126T213703Z_20c2a1d5`: b10
TILT zwęża tylko grupy 9–11 (numeracja od 0), z 512 do 502/474/446.

1. **Pula, identyczna na obu zbiorach (maks. sześć nowych profili).** Do
   szerokości TILT dodajemy poniższe liczby kanałów; pozostałe szerokości
   zaczynają od TILT. W każdej z dwóch trójek (S01–S03 i S04–S06)
   warianty przenoszą podobną liczbę parametrów.

   | Profil | `stage4_out` (10) | `layer4.1_hidden` (11) |
   |---|---:|---:|
   | S01 | 4 | 0 |
   | S02 | 0 | 6 |
   | S03 | 2 | 3 |
   | S04 | 8 | 0 |
   | S05 | 0 | 12 |
   | S06 | 4 | 6 |

   Zdejmujemy po jednym kanale naprzemiennie z `layer3.0_hidden` (6)
   i `layer3.1_hidden` (8), zaczynając od 6, do pierwszego kroku z
   `P ≤ P(TILT)` na **obu** zbiorach. Maksymalnie 16 zdjętych kanałów
   w każdej z tych grup. Bez zmian `stage3_out` i `layer4.0_hidden`.
   Na CPU wymagamy unikalnych profili, szerokości w granicach modelu,
   `0,999·P(U) ≤ P ≤ P(U)`, redukcji P ≥9,5% do pełnego oraz
   `P ≤ P(TILT)` i `MAC ≤ MAC(TILT)` dla każdego zbioru. Brak przejścia
   kontroli geometrii zatrzymuje próbę. MAC oznacza Conv/Linear na obraz.
2. **Dane i selekcja.** Na zbiór jeden podzbiór 8 000 obrazów z
   `train \ A4096` bez augmentacji, rozłączny z dev. Sortujemy pulę
   indeksów, losujemy bez zwracania przez świeży `PCG64(20261126)` na zbiór
   i sortujemy wylosowane indeksy; zapisujemy je przed punktacją. Jeden
   profil na zbiór (wspólny dla seedów 42–45) to minimum średniego NLL
   etykiet twardych po czterech seedach, z remisem rozstrzyganym rosnącym
   ID. Tylko sześć nowych profili podlega selekcji. Podzbiór train był
   dostępny modelom podczas ich treningu E048; służy do wyboru, nie do
   raportowania poprawy generalizacji.
3. **Pomiar.** Jeden bieg, dwie fale po cztery GPU, seed 42–45 przypisany
   do GPU 0–3 w obu falach, Conda `attention`. W obu fazach checkpointy
   `last.pt` E048, identyczny adapter i upstream FOLD jak w E051
   (z zachowaniem biasu FC), FP32, te same widoki A4096 i CMA po resecie
   BN (32×128; Tiny crop min. 0,65). Bez treningu/dostrajania. Po wyjściu
   wszystkich workerów selekcji wynik reguły i hashe zapisów train
   zamrażamy w `selection.json`. Dopiero po tym uruchamiamy dev: raz po
   jednym zamrożonym profilu i po cztery kontrole `full`, `b10_U`,
   `b10_T`, `b10_L06` na zbiór i seed. Nie używamy oficjalnych testów/val,
   rezerwy CIFAR‑100, ImageNetV2 ani b5. Stan wybranego modelu po CMA
   musi być identyczny w obu fazach. Stan, A4096, trafienia i koszty
   kontroli muszą odtworzyć E051.
4. **Rozstrzygnięcie (dev).** Sparowana strata do pełnego ≤3 pp,
   średnio ≥1 pp ponad UNIFORM i ≥3/4 ściśle dodatnie pary względem U,
   oddzielnie na obu zbiorach. Dodatkowo Tiny ≥59,6125% (przy tym samym
   pełnym modelu), CIFAR‑100 ≥79,7500%. Raport: top‑1, strata, różnice
   do U, TILT i Frozen offsets per seed, rzeczywiste P i MAC. Wynik
   diagnostyczny na wcześniej używanym dev; bez poszerzania puli po
   punktacji. Oś porównania: dopasowane P i ten sam operator, MAC jako koszt.
