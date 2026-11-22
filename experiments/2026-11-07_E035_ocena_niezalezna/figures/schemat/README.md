# E035 — schemat procedury do szkicu artykułu

Jednostronicowy schemat (PDF wektorowy + PNG) pokazujący, gdzie w procedurze
E035 jest alokacja budżetu parametrów, a gdzie są znane operatory i standardowa
rekalibracja BN. Materiał pomocniczy do szkicu; nie jest nowym wynikiem ani
oceną modeli.

## Co pokazuje

- **Checkpoint i legalne grupy kanałów.** Decyzja dotyczy grupy sprzężonych
  osi (producent → fan-in, affine BN, conv skrótu), a nie pojedynczej warstwy;
  12 grup w ResNet18 i 13 w PreActResNet18.
- **Profile UNIFORM i TILT przy docelowym P.** Przykład b50 ResNet18/CIFAR-10
  z zamrożonego planu E035: ochrona wczesnych grup i większa kompresja późnych.
- **Wspólna geometria k dla FOLD i MAG2.** Oba operatory są znane i przy tym
  samym k mają identyczne szerokości, P i MAC.
- **BN bez gradientów.** Reset BN i CMA na A ⊂ train (4096 obrazów, crop/flip,
  batch 128), bez aktualizacji wag, na osobnych kopiach stanu.
- **Ocena i dwa liczniki.** P (parametry sieci, bez buforów BN) i MAC
  (Conv/Linear dla ustalonego wejścia) są liczone osobno. Równy budżet P nie
  wymusza równego MAC: dla przykładu T/U = 0,99964, a MAC T/U = 1,364 (E036).

## Rozdzielenie faz — ważne

- **E035 (ocena niezależna):** wyłącznie FOLD, oficjalny test CIFAR-10/100
  i oficjalny val Tiny.
- **E038 (kontrola dev):** MAG2, ten sam dev. Schemat **nie** sugeruje oceny
  MAG2 na teście E035.

## Źródła liczb

- Wektor szerokości i liczby parametrów: `E035/frozen.json`, para
  `resnet_cifar10`, budżet `b50`
  (SHA256 `0069e0126d472212f5cd18352301ef0319cc8875f45a7c50709b4bdab2321b78`).
- Iloraz MAC T/U: raport E036 (R/C10, 50%).
- Opis procedury i faz: `OPIS.md` / `WYNIKI.md` E035 oraz `WYNIKI.md` E036/E038.

Skrypt nie czyta obrazów, danych, checkpointów ani wag. Nie uruchamia GPU.

## Pliki

| Plik | Rola |
|---|---|
| `generuj_schemat.py` | generator (matplotlib, CPU); czyta tylko `../../frozen.json` |
| `schemat_procedury.pdf` | wersja wektorowa do szkicu |
| `schemat_procedury.png` | podgląd 230 dpi |
| `README.md` | ten opis |

## Odtworzenie

```bash
source /home/users/s224574/miniconda3/etc/profile.d/conda.sh
conda activate attention
cd /home/users/s224574/attention/badania_atencja_cnn/experiments/2026-11-07_E035_ocena_niezalezna/figures/schemat
python generuj_schemat.py
```

Skrypt zapisuje wyłącznie `schemat_procedury.png` i `.pdf` w tym katalogu
(guard sprawdza ścieżkę). Znaczniki czasu PDF są wyłączone
(`SOURCE_DATE_EPOCH=0`), więc przebieg jest powtarzalny wizualnie.

## Czytelność i skala

Figura ma 15,5 × 9,7 cala (między A4 a A3 poziomo). Najmniejsze etykiety mają
5,3–5,8 pkt, tekst opisowy 6,2–7,4 pkt w skali 100%. PDF ma jedną stronę,
jest wektorowy i tekstowy (osadzone podzbiory DejaVu Sans), więc powiększenie
nie traci jakości. Przy zmniejszeniu do szerokości A4
najdrobniejsze podpisy będą zbyt małe — do druku w A4 potrzebny jest podział
figury lub skala A3.
