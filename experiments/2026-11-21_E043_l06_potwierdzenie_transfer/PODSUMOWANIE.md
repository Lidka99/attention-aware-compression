# E043 — ukończony; kontrola CPU PASS

**[Wynik na rezerwie](WYNIKI.md):** L06−UNIFORM po CMA **+1,215 pp (3/4)**
w b50 — główne kryterium spełnione. W b30 **+0,945 pp (4/4)** — próg +1,00 pp
niespełniony. Rezerwa CIFAR-100 jest odtąd użyta.

**PASS:** cztery checkpointy E015 (42–45), po sześć modeli skompresowanych
i jednym nieskompresowanym na seed. Odtworzono dokładnie trzy profile b50
z E042 i przygotowano trzy profile b30. W fazie CPU nie mierzono trafności
ani nie kalibrowano BN na danych. [Protokół](OPIS.md) · [Odtworzenie](URUCHOMIENIE.md).

## Geometria

Przesunięcia L06 sprzed zaokrąglenia i projekcji, w setnych `k/n`:
`[+2, −15, −14, +13, −1, −5, −6, +9, −12, −13, +17, −6]`.
Kolejność: `relu_conv`, `layer1.0.relu1`, `layer1.1.relu1`, `layer2.0.relu1`,
`layer2.0.relu2`, `layer2.1.relu1`, `layer3.0.relu1`, `layer3.0.relu2`,
`layer3.1.relu1`, `layer4.0.relu1`, `layer4.0.relu2`, `layer4.1.relu1`.
Wektory całkowite przed i po projekcji zapisano w `frozen.json` runu.

| Redukcja nominalna P | Profil | P | MAC/obraz | P / U | MAC / U |
|---:|---|---:|---:|---:|---:|
| 50% | UNIFORM | 5 581 542 | 275 335 732 | 100% | 100% |
| 50% | T-proj | 5 577 102 | 275 222 472 | 99,920% | 99,959% |
| 50% | L06 | 5 576 517 | 275 293 400 | 99,910% | 99,985% |
| 30% | UNIFORM | 7 846 668 | 386 292 784 | 100% | 100% |
| 30% | T-proj | 7 840 401 | 386 044 980 | 99,920% | 99,936% |
| 30% | L06 | 7 845 652 | 385 996 368 | 99,987% | 99,923% |

P/MAC z rzeczywistego modelu zgadzają się z rachunkiem geometrii dla każdej
konstrukcji. Każdy profil mieści się w **[99,9%, 100%] obu** pułapów; przy
b30 wszystkie trzy są odrębne (RMS w `k/n`: L06–U 0,0750, L06–T-proj 0,0926,
T-proj–U 0,0322). Mapa grup, szerokości, niepuste klastry, bias FC i brak
gradientów sprawdzone; modele źródłowe niezmienione. Teacher: **11 220 132 P /
555 468 800 MAC**.

## Rezerwa — audyt indeksów i pochodzenia

`data/cifar100/splits.json` jest identyczny z kopią treningu E015:
40 000 train / 5 000 dev / 5 000 rezerwy, rozłączny podział oficjalnego
`train`. Przecięcie rezerwy z train, dev i listami A/B/C z E024: **po 0**.
Listy A i dev są identyczne w E035/E042; checkpointy `last.pt` (4/4 zgodne
SHA-256) pochodzą z końcowej epoki E015, bez wyboru najlepszego checkpointu.
Zgodnie z metadanymi L06 wybrano po ocenie na dev, a E035 użył osobnego
oficjalnego `test` (10 000 obrazów). Audyt dotyczy ról i rozłączności
**indeksów**; kolejność i hashe list zamrożono. CPU nie otwierał plików
obrazów ani etykiet; oczekiwany hash pliku oficjalnego `train` ustalono z
metadanych i sprawdzono po zgodzie przed deserializacją w fazie GPU.

## Ślad wykonania

- [Run CPU](../../runs/E043/cpu_20261121T230452Z_8b558297/manifest.json),
  **exit 0 / 281,19 s**, Conda `attention`; [zamrożona konfiguracja](../../runs/E043/cpu_20261121T230452Z_8b558297/frozen.json),
  SHA-256 `f84a40c4d4b435495904723aef61664ef11b359de4bee12ecc9160d6d81ded81`.
- Dwa wcześniejsze starty CPU zachowane: `cpu_20261121T230328Z_c0229221`
  (błędne pole w metadanych audytu E035) i `cpu_20261121T230420Z_fdd60cf5`
  (import helpera E042). Oba przerwane przed konstrukcją modeli; poprawki nie
  zmieniły protokołu ani reguły projekcji.
- Po osobnej zgodzie: `runs/E043/20261122T000341Z_9e23e96f`, 28/28 ocen,
  cztery GPU po jednym seedzie, 4×exit 0, audyt PASS. [Wyniki](WYNIKI.md).
