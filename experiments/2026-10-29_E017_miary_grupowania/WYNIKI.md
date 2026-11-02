# E017 — wyniki Etapu 1 (benchmark miar grupowania)

## Uruchomienie
- Run: `runs/E017/20261029T022532Z_cb9ffaf_f50f95ea`.
- **42/42 zadania completed**, cztery GPU 0–3, po jednym zadaniu na GPU.
- Nadzór: exit 0, 457,6 s. Analiza:
  `runs/E017/analizy/20261029T225633Z_bfd26c4d/`.
- Abort infrastrukturalny przed wynikami: `runs/E017/20261029T022222Z_cb9ffaf_7a314ac4`
  (przerwany nadzór po 120 s; zachowany, bez wyników naukowych).
- Brak treningu wag; test i rezerwa nietknięte.

## Wynik główny — dev top-1 po BN [%]

| Zbiór | Budżet kanałów | Reference (E012) | Random (średnia 3) | Cosine | Standardized | BN-effective |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | 10% | 94,18 | 92,54 | 93,66 | 94,30 | 93,14 |
| CIFAR-10 | 30% | 89,22 | 68,35 | 88,90 | 89,02 | 80,78 |
| CIFAR-10 | 50% | 70,48 | 28,79 | **78,62** | 70,62 | 46,98 |
| CIFAR-100 | 10% | 70,94 | 64,73 | 68,02 | 71,18 | 64,86 |
| CIFAR-100 | 30% | 50,60 | 25,33 | 46,66 | 51,28 | 32,96 |
| CIFAR-100 | 50% | 26,36 | 3,20 | 22,64 | 23,04 | 8,12 |

Rzeczywista redukcja **parametrów** (identyczna dla wszystkich miar):
10% kanałów → 19,3% parametrów; 30% → 51,1%; 50% → 75,0%.
Liczba parametrów po kompresji: 8 018 358 / 5 459 866 / 2 797 610
(CIFAR-10) i 9 059 848 / 5 492 176 / 2 820 740 (CIFAR-100).

## Różnice względem reference [pp]

| Zbiór | Budżet | Cosine | Standardized | BN-effective |
|---|---:|---:|---:|---:|
| CIFAR-10 | 10% | −0,52 | +0,12 | −1,04 |
| CIFAR-10 | 30% | −0,32 | −0,20 | −8,44 |
| CIFAR-10 | 50% | **+8,14** | +0,14 | −23,50 |
| CIFAR-100 | 10% | −2,92 | +0,24 | −6,08 |
| CIFAR-100 | 30% | −3,94 | +0,68 | −17,64 |
| CIFAR-100 | 50% | −3,72 | −3,32 | −18,24 |

## Decyzja według zamrożonego kryterium

**Żaden kandydat nie spełnia kryterium** (+0,50 pp vs reference i vs random
przy 30% i 50% na obu zbiorach). `E017_CONFIRMED=[]`.

- **Standardized** jest praktycznie równoważna reference (od −0,20 do +0,68 pp).
  Standaryzacja współrzędnych nie wnosi korzyści.
- **BN-effective** jest wyraźnie gorsza na obu zbiorach. Skala BN nie jest
  dobrą podstawą grupowania w tej postaci.
- **Cosine** przegrywa na CIFAR-100 i przy 30%, ale przy **50% na CIFAR-10**
  daje **+8,14 pp**. To obserwacja **post hoc** z siatki 18 komórek, pojedynczy
  seed — nie wynik potwierdzony.
- **Kontrola losowa** jest znacznie gorsza od reference (np. 28,79 vs 70,48
  przy 50% CIFAR-10), więc samo grupowanie ma duże znaczenie; nie jest to
  jednak test „czy cosine jest lepsza niż brak informacji”.

## Reprodukcja reference
Reference odtwarza E012 bit w bit na poziomie etykiet i modelu (kontrola CPU)
oraz liczbowo na dev: **94,18 / 89,22 / 70,48** vs E012
**94,20 / 89,30 / 70,50**. Różnica wynika z `preserve_fc_bias=True`
w E017; E012 używał `False` (oficjalny styl). Nie wpływa na porównania
między miarami, bo wszystkie używają tej samej polityki.

## Obserwacja mechanistyczna (hipoteza, nie dowód)
Przy 50% kanałów na CIFAR-10 reference ma grupę, w której **18,8%** kanałów
trafia do jednego klastra; cosine ma maksymalnie **7,8%**. Cosine daje
bardziej wyrównane klastry (średni udział największego klastra 3,0% vs 7,0%).
Możliwe, że reference nadmiernie scala kanały o dominującym kierunku.
To hipoteza do sprawdzenia, nie wyjaśnienie przyczyny.

## Ograniczenia
- Jeden seed42 na zbiór; brak przedziałów ufności i replikacji.
- Dev był używany wcześniej; to etap rozwojowy, nie niezależny test.
- Kryterium było zapisane przed biegiem, ale wybór „cosine przy 50% na
  CIFAR-10” jako tropu jest **post hoc** — wymaga replikacji na innych seedach.
- Nie porównujemy z oficjalnym kodem 1:1; to adaptacja operatora E012.

## Następny krok
Zamrożona replikacja hipotezy cosine: **seedy 43–45** (nie brały udziału
w odkryciu), CIFAR-10 przy 50% jako test główny; CIFAR-100 50% jako kontrola
specyficzności. Kryterium i protokół osobno, przed uruchomieniem.
