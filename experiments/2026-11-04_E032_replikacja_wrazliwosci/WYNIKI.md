# E032 — późna wrażliwość CIFAR-100 replikuje się, różnice zysku TILT niewyjaśnione

Status: **completed**, raport2026-11-05. **H1 i H2 spełnione.**
Późna wrażliwość po BN występuje na wszystkich4 checkpointach CIFAR-100,
w przeciwieństwie do CIFAR-10. Nie jest jednak dobrym wyjaśnieniem
różnic korzyści TILT między seedami CIFAR-100: profile są bardzo podobne,
a ich kontrast późne−wczesne nie ma oczekiwanego ujemnego związku z zyskiem.

## Wykonanie i integralność

- [OPIS.md](OPIS.md) zamrożony przed runem; hash w check.json i manifeście.
- Run: `runs/E032/20261104T235711Z_fc9d5eaa`.
- Analiza: `analysis_20261105T000640Z/`:
  `analysis.json`, `groups.csv`, `profiles.csv`, `stability.csv`, `TABLES.md`.
  Kontrola niezależna: `verification.json`, skrypt `audit/verify.py`.
- **96 modeli pojedynczej grupy×3stany =288 ocen +24 nowe kontrole teacher**,
  wszystkie ukończone.4GPU/seedy42–45, exit `[0,0,0,0]`,149,79s (2,50min).
  Artefakty przed analizą ok.9,38MiB; limit0,5GiB i90min zachowany.
- Conda attention,4×RTX2080Ti, batch128/GPU/przebieg. Folding CPU,
  BN i ocena GPU. Manifest zawiera PID, wersje pakietów, przypisanie GPU,
  SHA256 kodu/protokołu/danych/checkpointów; analiza hashuje104 pliki
  pomiarów i kontroli. Weryfikator niezależnie przelicza rangi i H1/H2.
- Dane i A4096/crop/flip jak E031; teacher PRE/CMA/repair_bn odtworzył
  top1 E031 na wszystkich24 kontrolach. Szkoda liczona względem teacher
  po **tej samej** procedurze BN. Bez testu, rezerwy, nowych danych i treningu.
- Operator LEGACY E012/E014. Pozostałe grupy mają etykiety tożsamościowe.
  Oryginalne `repair_bn` z commit
  `694691556e8314a835178a741938dd3d4d80c171`, upstream niezmieniony;
  modele we wszystkich ramionach to **nasza klasa torchvision CIFAR ResNet18**.
  To nie nowy audyt oficjalnego operatora na pojedynczych grupach.
- Pierwsza kontrola CPU wykryła zbyt silne założenie o identyczności
  wszystkich buforów przy ratio0: LEGACY odbudowuje BN i zeruje
  `num_batches_tracked`. Przed runem doprecyzowano protokół i kontrolę:
  parametry, running mean/var i logity identyczne, liczniki wyłączone
  z porównania. Nie wpływają na eval, CMA/reset ani retained EMA0.1.
  Po korekcie CPU PASS; jeden run GPU, bez nieudanych runów GPU.

## 1. Primary: czy późna wrażliwość jest powtarzalna?

W każdym pomiarze **75% redukcji kanałów tylko jednej grupy**, reszta
modelu bez kompresji. To inny reżim niż70% globalnej redukcji w E031.
Wczesne grupy=0–2, późne=9–11. H1: rho≥0,5 i L−E>0 na≥3/4 seedach
CIFAR-100 po CMA. L−E to różnica średnich szkód późnych i wczesnych grup.

| Seed | CIFAR-100 rho(CMA) | CIFAR-100 L−E [pp] | H1 | CIFAR-10 rho(CMA) | CIFAR-10 L−E [pp] |
|---:|---:|---:|---|---:|---:|
| 42 | 0,629 | +16,480 | TAK | 0,000 | −1,153 |
| 43 | 0,622 | +17,407 | TAK | 0,077 | −0,027 |
| 44 | 0,629 | +17,540 | TAK | −0,049 | −1,480 |
| 45 | 0,629 | +17,220 | TAK | −0,224 | −2,340 |

**H1:4/4, w tym3/3 nowych seedów43–45.** Seed42 mierzony na nowej
kalibracji, dlatego nie kopiowano jego wcześniejszego wyniku E022.
**H2: spełnione** — kontrast rho CIFAR-100−CIFAR-10 średnio **+0,677**
(próg≥0,3), dodatni na4/4 seedach (próg≥3/4).

Wynik wtórny `repair_bn`: rho CIFAR-100 **0,503/0,601/0,531/0,531**,
L−E **+13,58/+17,60/+15,84/+14,31pp**. Kierunek utrzymuje się4/4.
Na CIFAR-10 rho **0,000/0,077/−0,084/−0,259**, L−E ujemne4/4.
To progi opisowe, nie test istotności statystycznej.

## 2. Co zostaje po naprawie BN?

Średnie szkody względem odpowiedniej kontroli teacher, po4 seedach:

| Zbiór | BN | Wczesne grupy [pp] | Późne grupy [pp] | Wszystkie12 grup [pp] |
|---|---|---:|---:|---:|
| CIFAR-100 | PRE | 31,528 | 28,888 | 30,032 |
| CIFAR-100 | CMA | **3,482** | **20,643** | 9,617 |
| CIFAR-100 | repair_bn | 5,318 | 20,652 | 10,401 |
| CIFAR-10 | PRE | 15,073 | 1,567 | 13,852 |
| CIFAR-10 | CMA | **2,213** | **0,963** | 2,869 |
| CIFAR-10 | repair_bn | 2,243 | 0,927 | 2,931 |

Na CIFAR-100 przed BN wczesne grupy nie są mniej szkodliwe od późnych.
Dopiero **nierówny odzysk po BN** odsłania silny koszt późnych grup:
po CMA średnia szkoda wczesnych spada o28,047pp, późnych tylko o8,245pp
(różnica szkód kontrolowanych, teacher również ma korektę BN).
Nie jest to stała „wrażliwość na głębokość” niezależna od procedury BN.

## 3. Które grupy są krytyczne?

Szkoda po CMA, średnia±SD po4 seedach; kompresja75% kanałów grupy:

| Grupa | CIFAR-100 [pp] | CIFAR-10 [pp] |
|---|---:|---:|
| `relu_conv` | 9,325±1,378 | 5,965±1,601 |
| `layer1.0.relu1` | 0,585±0,241 | 0,405±0,261 |
| `layer1.1.relu1` | 0,535±0,560 | 0,270±0,225 |
| `layer2.0.relu1` | 2,385±0,549 | 1,385±0,298 |
| `layer2.0.relu2` | 10,605±2,849 | 7,195±1,265 |
| `layer2.1.relu1` | 1,500±0,190 | 0,770±0,205 |
| `layer3.0.relu1` | 4,830±0,464 | 3,135±0,501 |
| `layer3.0.relu2` | 22,210±0,913 | 11,025±0,711 |
| `layer3.1.relu1` | 1,495±0,511 | 1,390±0,362 |
| `layer4.0.relu1` | 11,095±0,391 | 0,965±0,347 |
| `layer4.0.relu2` | **30,355±1,314** | **1,460±0,586** |
| `layer4.1.relu1` | **20,480±1,480** | **0,465±0,246** |

To grupy strukturalne uzgodnionych osi kanałów, nie12 niezależnych
pojedynczych warstw. Kompresja grupy może dotykać kilku modułów.

## 4. Czy to wyjaśnia wyjątkowo dobry seed45 w E031?

**Nie w badanej prostej postaci.** Ranking szkód CIFAR-100 jest bardzo
stabilny między checkpointami: korelacje wszystkich6 par seedów po CMA
**0,958–0,986**, po repair_bn **0,965–0,993**. CIFAR-10 także stabilny,
ale nieco słabiej: CMA0,846–0,951.

| CIFAR-100 seed | L−E CMA [pp] | Zysk TILT E031/b70 CMA [pp] |
|---:|---:|---:|
| 42 | 16,480 | +0,04 |
| 43 | 17,407 | −0,10 |
| 44 | 17,540 | +1,46 |
| 45 | 17,220 | +5,08 |

Spearman L−E vs zysk TILT (4punkty): CIFAR-100 **0,00 CMA /+0,20 repair_bn**.
Nie ma przewidywanego ujemnego związku. Seed45 nie ma wyraźnie mniejszej
późnej wrażliwości, mimo większego zysku TILT. CIFAR-10 daje−0,80/−1,00,
ale przy4 punktach nie przenosimy tego wyniku na CIFAR-100 ani populację.

## Wniosek i decyzja

1. **Nowy potwierdzony wynik:** wysoka szkoda późnych grup CIFAR-100 po BN
   jest powtarzalna, nie jest przypadkiem seed42. To wzmacnia hipotezę
   konfliktu z regułą TILT zwiększającą kompresję późnych grup.
2. **Granica wyjaśnienia:** profil przy75% kompresji pojedynczej grupy
   nie wyjaśnia rozrzutu zysku TILT między seedami. Interakcje grup i inne
   poziomy kompresji pozostają kandydatami, ale nie zostały tu zmierzone.
   Nie twierdzimy, że wykazano przyczynę różnicy wyników całych sieci.
3. Nie dowiedziono, że ochrona późnych grup poprawi globalną alokację;
   negatywne wyniki ANTI-TILT i AUTO50 nadal obowiązują. E015 i E031
   zachowują swoje wcześniejsze werdykty.
4. Ograniczenia:4seedy, używany wcześniej dev, duża pojedyncza perturbacja,
   różne koszty parametrów grup, jedna rodzina architektury. H2 nie izoluje
   wpływu danych od liczby klas i przepisu treningu. Brak nowych modeli/Tiny.

**Następna decyzja:** jeśli dalej wyjaśniamy CIFAR-100, potrzebny jest
osobny protokół interakcji przy rzeczywistych profilach E031 zamiast
kolejnej replikacji tego samego wskaźnika. Niczego takiego nie uruchomiono.
Plan drugiej architektury wymaga kosztorysu i wyraźnej zgody na trening;
search pozostaje zamknięty. Decyzja publikacyjna nadal otwarta.
