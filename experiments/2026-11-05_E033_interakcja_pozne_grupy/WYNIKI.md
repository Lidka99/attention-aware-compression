# E033 — prosta hipoteza późnych grup nie została potwierdzona

Status: **completed, primary negatywny, secondary dodatni**, 2026-11-05.
**Zgodność operatora 24/24.** Usunięcie dodatkowej kompresji późnych grup
(profil EP) **nie poprawiło** CIFAR-100/b70 po CMA względem TILT: +0,135 pp,
2/4 seedów. Po `repair_bn` EP był lepszy o+0,95 pp (3/4), ale wynik nie jest
spójny między procedurami BN. Na CIFAR-10 i Tiny EP był znacznie gorszy
od TILT. Prosta hipoteza „CIFAR-100 cierpi głównie przez nadmierną kompresję
późnych grup” **nie zyskała wsparcia** w tym teście.

## Wykonanie i integralność

- [OPIS.md](OPIS.md) zamrożony przed runem, SHA256 w `check.json` i manifeście.
- Finalny run: `runs/E033/20261105T010139Z_f193f1d4`.
  Analiza: `analysis_20261105T011101Z/` (`analysis.json`, `metrics.csv`,
  `mechanism.csv`, `quality.csv`). Kontrola niezależna: `verification.json`,
  skrypt `audit/verify.py`.
- **432 oceny +36 kontroli teacher**, 3 zbiory×2 budżety×4seedy×6ramion×3stany.
  4GPU/seedy42–45/GPU0–3, exit `[0,0,0,0]`, **539,07 s (8,98 min)**.
  Artefakty ok.0,68 MiB; limity90min/0,5GiB zachowane.
- Conda attention, Python3.11.15, Torch2.12.0+cu130, CUDA13.0,
  4×RTX2080Ti, batch128/GPU. Folding CPU, BN i ocena GPU.
  Bez gradientów, treningu, nowych danych, testu i rezerwy.
- Operator upstream commit `694691556e8314a835178a741938dd3d4d80c171`
  niezmieniony. CIFAR: klasa autorów ze strict load naszych wag.
  Tiny64: niezmieniony operator autorów na naszym torchvision ResNet18.
  Profile EP/LC to nasze adaptacje `keep_ratio` per grupa, nie oficjalne
  warianty autorów.
- Kontrola `LEGACY_EP/LC` vs `UP_EP/LC`: **parametry max_abs=0, ARI=1,
  top1 identyczny** we wszystkich96 seed-checkach; max różnica logitów
  CMA/repair_bn ≈3,8×10⁻⁶. Bramki jakości **24/24**.
- Ramiona `UP_U`/`UP_T` odtwarzają metryki E031 **identycznie w24/24
  komórkach** (top1 do1e-12); nowe efekty nie pochodzą z dryfu baseline.
  Kontrole teacher zgodne z E031 na wszystkich24 parach.
- Jeden nieudany start `20261105T004647Z_83d7735f` jest zachowany:
  `AttributeError` w obsłudze diagnostyki LEGACY (iterowano klucze dict
  zamiast `diag['groups']`). Nie zapisano żadnego summary; brak wyników
  naukowych. Poprawka wyłącznie w kodzie uruchomienia, przed finalnym runem.

## Profile

`U` i `T` są dokładnie profilami E031. `EP`: grupy9–11 (`layer4`) mają
szerokości UNIFORM, grupy0–8 kształt TILT+0.30 z `base` dobranym do
wspólnego targetu. `LC`: grupy0–8 mają UNIFORM, grupy9–11 kształt
TILT+0.30 z `base` dobranym do targetu. Wszystkie profile≤target,
różnice≤0,1%.

CIFAR-100/b70 (30% zachowanych parametrów), szerokości12 grup:

| Profil | Szerokości | Parametry |
|---|---|---:|
| U | 34,34,34,69,69,69,139,139,139,279,279,279 | 3 334 845 |
| T | 57,57,57,109,102,95,177,163,149,272,243,215 | 3 334 612 |
| EP | 55,52,48,90,83,76,139,125,111,279,279,279 | 3 334 564 |
| LC | 34,34,34,69,69,69,139,139,139,309,281,253 | 3 332 301 |

**Ważne ograniczenie projektowe:** przy dopasowanym globalnym budżecie
nie da się zmienić tylko późnych grup. W EP grupy6–8 są bardziej
skompresowane niż w TILT (139/125/111 vs177/163/149), bo trzeba zapłacić
za szerokie `layer4` jak w UNIFORM. W LC region wczesny/środkowy jest
UNIFORM, więc budżet na późne grupy wypada większy niż w TILT
(309/281/253 vs272/243/215). To nie są czyste jednoczynnikowe interwencje;
EP jest jednak bezpośrednim testem „usuń dodatkową kompresję późnych grup
przy tym samym budżecie”.

## 1. Primary: CIFAR-100/b70

Różnice średnie po4 seedach [pp]; EP−T jest primary, LC−T eksploracyjne.

| Kontrast | CMA | Dodatnie | repair_bn | Dodatnie |
|---|---:|---:|---:|---:|
| EP−T | **+0,135 ±1,556** | **2/4** | **+0,950 ±1,536** | **3/4** |
| LC−T | −1,855 ±2,614 | 2/4 | −1,395 ±2,427 | 2/4 |
| EP−U | +1,755 ±1,792 | 3/4 | +1,630 ±1,277 | 4/4 |
| LC−U | −0,235 ±0,862 | 3/4 | −0,715 ±0,475 | 0/4 |

EP−T per seed42/43/44/45:
- CMA: **+1,08 / +1,80 / −1,48 / −0,86 pp**;
- repair_bn: **+2,16 / +2,34 / +0,04 / −0,74 pp**.

**Primary niespełniony:** średnia>0, ale tylko2/4 dodatnich po CMA.
Secondary repair_bn przechodzi3/4, lecz nie jest niezależnym potwierdzeniem:
to ta sama interwencja i ten sam dev, inna procedura BN. Nie wybieramy
korzystniejszej procedury jako werdyktu.

Pełne średnie CIFAR-100/b70:

| Profil | CMA | repair_bn |
|---|---:|---:|
| U | 31,400 | 34,170 |
| T | 33,020 | 34,850 |
| EP | **33,155** | **35,800** |
| LC | 31,165 | 33,455 |

EP jest lepszy od UNIFORM, ale **nie od TILT** po CMA. Po repair_bn EP
wyprzedza T o0,95 pp, jednak przewaga nad U rośnie tylko do1,63 pp.

## 2. Pozostałe zbiory i budżety

CIFAR-100/b30: EP−T **+0,115 CMA / −0,075 repair_bn**, 2/4 oba;
brak efektu. LC−T −0,425/−0,500, 1/4.

CIFAR-10/b70 CMA: EP−T **−7,135**, 0/4; LC−T **−11,120**, 0/4.
EP wciąż bije U o+3,775 (4/4), ale traci większość zysku TILT.
Tiny/b70 CMA: EP−T **−5,065**, 0/4; LC−T **−7,632**, 0/4;
EP−U +2,682 (4/4). Na b30 kierunek ten sam, mniejszy.

To pokazuje, że na CIFAR-10/Tiny późna kompresja jest częścią korzystnego
profilu, a nie prostym kosztem. Na CIFAR-100 usunięcie jej nie odzyskuje
przewagi TILT; możliwe, że koszt przesunięcia budżetu do grup środkowych
niweluje korzyść z ochrony `layer4`. Ten test nie rozstrzyga, która
z tych dwóch zmian dominuje.

## 3. Co to znaczy dla mechanizmu z E032

E032 pokazał, że późne grupy CIFAR-100 są wrażliwe po BN. E033 pokazuje,
że **sama ta wrażliwość nie wystarcza do wyjaśnienia słabego zysku TILT**:
przy dopasowanym budżecie usunięcie dodatkowej kompresji późnych grup
nie dało stabilnej poprawy. Nie potwierdzono więc prostej narracji
„TILT szkodzi CIFAR-100, bo dociska layer4”. Nie znaczy to, że późna
kompresja jest optymalna ani że należy ją zwiększać; ANTI-TILT i wcześniejsze
kontrole pozostają negatywne. Mechanizm jest najpewniej interakcją
kilku grup i procedury BN, a nie efektem jednego regionu.

## Kryteria

| Warunek | Wynik |
|---|---|
| Zgodność LEGACY/UP dla EP/LC, CMA+repair_bn,24bramki | **TAK,24/24** |
| Primary EP−T CIFAR-100/b70/CMA,>0 i≥3/4 | **NIE,2/4** |
| Secondary EP−T CIFAR-100/b70/repair_bn,>0 i≥3/4 | TAK,3/4 |
| Integralność,4GPU/exit0, limity, test/rezerwa | TAK |
| LC−T | eksploracyjne, bez progu; ujemne na CIFAR-100 |

## Ograniczenia i decyzja

- Profile hybrydowe są sprzężone budżetem; EP zmienia też grupy6–8,
  a LC nie jest czystym „tylko późna kompresja”. Wynik nie izoluje
  pojedynczej grupy ani nie jest addytywną dekompozycją.
- Tylko4seedy, używany wcześniej dev, jedna rodzina architektury,
  Tiny z innym stem. Brak niezależnego testu i pełnej reprodukcji paperu.
- **Nie budować na tej podstawie nowego alokatora „chroń layer4”.**
  Prosta hipoteza późnych grup została zamknięta jako niewystarczająca.
  Nie wracać do search, AUTO50 ani miar grupowania.
- Następny etap to koszt, przepis i kryteria drugiej architektury
  (np. PreActResNet18); **nowy trening wyłącznie po wyraźnej zgodzie**.
  Decyzja publikacyjna pozostaje otwarta.
