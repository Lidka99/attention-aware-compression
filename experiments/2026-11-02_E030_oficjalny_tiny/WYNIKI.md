# E030 — oficjalny operator na Tiny ImageNet
Status: **completed, kryteria spełnione**, 2026-11-02.
Trzeci zbiór potwierdza: nasza referencja odtwarza operator autorów,
a alokacja TILT poprawia wynik również na nim i przy ich `repair_bn`.

## Granica twierdzenia
Upstream nie udostępnia klasy ResNet18 dla Tiny64 (jego klasa to CIFAR
3×3/no-maxpool, `avg_pool2d(4)`). Dlatego użyto **niezmienionego operatora**
`ResNet18_ModelFolding` na **naszym** torchvision ResNet18 ze standardowym
stem7×7+maxpool, fc512→200. Mapa grup upstream odwołuje się tylko do nazw
`conv1/bn1/layerX/downsample/fc`, więc pasuje do torchvision. To „oficjalny
operator na naszym modelu”, **nie** „oficjalna architektura Tiny”.
CLIP stub jak w E029; matematyka CNN bez zmian.

## Wykonanie
- [OPIS.md](OPIS.md) frozen przed runem; commit upstream
  `694691556e8314a835178a741938dd3d4d80c171`, repo czyste.
- Run: `runs/E030/20261102T013740Z_609b7737`; analiza
  `analysis_20261102T014235Z/`.
- 4/4 seedy, GPU0–3, **84 oceny +12 kontroli teacher**, exit `[0,0,0,0]`.
- **119,55s (1,99min)**, artefakty <1MiB; Conda attention, batch128/GPU.
- Budżet≈50%: target UNIFORM **5 623 142**, TILT 5 622 878 (pod targetem,
  różnica0,005%). 12 grup, dokładne parametry per ramię.
- BN: A4096 z train, deterministyczny RandomResizedCrop64 scale(0.4,1.0)
  +flip (RNG20261103+indeks), normalizacja Tiny; dev10k z pliku.
  Rezerwa i oficjalny val/test nietknięte. Bez treningu i gradientów.
- Dwa przerwane starty zachowane jako infrastruktura (asercja progu
  hash-a augmentacji, potem brakujący helper dev); **żadnych ocen modeli
  w nich nie wykonano**. Poprawki dotyczyły wyłącznie kodu uruchomienia.

## 1. Zgodność z upstream (CMA)
| Implementacja | mean\|Δtop1\| | max\|Δtop1\| | Min ARI | Max różnica wag | Bramka |
|---|---:|---:|---:|---:|---|
| LEGACY U | 0,000 | 0,000 | 1,000 | 0,000 | TAK |
| LOCAL U | 0,165 | 0,360 | 0,657 | 0,264 | TAK |
| LEGACY T | 0,000 | 0,000 | 1,000 | 0,000 | TAK |
| LOCAL T | 0,110 | 0,270 | 0,762 | 0,261 | TAK |

**LEGACY E012/E014 jest identyczny z upstream także na Tiny** (wagi,
partycje, logity; max_abs≤2×10⁻⁶). **LOCAL E024 nie jest identyczny**:
zgodność predykcji jest bardzo wysoka, ale partycje różnią się częściowo,
a logity do ok.3,9. Przyczyna nie jest rozstrzygnięta; kandydat to inna
kolejność kolumn cech przy PCA, co zmienia wynik HKMeans. Nie nazywamy
LOCAL „bitowo równoważnym”. **Bramka jakości przechodzi** (średnio≤0,5pp,
maks.≤1pp), ale jest to zgodność empiryczna, nie algebraiczna.
Na CIFAR-10/100 LOCAL był praktycznie identyczny (E029); różnica jest
zależna od architektury/wejścia, więc raportujemy ją jawnie.

## 2. TILT na operatorze autorów
| Operator | BN | UNIFORM [%] | TILT [%] | TILT−UNIFORM [pp] | Dodatnie |
|---|---|---:|---:|---:|---:|
| UP_FIXED | CMA/reset | 19,925±1,804 | 29,333±1,285 | **+9,408±0,603** | **4/4** |
| UP_FIXED | oryginalne repair_bn | 20,538±1,637 | 29,715±1,225 | **+9,178±0,498** | **4/4** |
| LEGACY | CMA/reset | 19,925±1,804 | 29,333±1,285 | +9,408±0,603 | 4/4 |
| LOCAL | CMA/reset | 19,775±1,734 | 29,443±1,325 | +9,668±0,579 | 4/4 |

To trzeci zbiór z dodatnim kierunkiem na operatorze autorów. Wynik jest
zgodny z wcześniejszym E019 (+10,32pp przy50% na naszym operatorze),
choć nie jest jego repliką: inna procedura BN i inny A.

## 3. Bias FC i BN
Zachowanie biasu FC: `UP_FIXED−UP_RAW` = **−0,025pp (CMA)**, −0,025pp
(pre), 0,000pp (repair_bn); efekt nieistotny, nie tłumaczy przewagi.
Teacher: pre50,82%, CMA51,22%, repair_bn51,24%.

## Kryteria
| Warunek | Wynik |
|---|---|
| LEGACY/LOCAL vs UP_FIXED, CMA: ≤0,5pp średnio i≤1pp maks. | TAK,4/4 |
| UP_FIXED_T−UP_FIXED_U>0 i≥3/4, CMA i repair_bn | TAK,4/4+4/4 |
| Integralność: strict load,12 grup, dokładne parametry, 4GPU/exit0 | TAK |
| Limity90min/0,5GiB | TAK |

## Ograniczenia i wniosek
Jeden budżet, jedna architektura Tiny (inny stem niż CIFAR), jeden seed
augmentacji, dev używany wcześniej. Nie jest to reprodukcja paperu1:1.
**Wniosek:** po CIFAR-10 i CIFAR-100 również Tiny potwierdza, że
przewaga alokacji nie wynika ze słabszego operatora baseline ani
z losowego biasu FC; utrzymuje się przy oryginalnej funkcji BN autorów.
Pełny obraz (TILT−UNIFORM na operatorze upstream, CMA): **+4,00/+1,70/+9,41pp**
dla CIFAR-10/100/Tiny przy~50%.
Następny krok pozostaje publikacyjny: druga architektura wymaga nowego
treningu i zgody użytkowniczki.
