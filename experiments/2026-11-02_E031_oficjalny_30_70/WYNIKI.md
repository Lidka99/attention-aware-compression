# E031 — audyt oficjalnego operatora przy 30/70% kompresji

Status: **completed, wynik mieszany**, 2026-11-04. Protokół z 2026-11-02.
**Zgodność operatorów: 24/24 bramki. Kierunek alokacji: 11/12.**
Nie jest spełnione łączne kryterium rozszerzenia kierunku na wszystkie warunki:
CIFAR-100/b70 z oryginalnym `repair_bn` ma tylko 2/4 dodatnich seedów.

## Zakres i wykonanie

- [OPIS.md](OPIS.md) zamrożony przed pomiarami, SHA256
  `d338daff24156cca91e012f845c879b0d83cd103cff3a0f69ec2eed98f56536f`.
- b30/b70 oznaczają **30/70% redukcji parametrów**, czyli pozostaje 70/30%.
- Upstream `osaukh/folding_as_projection`, commit
  `694691556e8314a835178a741938dd3d4d80c171`, repo niezmienione.
  CIFAR: klasa autorów ze strict load naszych wag. Tiny64: niezmieniony
  operator autorów na **naszym torchvision ResNet18**, ponieważ upstream
  nie ma klasy Tiny64. TILT to nasz adapter ratio per grupa.
- Finalny run: `runs/E031/20261104T212951Z_6c3d7356`.
  Analiza: `analysis_20261104T213425Z/` (JSON i CSV, pełne tabele).
  Niezależna kontrola agregacji i reuse: `verification.json`, skrypt
  `audit/verify.py`.
- Finalny zbiór: **504 oceny +36 kontroli**, 24 komórki zbiór×budżet×seed.
  Wznowienie: **399 ocen z 19 kompletnych komórek +33 kontrole ponownie
  użyte**, **105 ocen z 5 brakujących komórek +3 kontrole nowe**.
  163 kopiowane pliki zweryfikowano SHA256 względem źródła i kopii.
- Kontynuacja: 4×RTX2080Ti, seedy42–45/GPU0–3, exit `[0,0,0,0]`,
  **209,85 s (3,50 min)**. Ten czas dotyczy wyłącznie kontynuacji,
  nie całego audytu wraz z przerwanymi próbami. Artefakty ok.8,35 MiB.
  Conda attention, Python3.11.15, Torch2.12.0+cu130, CUDA13.0,
  batch128/GPU; folding CPU, BN/ocena GPU.
- BN: wspólne A4096 z augmentacją, CMA/reset lub retained EMA0.1
  z niezmienionej funkcji `repair_bn`. Bez gradientów i treningu wag.
  Oficjalny test i rezerwa nietknięte.

## 1. TILT−UNIFORM na operatorze autorów

UP_FIXED zachowuje bias FC checkpointu. Średnia z4 seedów; SD dotyczy
sparowanych różnic. Kryterium: średnia>0 oraz co najmniej3/4 dodatnich.

| Zbiór | Redukcja | BN | UNIFORM [%] | TILT [%] | Różnica ± SD [pp] | Dodatnie |
|---|---:|---|---:|---:|---:|---:|
| CIFAR-10 | 30% | CMA | 93,495 | 94,520 | +1,025 ±0,191 | 4/4 |
| CIFAR-10 | 30% | repair_bn | 93,485 | 94,520 | +1,035 ±0,229 | 4/4 |
| CIFAR-10 | 70% | CMA | 77,305 | 88,215 | +10,910 ±2,116 | 4/4 |
| CIFAR-10 | 70% | repair_bn | 79,230 | 88,490 | +9,260 ±2,063 | 4/4 |
| CIFAR-100 | 30% | CMA | 66,340 | 66,960 | +0,620 ±0,722 | 3/4 |
| CIFAR-100 | 30% | repair_bn | 66,370 | 67,295 | +0,925 ±0,696 | 4/4 |
| CIFAR-100 | 70% | CMA | 31,400 | 33,020 | +1,620 ±2,412 | 3/4 |
| CIFAR-100 | 70% | repair_bn | 34,170 | 34,850 | **+0,680 ±2,629** | **2/4 — NIE** |
| Tiny | 30% | CMA | 34,808 | 38,588 | +3,780 ±1,094 | 4/4 |
| Tiny | 30% | repair_bn | 35,125 | 38,818 | +3,693 ±1,015 | 4/4 |
| Tiny | 70% | CMA | 8,433 | 16,180 | +7,7475 ±0,757 | 4/4 |
| Tiny | 70% | repair_bn | 9,443 | 16,778 | +7,335 ±0,725 | 4/4 |

**CIFAR-100/b70 per seed42/43/44/45:**
- CMA: **+0,04 / −0,10 / +1,46 / +5,08 pp** — bramka przechodzi, ale
  średnią silnie podnosi seed45.
- repair_bn: **−1,60 / −1,00 / +1,10 / +4,22 pp** — bramka nie przechodzi.

Na b70 repair_bn pomaga UNIFORM bardziej niż TILT: CIFAR-100 odpowiednio
+2,77 i+1,83 pp względem CMA; przewaga alokacji maleje o0,94 pp.
Nie jest to dowód błędu operatora: LEGACY/LOCAL odtwarzają tę samą różnicę.

## 2. Zgodność operatorów

LEGACY E012/E014: **identyczne wagi i partycje (ARI1)** względem UP_FIXED
we wszystkich48 porównaniach zbiór×budżet×seed×profil; top1 identyczny
przed BN i w obu procedurach BN. Zgodność wag nie oznacza bitowo identycznego
wykonania dwóch klas architektury CIFAR.

LOCAL E024: wszystkie bramki jakości CMA przechodzą, lecz nie ma
identyczności wag/partycji. Największe różnice top1:

| Zbiór | Maks. średnia \|Δtop1\| po4 seedach [pp] | Maks. pojedyncza \|Δtop1\| [pp] |
|---|---:|---:|
| CIFAR-10 | 0,035 | 0,140 |
| CIFAR-100 | 0,000 | 0,000 |
| Tiny | 0,265 | 0,340 |

Najniższe ARI LOCAL na Tiny:0,683; max różnica logitów CMA ok.4,316.
To zgodność empiryczna jakości, nie równoważność algebraiczna.
Łącznie **24/24** kryteria ≤0,5 pp średnio i≤1 pp maksymalnie spełnione.

## 3. Budżety i relacja do dawnych profili

| Zbiór | Redukcja | UNIFORM parametry | TILT parametry | TILT−U |
|---|---:|---:|---:|---:|
| CIFAR-10 | 30% | 7 808 058 | 7 804 213 | −3 845 |
| CIFAR-10 | 70% | 3 345 905 | 3 345 642 | −263 |
| CIFAR-100 | 30% | 7 846 668 | 7 841 653 | −5 015 |
| CIFAR-100 | 70% | 3 334 845 | 3 334 612 | −233 |
| Tiny | 30% | 7 845 193 | 7 840 865 | −4 328 |
| Tiny | 70% | 3 366 925 | 3 366 572 | −353 |

TILT zawsze pod targetem, rozbieżność≤0,064%. Wszystkie operatory dla
tego samego profilu mają dokładnie tę samą liczbę parametrów.
UNIFORM identyczny z historycznym; TILT liczy bieżący solver E029.
Przy30% różnice szerokości względem dawnych profili sięgają45/40/37
kanałów (CIFAR-10/100/Tiny), przy70% tylko2/1/2. To jawna różnica
zamrożona przed runem: **nie jest to ścisła replikacja tabel E014/E015/E019**.

## 4. Bias FC i kontrole teacher

Średni wpływ zachowania biasu FC po BN: od−0,060 do+0,095 pp we wszystkich
komórkach; mały względem głównych efektów. Nie przeprowadzano testu
istotności tego wpływu. Pełne wartości PRE/CMA/repair_bn w analizie.

Teacher PRE→CMA→repair_bn: CIFAR-10 **95,460→95,440→95,415%**;
CIFAR-100 **75,235→75,140→75,115%**; Tiny **50,820→51,220→51,243%**.

## 5. Przerwania i kontynuacja

- `20261102T032357Z_aaab1c44`: failed, zachowany. Błąd odwołania do
  `bn.BASELINE` po ukończeniu CIFAR-10; poprawiono lookup wartości teacher.
  Wyniki tego startu nie weszły do finalnej agregacji.
- `20261104T210033Z_4992f5fd`: przerwanie polecenia przez użytkownika;
  procesy nie działały przy wznowieniu. Historyczny manifest został
  zachowany ze statusem `running`, który nie oznacza aktywnego procesu.
  Odzyskano tylko19 kompletnych komórek i11 plików kontroli. Częściowe
  pomiary bez summary nie są użyte.
- Nowy run `20261104T212951Z_6c3d7356` zawiera kopie zweryfikowane hashami
  oraz brakujące pomiary. Zmieniono wyłącznie obsługę reuse, kompletność
  provenance i agregację kryteriów (PRE nie należy do12 zamrożonych bramek).
  Protokół, profile, operator, BN i progi nie zmieniły się.
- Kontrola CPU początkowo miała błędną asercję rozłączności indeksów Tiny
  z rezerwą: rezerwa indeksuje osobny cache_val, nie cache_train.
  Usunięcie tej asercji nie oznacza dostępu do rezerwy. Jeden kontrolny
  proces CPU przy wznowieniu przekroczył limit200 s; ponowiono z ograniczeniem
  wątków BLAS do2, zakończenie PASS.

Odtworzenie kontynuacji (Conda attention, katalog projektu):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B experiments/2026-11-02_E031_oficjalny_30_70/run.py --reuse-from runs/E031/20261104T210033Z_4992f5fd
```
Komenda tworzy kolejny unikalny run; bez argumentu uruchamia pełny audyt.

## Wniosek i dalsza decyzja

**Zgodność operatora została rozszerzona na30/70%.** TILT jest stabilnie
dodatni na CIFAR-10 i Tiny w obu procedurach BN. Na CIFAR-100 wynik jest
słabszy i zależy od seeda oraz BN; **nie wolno napisać, że wszystkie warunki
potwierdzają stabilny kierunek alokacji**. E015 pozostaje negatywny względem
swojego progu; wcześniejszych tabel nie zmieniono.

Jakość bezwzględna przy70% nadal mocno spada: CMA TILT33,02% na CIFAR-100
i16,18% na Tiny wobec teacher75,24% i50,82%. To przewaga względna,
nie dowód użytecznej jakości kompresji. Dev był wielokrotnie używany,
jedna rodzina architektury i jeden przepis checkpointów; brak pełnej
reprodukcji paperu1:1 i niezależnego testu.

Następny etap: koszt, przepis i kryteria drugiej architektury do uzgodnienia.
**Nowy trening wyłącznie po wyraźnej zgodzie użytkowniczki.** Nie uruchomiono
go; decyzja publikacyjna pozostaje otwarta.
