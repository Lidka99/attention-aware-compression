# 2026-11-02_E031_oficjalny_30_70

Status: **frozen**, 2026-11-02, przed oceną nowych modeli na dev.

## Cel i zakres

E029 sprawdził niezmieniony operator autorów CLFM na CIFAR-10/100 przy
kompresji 50%, a E030 na Tiny ImageNet64. E031 **domyka ten sam audyt dla
kompresji 30% i 70%** na istniejących checkpointach, bez treningu.

W tym protokole **b30 oznacza 30% redukcji parametrów (70% pozostaje)**,
a **b70 oznacza 70% redukcji (30% pozostaje)**. To ta sama konwencja co
w E014/E015/E019; nie mylić z frakcją zachowanych parametrów. Uzasadnienie:
przy 70% kompresji klasteryzacja jest bardziej niszcząca, więc zgodność
operatora i kierunek alokacji wymagają sprawdzenia także tam; przy 30%
efekt alokacji może być mały i musi być raportowany bez zaokrąglania do
„działa”.

E031 **nie** jest nową metodą, search, drugą architekturą ani pełną
reprodukcją artykułu. To audyt operatora i kierunku alokacji na naszych
wagach, naszym dev i wspólnej procedurze BN.

## Kod autorów i granica twierdzenia

Źródło: `osaukh/folding_as_projection`, commit
**694691556e8314a835178a741938dd3d4d80c171**, klon
`runs/E029/upstream_6946915`; pliki pozostają niezmienione, `git status`
i commit są sprawdzane przed runem. Używamy bezpośrednio
`ResNet18_ModelFolding.apply/compress_function` oraz
`utils.tune_utils.repair_bn`; nie kopiujemy ich matematyki. Brak opcjonalnego
`open_clip` obsługuje stub wyłącznie importu, którego konstruktor rzuca
wyjątkiem; kod CNN nie jest zastępowany.

Dla CIFAR-10/100 ładujemy state_dict do oryginalnej architektury autorów
(strict) i porównujemy logity z torchvision. Dla Tiny upstream nie ma klasy
modelu 64×64, więc używamy **niezmienionego operatora upstream na naszym
torchvision ResNet18** (standardowy stem 7×7+maxpool, fc512→200). To jawna
granica: „oficjalny operator na naszym modelu”, nie „oficjalna architektura
Tiny”. Mapa grup upstream odwołuje się wyłącznie do nazw
conv1/bn1/layerX/downsample/fc, zgodnych z torchvision.

## Wejścia

- Checkpointy: E011 (CIFAR-10), E015 (CIFAR-100), E019 (Tiny), seedy42–45,
  hashe sprawdzane strict. Bez publicznych wag i bez treningu.
- Dane: istniejące train/dev i podziały E011/E015/E019. A4096 do BN:
  CIFAR-10/100 crop32/padding4+flip (E025), Tiny RandomResizedCrop64
  scale(0.4,1.0)+flip (E030). Dev: 5000/5000/10000. **Oficjalny test
  i rezerwa nie są ładowane ani oceniane.**
- Indeksy A/dev: CIFAR-10 z E028, CIFAR-100 z E024, Tiny z E019
  (RNG20261103). A∩dev=∅; A nie należy do rezerwy. Hashe zapisane
  w `indices.json` runu.

## Profile i ramiona

Profile liczy bieżący solver E029 (`Geometry.profile`/`fill`), nie
historyczne specyfikacje E014/E015/E019. Dla kompresji b:
UNIFORM przy zachowaniu `round((1-b)·N)` parametrów; faktyczna liczba
parametrów UNIFORM staje się wspólnym targetem; TILT+0.30 dobierany do tego
targetu. Wymagania: dokładne szerokości per profil, TILT≤target i różnica
≤0,1%, 12 grup. Szerokości są zamrożone w `check.json` przed runem.
Uniform powinien być identyczny z historycznym; TILT może się różnić,
zwłaszcza przy b30 — to audyt bieżącej reguły, nie ponowne liczenie tabel
E014–E019. Nie interpretować ratio kanałów jako procentu globalnych
parametrów.

Siedem ramion na zbiór×budżet, identyczne szerokości per profil:
1. `UP_RAW_U` — niezmieniony upstream, uniform, losowy bias FC (seed42).
2. `UP_FIXED_U` — kopia z odtworzonym biasem FC checkpointu.
3. `LEGACY_U` — operator referencyjny E012/E014, te same szerokości.
4. `LOCAL_U` — samodzielny operator E024, te same szerokości.
5. `UP_FIXED_T` — upstream z `keep_ratio` zmienianym per grupa przed
   `compress_function`; to **nasz adapter TILT**, nie oficjalny wariant.
6. `LEGACY_T` — E012/E014 przy szerokościach T.
7. `LOCAL_T` — E024 przy szerokościach T.

Dla każdego ramienia: `PRE`, `CMA` (E025 A/aug/reset+CMA) i `UP_REPAIR`
(niezmienione `repair_bn`, retained EMA0.1) na tym samym A. Teacher:
`PRE`/`CMA`/`UP_REPAIR`, osobno per zbiór×seed. Razem
3 zbiory×2 budżety×4 seedy×7 ramion×3 stany = **504 oceny** +36 kontroli
teacher. Bez gradientów, bez zmian wag przez BN, bez checkpointów wyjściowych.

## Kryteria zamrożone

1. **Zgodność operatora:** dla każdego zbioru×budżetu×profilu (U,T)
   `LEGACY` i `LOCAL` kontra `UP_FIXED` po CMA: średnia |Δtop1|≤0,5 pp
   i maksimum≤1 pp po4 seedach. Razem 24 bramki.
2. **Kierunek alokacji:** dla każdego zbioru×budżetu
   `UP_FIXED_T−UP_FIXED_U` średnia>0 i ≥3/4 dodatnich seedów, **osobno**
   dla CMA i `UP_REPAIR`. Razem 12 bramek.
3. **Integralność:** strict load, 12 grup, dokładne parametry, niepuste
   klastry, skończone modele, brak gradientów, 4 GPU/exit0, limity
   90 min i 0,5 GiB, brak testu/rezerwy.
4. **Raportowanie wtórne:** bias FC (`UP_FIXED_U−UP_RAW_U`), różnica
   `UP_REPAIR−CMA`, ARI/logity/parametry, alokacja LEGACY/LOCAL, porównanie
   profili z historycznymi. Bez wybierania korzystnej procedury BN.
   Wynik negatywny zachowujemy.

**Decyzja:** jeśli 1 i 2 przechodzą na wszystkich trzech zbiorach i obu
budżetach, możemy napisać, że w tym zakresie zgodność operatora i kierunek
TILT utrzymują się również przy 30/70% kompresji. Jeśli którakolwiek bramka
nie przechodzi, raportujemy wynik negatywny i nie rozszerzamy twierdzenia;
nie zmieniamy progów po dev. Oczekiwane ograniczenia: przy b30 efekt może
być mały, a CIFAR-100 pozostaje słabszy.

## Czego E031 nie twierdzi

- Nie jest pełną reprodukcją CLFM 1:1: nasze checkpointy, dev zamiast
  oficjalnego testu, A4096 crop/flip bez RandomRotation15, Tiny na naszym
  modelu.
- Nie jest drugą architekturą i nie wymaga nowego treningu.
- Nie jest ponownym uruchomieniem tabel E014–E019 ani ich korektą;
  historyczne wyniki pozostają nietknięte.
- Nie dowodzi uniwersalności alokacji poza sprawdzonymi zbiorami,
  architekturą i budżetami.

## Wykonanie

Conda `attention`, 4×GPU0–3, seed42+i/GPUi (cztery niezależne seedy,
nie cztery repliki tego samego seeda), jeden launcher. Folding CPU,
ocena/BN GPU, batch128/GPU/przebieg, deterministic algorithms. Każdy run
ma unikalny katalog `runs/E031/<UTC>_<token>/` z manifestem, hashami,
`source/`, logami i PID. Bez nowych danych, 96×96, publicznych wag,
testu, rezerwy i konkurencyjnych zadań GPU.

## Odtworzenie

```
CUDA_VISIBLE_DEVICES='' python -B experiments/2026-11-02_E031_oficjalny_30_70/check.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B experiments/2026-11-02_E031_oficjalny_30_70/run.py
CUDA_VISIBLE_DEVICES='' python -B experiments/2026-11-02_E031_oficjalny_30_70/analyze.py --run-dir runs/E031/<run-ID>
```
