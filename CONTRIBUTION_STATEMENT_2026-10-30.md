# Contribution statement — 2026-10-30

Dokument do wysłania promotorom. Stanowi uczciwe podsumowanie tego, co mamy,
czego nie mamy i co proponujemy jako wkład. Nie jest deklaracją publikacji.

---

## Wersja polska

### Tytuł roboczy
**„Gdzie kompresować? Alokacja budżetu w foldingu bez fine-tuningu”**

### Problem

**Audyt literatury:** [literature/AUDYT_ALOKACJA_FOLDING.md](literature/AUDYT_ALOKACJA_FOLDING.md).
*Cut Less, Fold More* (Saukh i in., 2026) pokazuje, że folding jest lepszy od
pruningu i bada wpływ treningu, optymizatorów i augmentacji. W eksperymentach
stosuje jednak **jednolity współczynnik kompresji dla każdej warstwy**.
Co ważne, autorzy oryginalnego foldingu (Wang i in., 2025) sami piszą
w ograniczeniach: *„it does not optimize sparsity levels per layer, leaving
this for future work”*. Nasza praca adresuje to konkretne otwarte pytanie:
czy przy **tym samym globalnym budżecie parametrów** lepszy jest nierówny
rozdział kompresji między warstwy?

### Co pokazaliśmy
Przy dopasowanym budżecie parametrów, bez fine-tuningu, na trzech zbiorach
i czterech seedach:

| Zbiór | TILT+0.30 vs UNIFORM (30/50/70% parametrów) | Pary dodatnie |
|---|---|---|
| CIFAR-10 | **+0,87 / +3,77 / +11,27 pp** | 12/12 |
| Tiny ImageNet | **+3,71 / +10,32 / +7,60 pp** | 12/12 |
| CIFAR-100 | +0,81 / +1,96 / +1,60 pp | 11/12 |

TILT to **opracowana w tej pracy** prosta reguła depth-aware: mniej
kompresji wczesnych grup, więcej późnych, z solverem dobierającym profil
do wspólnego budżetu. Pierwsze użycie: E013 (2026-10-25).

### Kontrole negatywne
- Alternatywne miary grupowania (cosine, standaryzacja współrzędnych, skala BN)
  **nie poprawiają** referencji; cosine nie replikuje się na seedach 43–45.
- Alokacja oparta na błędzie rekonstrukcji wag (AUTO50) **przegrywa** z TILT.
- Losowe grupowanie jest znacznie gorsze — grupowanie ma znaczenie, ale nie
  znaleźliśmy lepszej miary.

### Proponowany wkład
1. **Pierwsze systematyczne badanie alokacji budżetu w calibration-free
   foldingu** — osi pominiętej w *Cut Less, Fold More*, która stosuje jednolite
   ratio per warstwa.
2. **Prosta, powtarzalna reguła opracowana w tej pracy** (TILT) dająca duże zyski przy dopasowanym
   budżecie, największe przy średniej/wysokiej kompresji.
3. **Rozdzielenie dwóch dźwigni:** alokacja budżetu ma większe znaczenie niż
   dobór miary podobieństwa kanałów; standardowe miary i alokacja oparta na
   wrażliwości nie wystarczają.
4. **Mechanistyczne wyjaśnienie** (w toku): dlaczego wczesne grupy są wrażliwsze
   i ile szkody odzyskuje korekta BN.

### Czego nie twierdzimy
- Że jesteśmy lepsi od *Cut Less, Fold More* „1:1” — to adaptacja ich operatora,
  bez uruchomienia ich oficjalnego kodu.
- Że metoda działa uniwersalnie — CIFAR-100 jest słaby, a przy 50–70% redukcji
  jakość bezwzględna mocno spada.
- Że jesteśmy w pełni „data-free” — korekta BN używa nieetykietowanego train.
- Że TILT+0.30 jest optymalny — to zamrożona hipoteza, nie wynik strojenia.

### Plan domknięcia
1. Mechanizm na istniejących checkpointach (bez treningu).
2. Łagodne budżety 10/20% — reżim użyteczny aplikacyjnie.
3. Oficjalny kod *Cut Less, Fold More* na naszych checkpointach.
4. Druga architektura (np. PreActResNet18).
5. Decyzja o publikacji po 1–4.

### Pytanie do promotorów
Czy tak zdefiniowany wkład (alokacja, nie miara) jest akceptowalny jako rdzeń
pracy doktorskiej, czy oczekują jednak nowej miary podobieństwa jako głównego
wkładu?

---

## English version

### Working title
**“Where to Fold? Budget Allocation in Calibration-Free Model Folding”**

### Problem

**Literature audit:** [literature/AUDYT_ALOKACJA_FOLDING.md](literature/AUDYT_ALOKACJA_FOLDING.md).
*Cut Less, Fold More* (Saukh et al., 2026) shows that folding outperforms
pruning and studies the role of training, optimizers and augmentation. Its
experiments use a **uniform per-layer compression ratio**. Notably, the
original folding paper (Wang et al., 2025) states in its limitations:
*“it does not optimize sparsity levels per layer, leaving this for future
work”*. Our work addresses this specific open question: whether, at the
**same global parameter budget**, a non-uniform allocation across layers is
better.

### What we found
At matched parameter budgets, without fine-tuning, on three datasets and four
seeds:

| Dataset | TILT+0.30 vs UNIFORM (30/50/70% parameters) | Positive pairs |
|---|---|---|
| CIFAR-10 | **+0.87 / +3.77 / +11.27 pp** | 12/12 |
| Tiny ImageNet | **+3.71 / +10.32 / +7.60 pp** | 12/12 |
| CIFAR-100 | +0.81 / +1.96 / +1.60 pp | 11/12 |

TILT is a simple depth-aware rule **developed in this work**: less compression
in early groups, more in later ones, with a solver matching a common parameter
budget. First used in E013 (2026-10-25).

### Negative controls
- Alternative grouping measures (cosine, feature standardization, BN scale) do
  **not** improve over the reference; cosine does not replicate on seeds 43–45.
- Allocation from weight-reconstruction error (AUTO50) **loses** to TILT.
- Random grouping is far worse — grouping matters, but we found no better
  measure.

### Proposed contribution
1. **First systematic study of budget allocation in calibration-free folding** —
   an axis omitted by *Cut Less, Fold More*, which uses uniform per-layer ratios.
2. **A simple, reproducible rule developed in this work** (TILT) with large gains at matched budgets,
   strongest at moderate-to-high compression.
3. **Separating two levers:** where to spend the budget matters more than which
   channel-similarity measure is used; standard measures and sensitivity-based
   allocation are insufficient.
4. **Mechanistic explanation** (ongoing): why early groups are more sensitive and
   how much damage BatchNorm repair recovers.

### What we do not claim
- That we beat *Cut Less, Fold More* “1:1” — this is an adaptation of their
  operator, without running their official code.
- That the method is universal — CIFAR-100 is weak, and absolute accuracy drops
  sharply at 50–70% parameter reduction.
- That the method is fully data-free — BN repair uses unlabeled training data.
- That TILT+0.30 is optimal — it is a frozen hypothesis, not a tuned result.

### Completion plan
1. Mechanism analysis on existing checkpoints (no training).
2. Mild budgets (10/20%) — the practically useful regime.
3. Official *Cut Less, Fold More* code on our checkpoints.
4. A second architecture (e.g., PreActResNet18).
5. Publication decision after 1–4.

### Question for supervisors
Is this contribution (allocation, not similarity measure) acceptable as the core
of the thesis, or is a new similarity measure still expected as the main
contribution?

---

## Krótka wersja do maila (PL)

> Sprawdziliśmy wskazany kierunek: alternatywne miary grupowania kanałów nie
> poprawiają foldingu — to dobrze skontrolowany wynik negatywny. Znaleźliśmy
> ortogonalną dźwignię: nierówną alokację budżetu między warstwy. Przy
> dopasowanym budżecie parametrów daje ona +0,87/+3,77/+11,27 pp na CIFAR-10,
> +3,71/+10,32/+7,60 pp na Tiny ImageNet i +0,81/+1,96/+1,60 pp na CIFAR-100
> (4 seedy). Chcemy domknąć to mechanizmem, porównaniem z oficjalnym kodem
> i drugą architekturą. Proszę o opinię, czy tak zdefiniowany wkład jest
> akceptowalny jako rdzeń pracy.

## Short email version (EN)

> We tested the suggested direction: alternative channel-similarity measures do
> not improve folding — a well-controlled negative result. We found an
> orthogonal lever: non-uniform budget allocation across layers. At matched
> parameter budgets it yields +0.87/+3.77/+11.27 pp on CIFAR-10,
> +3.71/+10.32/+7.60 pp on Tiny ImageNet, and +0.81/+1.96/+1.60 pp on
> CIFAR-100 (four seeds). We plan to close this with a mechanism analysis, the
> official baseline, and a second architecture. Could you advise whether this
> contribution is acceptable as the core of the thesis?
