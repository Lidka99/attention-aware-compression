# Audyt: czy alokacja budżetu per warstwa w foldingu jest zbadana?

Data: 2026-10-31. Autor: projekt badawczy (asysta AI).
Cel: sprawdzić, czy „nierówna alokacja budżetu między warstwy w calibration-free
foldingu” jest realną luką, czy tylko naszym wrażeniem.

## Metoda

1. Pełny tekst *Cut Less, Fold More* (arXiv 2602.18116) — eksperymenty,
   podpisy rysunków, tabele, dodatek D.
2. Oryginalna praca o model foldingu: Wang, Šikić, Thiele, Saukh,
   *Forget the Data and Fine-tuning! Just Fold the Network to Compress*,
   ICLR 2025, arXiv 2502.10216 — dodatek A, limitations, future work.
3. Repozytoria: `osaukh/folding_as_projection`, `nanguoyu/model-folding-universal`,
   `marza96/ModelFolding` (README i parametry uruchomienia).
4. alphaXiv: wyszukiwania „model folding”, „layer-wise compression”,
   „channel merging”, „allocation”, „non-uniform”.
5. Cytowania *Cut Less, Fold More* przez Semantic Scholar API (stan 2026-10-31).

## Ustalenia

### 1. Oryginalny folding sam wskazuje tę lukę jako future work

Wang et al. (2025), dodatek A:

> „For a folding experiment, we apply the same compression ratio to all layers.”

Wang et al. (2025), Limitations and future work:

> „Additionally, it does not optimize sparsity levels per layer, leaving this
> for future work.”

To najsilniejszy dowód: **autorzy foldingu wprost piszą, że nie optymalizują
sparsity per warstwa i zostawiają to na przyszłość.**

### 2. *Cut Less, Fold More* używa jednolitych ratio

- Podpis Rys. 1: „folding at **uniform per-layer compression ratios**”.
- Podpis Rys. 8 (MAG2): to samo sformułowanie.
- Tabela 1 (LLaMA): pojedynczy „layer-wise pruning ratio of 20% and 50%”.
- Ablacje dotyczą optymalizatora, learning rate, SAM, augmentacji i treningu —
  nie alokacji budżetu między warstwy.

### 3. Kod oficjalny nie eksponuje alokacji per warstwa

- `nanguoyu/model-folding-universal`: folding sterowany jednym parametrem
  ratio (`pr` w recipe; README: „adjust the folding ratio by changing the
  parameter `pr`”).
- `osaukh/folding_as_projection`: `keep_ratio` jest wspólny; kod obsługuje
  kilka klasteryzatorów, ale nie szuka różnych ratio per warstwa.
- To nie wyklucza, że ktoś zmodyfikował kod prywatnie, ale publiczny
  interfejs nie oferuje alokacji.

### 4. Cytowania *Cut Less, Fold More*

Semantic Scholar zwrócił 4 prace cytujące (stan 2026-10-31):
- On the Interaction Between Model Compression and Test-Time Adaptation,
- Understanding Calibration and Truncation Error Propagation in Training-Free
  Low-Rank Compression for LLMs,
- DLR: Zero-Inference-Cost Latent Residuals for Low-Rank Pre-Training,
- GRAIL: Post-hoc Compensation by Linear Reconstruction for Compressed Networks.

Żadna nie dotyczy alokacji budżetu w foldingu.

### 5. Prace pokrewne, ale nie folding

Alokacja per warstwa jest standardem w:
- pruning LLM (np. PALS, Wanda i warianty),
- kompresji low-rank/SVD (np. UniRank, layer-wise error model search),
- alokacji szerokości w transformerach.

To ważne: **sam pomysł alokacji nie jest nowy w kompresji**. Nowe byłoby
systematyczne zbadanie go **w calibration-free foldingu**, gdzie oryginalna
praca zostawia to jako future work.

## Wnioski

1. **Luka jest realna i dobrze udokumentowana**: autorzy foldingu sami ją
   nazywają; *Cut Less, Fold More* jej nie bada; publiczny kod nie oferuje
   alokacji per warstwa.
2. **Nie jest to „pierwsza alokacja per warstwa w kompresji w ogóle”** — tylko
   pierwsze systematyczne badanie w foldingu. Tak trzeba to formulować.
3. Ryzyko recenzenckie: „to oczywiste rozszerzenie”. Odpowiedź: (a) oryginalna
   praca zostawia to jako otwarte, (b) nasze wyniki pokazują, że naiwna
   alokacja oparta na wrażliwości (AUTO50) nie działa, a prosta reguła TILT
   daje duże zyski — to nie jest trywialne, (c) potrzebny mechanizm.

## Ograniczenia audytu

- Nie jest to pełny systematic review: brak Google Scholar/Scopus,
  prac warsztatowych, tez i patentów.
- Cytowania oryginalnego foldingu nie zostały wyliczone (API zwróciło 429);
  sprawdzone tylko cytowania *Cut Less, Fold More*.
- Nie sprawdzono IFM (Chen et al., 2023) w pełnym tekście.
- Możliwe inne nazwy zjawiska: heterogeneous compression, adaptive folding,
  per-layer budget, non-uniform width.
- Kod sprawdzony przez README/interfejs, nie przez pełny audyt wszystkich
  skryptów.

## Co dalej

1. Dokończyć cytowania Wang et al. 2025 (Semantic Scholar/OpenAlex) i przejrzeć
   tytuły pod kątem „layer-wise/allocation/folding”.
2. Sprawdzić IFM.
3. Dodać do paperu cytat z Wang et al. (future work) i z CLFM (uniform ratios).
4. Formułować: „to the best of our knowledge, non-uniform budget allocation
   has not been systematically studied for calibration-free folding”.

## Źródła

- Wang et al., ICLR 2025: https://arxiv.org/abs/2502.10216
- Saukh et al., 2026: https://arxiv.org/abs/2602.18116
- Kod: https://github.com/nanguoyu/model-folding-universal ,
  https://github.com/osaukh/folding_as_projection
