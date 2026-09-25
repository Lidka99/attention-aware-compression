# Kandydat na lukę: folding uwzględniający kwantyzację

Data: 23 września 2026 r.
Status: wstępna hipoteza badawcza do dalszego przeglądu literatury; luka nie jest jeszcze potwierdzona.

## Punkt wyjścia

„Cut Less, Fold More” przedstawia folding jako alternatywę dla strukturalnego pruning’u. Autorzy formalizują obie metody przez geometrię projekcji i porównują je na wielu checkpointach CNN, ViT i małych modeli językowych. W ograniczeniach pracy piszą, że nie analizują interakcji foldingu z kwantyzacją ani destylacją oraz wskazują te połączenia jako dalszy kierunek.

Istotnym uzupełnieniem jest GRAIL: metoda kompensacji po kompresji, która wykorzystuje mały zbiór kalibracyjny do liniowej rekonstrukcji reprezentacji. Autorzy stosują ją również po foldingu. Wskazują jednak integrację kompensacji z kwantyzacją jako dalszą pracę. Zatem samo zastosowanie kompensacji GRAIL po foldingu nie byłoby nową kontrybucją.

Istnieją też prace o wspólnym stosowaniu pruning’u i kwantyzacji, w tym PQQP, które analizuje wpływ kolejności tych operacji. To ważne porównanie metodologiczne, choć pruning i folding są różnymi sposobami zmniejszania wymiaru sieci.

## Kandydackie pytanie badawcze

**Czy można dobierać grupy neuronów do scalenia tak, aby ograniczyć błąd końcowego modelu po foldingu, kompensacji i kwantyzacji, zamiast oceniać scalenie wyłącznie na wagach modelu w pełnej precyzji?**

Intuicja do sprawdzenia: folding zmienia wagi warstw odbierających wyjścia scalonych neuronów. Kwantyzacja zaokrągla te wagi do ograniczonego zestawu wartości, a kompensacja próbuje odzyskać część utraconego działania sieci. Dlatego grupy neuronów, które wyglądają na dobre do scalenia w FP32, mogą nie być najlepszym wyborem po kwantyzacji. To hipoteza, a nie ustalony fakt.

## Możliwy wkład

Kandydatem na wkład byłaby **reguła wyboru grup do scalenia świadoma końcowej kwantyzacji i kompensacji**. Powinna przewidywać jakość wynikowego modelu lepiej niż:

- standardowy folding oceniany w pełnej precyzji;
- standardowy folding, po którym stosuje się kompensację;
- proste sekwencyjne zastosowanie foldingu, kompensacji i kwantyzacji;
- odpowiednie punkty odniesienia z literatury o joint pruning–quantization.

Nowość nie powinna opierać się wyłącznie na zestawieniu istniejących etapów. Trzeba pokazać, że sama reguła doboru grup lub jej mechanistyczne uzasadnienie poprawia wynik przy równym koszcie i budżecie.

## Minimalny eksperyment pilotażowy

1. Wybrać jeden sprawdzony CNN i jego checkpoint, np. ResNet-18.
2. Ustalić jeden poziom kompresji, docelową precyzję i wspólny zbiór kalibracyjny.
3. Porównać co najmniej: model bazowy; folding; folding z kompensacją GRAIL; proponowany dobór grup z kompensacją i kwantyzacją; oraz dostępny punkt odniesienia z pruningiem i kwantyzacją.
4. Utrzymać porównywalny rozmiar wynikowy, precyzję, dane kalibracyjne i budżet dostrajania.
5. Raportować accuracy lub stratę, różnicę wyjść wobec oryginału, rzeczywisty rozmiar wyeksportowanego pliku, czas i pamięć wykonania. Jeśli stosowane jest fine-tuning, jego budżet musi być taki sam dla wszystkich metod.

Hipoteza otrzyma wstępne wsparcie, jeśli dobór grup uwzględniający końcową kwantyzację poprawi wynik względem zwykłego foldingu połączonego z kompensacją, przy porównywalnym rozmiarze i budżecie. Jeżeli przewaga znika po wyrównaniu kalibracji lub dostrajania, nie będzie podstaw do rozwijania tej reguły w obecnej postaci.

## Ryzyka i granice obecnej oceny

- Jawne wskazanie przyszłej pracy przez autorów nie dowodzi samo w sobie luki; trzeba sprawdzić dalszą i równoległą literaturę.
- Połączenie znanych etapów może być zbyt małym wkładem, jeśli nie pojawi się nowa reguła, analiza lub wynik mechanistyczny.
- Należy sprawdzić silne metody joint compression, w tym pruning–quantization, a nie ograniczać porównania do pruning’u według norm wag.
- Termin „folding” bywa używany dla różnych operacji. Na przykład FoldQuantVLA dotyczy spójnych transformacji współrzędnych i składania ich do wag w kwantyzacji modeli vision-language-action; nie jest automatycznie tą samą metodą co scalanie neuronów w „Cut Less, Fold More”.
- Wniosek dotyczy na razie kierunku do audytu, a nie potwierdzonej oryginalności ani gwarancji publikacyjnej.

## Źródła

1. Saukh i in., **Cut Less, Fold More: Model Compression through the Lens of Projection Geometry**, ICLR 2026. [Strona konferencji](https://proceedings.iclr.cc/paper_files/paper/2026/hash/82a227f51354b7ebafea51023d547d29-Abstract-Conference.html), [arXiv](https://arxiv.org/abs/2602.18116).
2. Tang i in., **GRAIL: Post-hoc Compensation by Linear Reconstruction for Compressed Networks**, 2026. [arXiv](https://arxiv.org/abs/2602.23795).
3. Kim i in., **Prune-then-Quantize or Quantize-then-Prune? Understanding the Impact of Compression Order in Joint Model Compression**, ICLR 2026. [arXiv](https://arxiv.org/abs/2603.18426).
4. Ho i in., **FoldQuantVLA: Native Low-Bit Quantization of Vision-Language-Action Models via Consistent Folding**, preprint, 2026. [arXiv](https://arxiv.org/abs/2609.24433).
