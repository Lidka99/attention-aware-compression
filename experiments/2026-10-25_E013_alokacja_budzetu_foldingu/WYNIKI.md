# E013 — przesiew alokacji budżetu foldingu

Status: **completed** (33/33 workerów, exit0). Run:
`runs/E013/20261027T093342Z_b60fd73_ca4f6700`. Jeden checkpoint E011 seed42.

## Cel

Sprawdzić, czy przy tym samym globalnym budżecie parametrów nierówne
rozdzielenie kompresji między grupy foldingu bije jednolity folding.

## Wynik (dev top-1 po BN)

| Budżet | UNIFORM | TILT+0.30 | TILT+0.15 | najlepszy RANDOM | MAG2 |
|---:|---:|---:|---:|---:|---:|
| 30% | 93,48% | 94,06% | 94,00% | 93,84% | 91,50% |
| 50% | 88,98% | **93,32%** | 91,92% | 92,00% | 83,48% |
| 70% | 77,16% | **87,86%** | 84,62% | 81,58% | 58,52% |

TILT+0.30 vs UNIFORM: **+0,58 pp** [+0,12; +1,04] przy 30%,
**+4,34 pp** [+3,62; +5,10] przy 50%, **+10,70 pp** [+9,62; +11,84]
przy 70%. McNemar p < 0,05 na każdym budżecie. Parametry dopasowane
(≤0,2% różnicy). TILT+0.30 bije też MAG1/MAG2 i najlepszą kontrolę RANDOM
na 50/70%.

## Wniosek

Jest wyraźny sygnał, że **alokacja budżetu ma znaczenie** i że warto
oszczędzać wczesne warstwy, a mocniej kompresować późne. To jeden seed —
wynik wymaga replikacji, którą wykonuje E014. Pełne tabele, kontrolki
i metryki: [OPIS.md](OPIS.md), `analysis.json` w katalogu biegu.
