# E040 — tabele z zamrożonej próby

Run: `runs/E040/20261111T032100Z_07ac54f7`. Źródło: `analysis/audit.json`, audyt PASS.

## Predykcja szkody na C

MAE w pp niezgodności decyzji z teacher, nie w pp trafności top1.
Średnia±SD(ddof=1) po4seedach; iloraz średnich, nie średnia ilorazów.

| Zbiór | Addytywny po BN | Pary po BN | Iloraz | Poprawa |
|---|---:|---:|---:|---:|
| cifar10 | 1.1330±0.3114 | 1.9997±0.7756 | 1.7650 | 0/4 |
| cifar100 | 1.6472±0.3694 | 3.8406±1.0789 | 2.3316 | 0/4 |

| Zbiór | Seed | MAE addytywny [pp] | MAE pary [pp] | Iloraz |
|---|---:|---:|---:|---:|
| cifar10 | 42 | 1.4557 | 2.1057 | 1.4465 |
| cifar10 | 43 | 1.3397 | 3.0548 | 2.2802 |
| cifar10 | 44 | 0.9125 | 1.3641 | 1.4950 |
| cifar10 | 45 | 0.8240 | 1.4740 | 1.7889 |
| cifar100 | 42 | 1.7548 | 5.1666 | 2.9443 |
| cifar100 | 43 | 2.0813 | 2.7466 | 1.3196 |
| cifar100 | 44 | 1.1993 | 4.2297 | 3.5267 |
| cifar100 | 45 | 1.5533 | 3.2196 | 2.0727 |

## Faktyczne koszty wybranych alokacji

Zakresy po4seedach. Wszystkie spełniają oba limity, bez ich przekroczenia.
Wspólne ograniczenia nie oznaczają dokładnie dopasowanych P i MAC.

| Zbiór | Operator/reguła | P min–max [M] | MAC min–max [M] |
|---|---|---:|---:|
| cifar10 | FOLD/ADD_POST | 4.832814–5.321473 | 271.204942–275.824206 |
| cifar10 | FOLD/PAIR_POST | 4.884325–5.408656 | 249.147274–274.717450 |
| cifar10 | FOLD/PAIR_PRE | 5.006801–5.558749 | 261.367690–273.596494 |
| cifar10 | MAG2/ADD_POST | 4.681294–5.513358 | 267.371086–274.541194 |
| cifar10 | MAG2/PAIR_POST | 4.564632–5.372802 | 258.136330–276.131374 |
| cifar100 | FOLD/ADD_POST | 5.445556–5.563557 | 270.883924–274.754260 |
| cifar100 | FOLD/PAIR_POST | 5.352136–5.569615 | 264.886356–273.424348 |
| cifar100 | FOLD/PAIR_PRE | 4.720948–5.379152 | 265.673812–272.927644 |
| cifar100 | MAG2/ADD_POST | 5.550258–5.563557 | 270.883924–274.578004 |
| cifar100 | MAG2/PAIR_POST | 5.550258–5.563557 | 261.778516–274.578004 |

## Referencje U/TILT i anchor

Te same szerokości dla FOLD i MAG2; jakość dev nieoceniona po STOP_C.

| Zbiór | Profil | P | MAC |
|---|---|---:|---:|
| cifar10 | U | 5584580 | 276383716 |
| cifar10 | TILT_05 | 5054938 | 276199680 |
| cifar10 | TILT_10 | 4528512 | 275994248 |
| cifar10 | TILT_15 | 4025340 | 276128240 |
| cifar10 | TILT_20 | 3543899 | 275614046 |
| cifar10 | TILT_25 | 3147343 | 276365216 |
| cifar10 | TILT_30 | 2879151 | 276376804 |
| cifar10 | TILT_PROJ | 4882317 | 276237070 |
| cifar10 | ANCHOR | 5246333 | 275824206 |
| cifar100 | U | 5581542 | 275335732 |
| cifar100 | TILT_05 | 5060392 | 275294492 |
| cifar100 | TILT_10 | 4531539 | 275286780 |
| cifar100 | TILT_15 | 4013238 | 273449560 |
| cifar100 | TILT_20 | 3565263 | 275300252 |
| cifar100 | TILT_25 | 3156083 | 275257920 |
| cifar100 | TILT_30 | 2859758 | 275137824 |
| cifar100 | TILT_PROJ | 4487553 | 269508220 |
| cifar100 | ANCHOR | 5426105 | 273062716 |

## Wykonanie

Czas ścienny4GPU: 3032.72s; rezerwacja≤3.3697GPU-h.
Konstrukcje: 1558; CMA: 1550.

| GPU | Max wykorzystanie w logu [%] | Próbki z aktywnością | Max pamięć [MiB] |
|---:|---:|---:|---:|
| 0 | 63 | 140 | 1182 |
| 1 | 61 | 134 | 1182 |
| 2 | 68 | 141 | 1182 |
| 3 | 58 | 138 | 1182 |

To telemetria wykonania profilowania, nie benchmark latencji skompresowanych modeli.
Brak nowych pomiarów jakości i brak wyborów na podstawie tej analizy.
