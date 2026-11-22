# E042 — wykres zapasu jakości pod pułapami UNIFORM

Figura do zakończonego runu `runs/E042/20261121T212257Z_207ae03f`
(audyt PASS). Pokazuje **dev**, bez testu i rezerwy, bez nowych obliczeń
na GPU. Protokół i kryterium: [`../../OPIS.md`](../../OPIS.md).

## Pliki

| Plik | Zawartość |
|---|---|
| `e042_zapas_dev.png` | figura 2640×1440 px (200 dpi) |
| `e042_zapas_dev.pdf` | ta sama figura, wektorowa |
| `make_figure.py` | deterministyczny generator (tylko CSV/JSON z `analysis/`) |

## Co jest na wykresie

Dwa panele: CIFAR-100 (główny) i CIFAR-10 (kontrolny), po **24 kandydatów**;
referencje U i T‑proj opisuje linia 0 i podpis, nie osobne punkty. Oś pozioma
to średnia Δ top‑1 po czterech sparowanych seedach (42–45) względem
**mocniejszej** z referencji U/T‑proj; dla obu zbiorów jest nią U. Wspólna
kolejność wierszy według Δ CIFAR-100 pozwala porównać rangi; L06 jest
pierwszy w obu panelach. Kolor i kształt markera
oznaczają rodzinę (wewnątrz etapu 8, między etapami 4, losowe 12).
Pionowe linie: 0 (mocniejsza referencja), mediana (kropkowana) i **+1 pp**
(przerywana, próg decyzyjny ilustracyjny). Wiersz L06 jest podświetlony;
puste kropki nad nim to jego cztery sparowane Δ per seed. Rzeczywiste P i MAC
wszystkich 24 profili leżą w [99,9%, 100%] pułapów U (patrz podpis figury).

## Najważniejsze liczby (dev)

| Zbiór | U | T‑proj | mediana Δ | średnie > 0 | ≥3/4 seedów | najlepszy | Δ | seedy |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| CIFAR-100 | 53,855% | 53,025% | **−1,265 pp** | **5/24** | 4/24 | L06 | **+1,175 pp** | 3/4 |
| CIFAR-10 | 89,435% | 89,425% | **+0,0025 pp** | **12/24** | 8/24 | L06 | +1,475 pp | 4/4 |

Na CIFAR-100 tylko L06 spełnia próg ≥+1 pp i ≥3/4 seedów (235 dodatkowych
trafień przy wymaganych ≥200); `summary.json` ma `primary_pass: true`.
Kontrolny CIFAR-10 nie jest łączony z bramką.
Mediana rozkładu na CIFAR-100 jest ujemna, więc zapas nie jest typowy dla
puli — pochodzi od pojedynczego profilu wybranego na tym samym dev.

## Weryfikacja

- Wejścia (SHA256, sprawdzane też na sztywno w generatorze):
  - `profiles.csv` `670ff82e1859f8aa5bb407f798975ed566717a5473cadc11bc91d961cfcd9eaf`
  - `per_seed.csv` `407dfa9fd3c23f36314db529b2e823b3ecd657b11b41f3240d7c5c5d6813c69a`
  - `summary.json` `4950bfe1fc8344d134284b4b5b870624a2fb143b195427ee3e812cea750abc9b`
- Generator asertuje: 26 wierszy na zbiór (24 kandydatów + U + T‑proj),
  rodziny 8/4/12, 216 wierszy per seed (4 seedy na profil), zgodność
  przeliczonych średnich, liczby dodatnich seedów i wyboru L06 z
  `summary.json`, oraz `p_over_cap`, `mac_over_cap` ∈ [0,999, 1,0].
- Oba panele mają dokładnie tę samą kolejność identyfikatorów (ranking
  CIFAR-100), zgodną z pozycjami punktów. SHA256 figury:
  PNG `84d5c8b8a4097b5594bc7ae961ac3f32eedef3f12ee4a7b3c60b6fe3a5975b5d`,
  PDF `016e1545115d4fed4324ea2965631b6274676c5a99c663253975c2e85404bca5`.

## Odtworzenie

```bash
cd /home/users/s224574/attention/badania_atencja_cnn
/home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-11-21_E042_zapas_pod_uniform/wykresy/20261121T212257Z_207ae03f/make_figure.py
```

CPU/matplotlib only. Ograniczenia: wybór najlepszego profilu na tym samym
dev może zawyżać zapas; kreska +1 pp nie jest testem istotności; wynik dotyczy
tylko tej zamrożonej puli, nie maksimum globalnego ani rzeczywistej latencji.
