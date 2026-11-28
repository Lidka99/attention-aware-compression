# Na wtorek — teza, główny wynik i jego koszt

[Manuskrypt (PDF)](main.pdf) · [Tabela 1: wszystkie wyniki E049](tables/tilt_new.csv) ·
[Rysunek 1: pary i MAC](../experiments/2026-11-24_E049_folding_mocniejszych_baz/wykresy/20261124T234559Z_b4ca081b/e049_matched_parameters_en.pdf)

**Teza do powiedzenia.** Przy tym samym operatorze foldingu, tej samej
procedurze rekalibracji BN i dopasowanej liczbie parametrów zmiana rozdziału
szerokości poprawia top‑1 względem jednolitej redukcji kanałów na badanych
modelach ResNet18 dla CIFAR‑100 i Tiny ImageNet, bez dostrajania wag.
Mierzymy wielkość zysku i jego cenę w MAC.

## Główny wynik: E049, przykład przy redukcji około 50% parametrów

| Zbiór | Pełny po BN | UNIFORM | TILT | Zysk TILT [pp] | Strata TILT do pełnego [pp] | MAC U → T [mln/obraz] |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR‑100 | 79,6800% | 51,7100% | **61,4150%** | **+9,7050** | 18,2650 | 275,34 → 376,14 |
| Tiny ImageNet | 61,2175% | 21,2450% | **30,7075%** | **+9,4625** | 30,5100 | 275,37 → 376,03 |

TILT ma 99,9–100% P UNIFORM. Dodatnie są wszystkie cztery pary
checkpointów w każdym z sześciu ustawień (2 zbiory × 30/50/70%).
Na różnych budżetach mierzymy te same checkpointy. E049 to replikacja
na nowych wagach, na wcześniej użytym oficjalnym teście CIFAR‑100 i val Tiny.
Wcześniejszy E035 wspiera kierunek po CMA w 15/15 ustawieniach dwóch
architektur rezydualnych.

## Dwa pytania do obrony

- **Co jest wkładem względem literatury?** [*Cut Less, Fold More*](https://www.alphaxiv.org/abs/2602.18116v1?utm_source=alphaxiv-mcp&utm_medium=mcp)
  zmienia operator przy tych samych szerokościach (rys. 1/8, dod. F.2).
  My zmieniamy szerokości przy tym samym operatorze. [Wang i in.](https://www.alphaxiv.org/abs/2502.10216v2?utm_source=alphaxiv-mcp&utm_medium=mcp)
  już używają zadanego nierównego wzorca w rys. 7 i pozostawiają optymalizację
  alokacji jako otwartą (§5). Profil liniowy ma precedensy, m.in.
  [ATP](https://www.alphaxiv.org/abs/2502.14770v1?utm_source=alphaxiv-mcp&utm_medium=mcp).
  Wkład jest empiryczny: kontrolowany pomiar, ile daje zmiana alokacji
  w foldingu przy danym P i co to kosztuje.
- **Czy to daje przewagę obliczeniową?** Przy dopasowanym P TILT zużywa
  1,105–1,824× MAC UNIFORM. Przy zbliżonym MAC szerszy UNIFORM (z większą
  liczbą parametrów) wygrywa 10/15 ustawień na dev (E037). W odrębnej
  kontroli dopasowującej oba koszty do TILT wynik zależy od zbioru (E041):
  +2,10 pp nad średnią alternatywnych profili na CIFAR‑100 i +0,11 pp na
  CIFAR‑10; pilotaż nie obejmował Tiny. MAC nie jest pomiarem latencji.

## Kolejność na spotkaniu

1. **Rdzeń:** E049, krótka informacja o E035 i jeden zwarty blok o koszcie
   z E037/E041. Przy zyskach zawsze pokazać top‑1 pełnego modelu i MAC.
2. **Kontekst praktyczny:** b10 na dev w dwóch zdaniach. TILT daje
   +1,7650 / +1,7125 pp nad UNIFORM (CIFAR‑100/Tiny); S04 odzyskuje na Tiny
   kolejne 0,2000 pp, do 59,2050%. Strata 3,4075 pp pozostaje powyżej
   roboczego celu 3 pp tej próby. [E051](tables/tilt_mild_dev.csv) ·
   [E052](../experiments/2026-11-26_E052_lokalne_przesuniecie_b10/WYNIKI.md).
3. **Na pytania:** dodatki C i D zawierają tabele b5/b10, lokalną próbę S04,
   crop BN i Frozen offsets, w tym ImageNetV2 na starszych wagach. W głównym
   tekście zostaje obserwacja, że pod oboma pułapami UNIFORM zyski b10 na
   nowych wagach były mniejsze (+0,615 / +0,675 pp).

**Pytanie do promotora:** Czy taki zakres — kontrolowane badanie wpływu
alokacji i kosztu — wystarcza jako główny wkład artykułu?
