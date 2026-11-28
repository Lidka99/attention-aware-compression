# E050 — łagodniejszy crop poprawia top‑1 wszystkich profili na Tiny dev

[Run](../../runs/E050/dev_20261125T231044Z_d44c66e4/) ·
[audyt](../../runs/E050/dev_20261125T231044Z_d44c66e4/audit.json) ·
[per seed](../../runs/E050/dev_20261125T231044Z_d44c66e4/per_seed.csv)

Na czterech checkpointach E048 zmieniono wyłącznie dolną skalę kadrowania
A4096/CMA z 0,40 na 0,65. Cztery GPU, 4×exit 0, 102,05 s, 32 oceny dev,
w tym 16/16 odtworzeń widoków, stanów i top‑1 E049. Parametry i MAC bez zmian.

| Profil | Top‑1 przy 0,40 [%] | Top‑1 przy 0,65 [%] | Zmiana [pp] |
|---|---:|---:|---:|
| Pełny | 61,6875 | 62,6125 | +0,9250 |
| UNIFORM | 56,1650 | 57,2925 | +1,1275 |
| T-proj | 54,9550 | 56,4050 | +1,4500 |
| L06 | 56,8900 | 58,2100 | +1,3200 |

Każdy profil poprawił top‑1 na 4/4 seedach. Strata L06 do pełnego maleje
na wszystkich seedach; średnio z 4,7975 do 4,4025 pp (−0,3950 pp). Zysk nad
UNIFORM rośnie z +0,7250 do +0,9175 pp (w obu warunkach 3/4); zmiana tego
kontrastu jest dodatnia tylko w 2/4 parach. T-proj zyskuje na zmianie cropu
więcej niż L06, ale wynik L06 pozostaje wyższy o +1,8050 pp (4/4).

**Bramka diagnostyczna niespełniona** przez efekt poniżej 1 pp i stratę ponad
3 pp. To poprawa wspólnej procedury kalibracji na dev i częściowy odzysk
jakości względnej, bez potwierdzenia na nowych obrazach i bez zmiany
werdyktu E049. Gdyby nowy pełny i UNIFORM utrzymały wyniki, L06 potrzebuje
jeszcze 1,4025 pp na Tiny dev, aby przejść oba progi.

[Protokół](OPIS.md) · [Odtworzenie](URUCHOMIENIE.md)
