# E002 — kontrola składowej stałej, v1.0

Eksperyment: `2026-10-01_E002_obserwacja_se`. Data: 2026-10-02 UTC.
Status protokołu: **frozen przed obliczeniem tej analizy**; status wykonania
podaje manifest. Uzupełnienie E002, nie nowy eksperyment E003.
Zakres ustalony po obejrzeniu pierwszego wyniku E002 — **analiza post hoc**,
nie niezależne badanie potwierdzające. Nie zmieniamy [protokołu odczytu](../../PROTOKOL_v1.md).

## Pytanie, cel i reguła interpretacji

Czy wysoki surowy cosinus wcześniej wybranych 100 par wynika głównie
ze wspólnej składowej stałej kanałów, czy utrzymuje się też po usunięciu tła?
Opisujemy **te same pary**, obejmujące 19 kanałów, bez nowego rankingu
i bez dobierania przykładów według rezultatu centrowania.

Porównamy surowy cosinus z dwoma centrowanymi oraz udział energii składowych
w każdym kanale. Stała składowa ma większość energii, jeśli jej udział
przekracza 1/2 — to definicja rachunkowa, nie próg skuteczności kompresji.
Wniosek opieramy na pełnej tabeli i medianach/kwartylach, nie na wybranych
po wyniku przykładach. Duży udział stałej oraz spadek cosinusa po centrowaniu
byłyby zgodne z dominacją tła. Utrzymanie wysokiego cosinusa przy małym
udziale stałej przemawiałoby przeciw temu wyjaśnieniu. Możliwy jest wynik
mieszany. Nie ustalamy progu „podobieństwa wystarczającego do scalenia”.

## Dokładne wejścia

Wyłącznie zapisane artefakty biegu:
`runs/E002/20261001T051649Z_e0b3121_seed42_a8l9uuks/`.

- `rank0.npz`: `f72eeca57cb003cf0f1d1969f663f7ea19d754801621e56a70f29f22f0fce0f8`.
- `rank1.npz`: `c75c4e2087afd909bcf85e79036c19b4c7c4b46bd2ce712f3fc3865b0c2e2fe8`.
- `rank2.npz`: `13d9d960deb7155bd3a1ab60a5db08916b58a19286eda7d2241483d950ca36ac`.
- `rank3.npz`: `a14a12be48981c43db6c5245934cbb02884f2ecc3c2f17d9918b77435d756247`.
- `SAMPLES.json`: `338eb31862ab3d35495c22c8ae58109aed280a8218f1c32e234c941c3d1bae1d`.
- `selected_pairs.csv`: `f1d166ae88144e6f45edcc078a5415b2b210111e058421b190b38b616b4218b8`.
- `artifacts.sha256`: `c7c162e76ed54f38941b278a6ba298af4da234baf204896aa5f21a9b98b0483f`.
- Kod odczytu źródłowego: `e0b3121bef65d7722861aebbbfdaafe563883831`.

Sprawdzamy cały manifest źródłowy i dokładne ID/etykiety rank::4.
Zachowujemy kolejność A/B, po 500 obrazów, po 5/klasę. Kontrola pokrycia
obejmuje 1000 unikalnych ID. Pary pochodzą z wcześniejszego rankingu na A;
główna prezentacja pozostaje na B, A to kontrola spójności.

## Trzy ortogonalne składowe, osobno przed i po SE

Niech U_c(x,p) oznacza h lub z kanału c, x obraz, p jedną z P=36 pozycji.
N=500 na połowę. Dla każdego kanału i połowy osobno:

1. `mu_c = mean(U_c)` po wszystkich obrazach i pozycjach — stałe tło.
2. `m_c(x) = mean_p U_c(x,p)` — średnia każdej mapy.
3. Rozkład: `U_c = mu_c + (m_c(x)-mu_c) + (U_c(x,p)-m_c(x))`.

Energia E=sum(U²) ma dokładny rozkład:

- E_const=N*P*mu²;
- E_image=P*sum_x((m(x)-mu)²) — zmienność średniej mapy między obrazami;
- E_spatial=sum((U-m(x))²) — szczegóły przestrzenne wewnątrz map.

Zapisujemy udziały trzech energii w E, średnią ze znakiem, normy oraz
liczniki numerycznie zerowych składowych. Dla normy surowej <=1e-12
udziały są nieokreślone, nie zastępujemy ich zerem.

Dla tych samych par obliczamy:

- s_raw: pierwotny cosinus pełnych map;
- s_global: cosinus po odjęciu osobnej mu_c od kanału;
- s_spatial: cosinus po odjęciu m_c(x) od każdej mapy;
- zmiany s_global-s_raw oraz s_spatial-s_raw;
- C_const=N*P*mu_c*mu_d/(norm(U_c)*norm(U_d));
- C_rest=dot(U_c-mu_c,U_d-mu_d)/(norm(U_c)*norm(U_d)).

Kontrola: s_raw=C_const+C_rest. Nie dzielimy C_const przez s_raw, bo
ten może być bliski zeru lub udział może przekraczać 1 przy ujemnym C_rest.
Składowe wkładu do cosinusa są liczbami ze znakiem, nie procentami.
Globalne centrowanie zachowuje zmienność średniej między obrazami;
przestrzenne centrowanie usuwa również tę informację. Wynik po SE może
mieć zmienność średnich wynikającą z samego ważenia — nie nazywamy jej
automatycznie artefaktem ani zbędną informacją.

Udziały energii raportujemy dla 19 wybranych kanałów oraz jako odniesienie
dla wszystkich 480. Nie wyliczamy nowego rankingu par. Tabele na A i B
zachowują wszystkie 100 par, również te nieokreślone po centrowaniu.

## Kontrole numeryczne ustalone przed uruchomieniem

- Całe obliczenia FP64 na CPU, wejścia zapisane FP32. Brak nowych danych,
  checkpointu, inferencji i zmian modelu.
- Normy <=1e-12: cosinus nieokreślony (NaN w CSV, licznik w JSON).
  Nie interpretujemy braku zmienności jako podobieństwa 0 lub 1.
- Dodatkowo oznaczamy składowe o normie <=1e-6 normy surowej jako bardzo
  małe względnie; nie usuwamy ich automatycznie z tabel ani median.
- Kontrola E=E_const+E_image+E_spatial: błąd względny <=1e-10
  dla E>1e-24. Dla mniejszych E błąd bezwzględny <=1e-24.
- Kontrola s_raw=C_const+C_rest oraz zgodności s_raw z oryginalnym CSV:
  atol=1e-10. Surowe cosinusy pochodzą z tych samych map, nie z nowej inferencji.
- NaN/Inf w wejściach lub naruszenie powyższych kontroli oznacza failed.
  Cosinus przycinamy do [-1,1] tylko dla zaokrągleń, z licznikiem korekt.
- Wszystkie założenia sprawdzamy najpierw na małych sztucznych danych CPU:
  stałe tło i przeciwne wzory, mapy stałe przestrzennie, zerowe kanały,
  rozkład energii i niezmienność wejść. TESTY_CPU.json zapisuje dowód.

## Wykresy i raport

Trzy wykresy PNG/PDF na B:

1. Surowy vs globalnie centrowany i surowy vs przestrzennie centrowany
   cosinus, oddzielnie h i z.
2. Udział energii stałego tła: 19 kanałów na tle wszystkich 480, h/z.
3. Trzy udziały energii dla każdego z 19 kanałów, oddzielnie h/z.

Zapisujemy kanały_A/B.csv (480 kanałów) i pary_A/B.csv (100 par), mediany,
kwartyle, zakresy i liczniki nieokreślonych wartości oraz składowych małych
względnie. Nie stosujemy p-values ani przedziałów ufności opartych na
traktowaniu zależnych par jako niezależnych. Syntetyczny przykład ani
udział energii nie dowodzą bezpieczeństwa scalenia.

## Uruchomienie i archiwizacja

Conda attention, Python 3.11.15, NumPy 2.4.6. Wyłącznie CPU, jeden wątek
OMP/MKL/OpenBLAS. Nie losujemy obrazów ani par. Testy mają oddzielny seed
20261002; analiza rzeczywistych zapisanych tensorów jest deterministyczna.

Pierwotne aktywacje uzyskano na **4 GPU, po 250 obrazów/rank**; ta analiza
CPU jedynie je podsumowuje, podobnie jak tworzenie wykresów. Nie deklarujemy
ponownego wykorzystania czterech GPU ani nowych niezależnych powtórzeń.

Z katalogu głównego repo, po lokalnym commicie kodu, configu, protokołu
i kontroli CPU, z czystego drzewa:

```bash
conda run --no-capture-output -n attention python experiments/2026-10-01_E002_obserwacja_se/analizy/2026-10-02_skladowa_stala/launch.py
```

Launcher zapisze commit i dokładne komendy, kopię źródeł, hashe wejść,
konfigurację, środowisko CPU/pip/Conda, logi, tabele i status do **nowego**
`runs/E002/analyses/<UTC>_<git>_constant_<unikalny>/`. Nie zmienia źródłowego
biegu. Wykresy i małe tabele/podsumowania: nowy
`experiments/2026-10-01_E002_obserwacja_se/wykresy/<analysis-ID>/`.
Raport w tym folderze analizy: WYNIKI.md. Hashe po zamknięciu logów,
weryfikacja przez sha256sum -c artifacts.sha256 oraz manifest wykresów.
Kod źródłowy, protokół, raport i małe wykresy w Git; surowe aktywacje
pozostają poza Git. Brak zewnętrznej kopii artefaktów jest ograniczeniem.

Ograniczenia: analiza post hoc, jeden checkpoint/blok, dane train, zależne
pary, brak oceny jakości po scaleniach. Usunięcie średniej może usunąć
użyteczny sygnał dla dalszej projekcji; nie zmieniamy reprezentacji modelu
ani kryterium wyboru kanałów w tej analizie. Zatrzymujemy się przed etapem 4.
