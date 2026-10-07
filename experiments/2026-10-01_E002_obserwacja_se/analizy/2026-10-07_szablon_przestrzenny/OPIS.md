# E002 — szablon przestrzenny i odpowiedź zmienna między obrazami

Eksperyment: `2026-10-01_E002_obserwacja_se`. Analiza z 2026-10-07 UTC.
Wersja: **1.0, frozen przed obliczeniem wyników**. Status wykonania
zapisze manifest. Uzupełnienie E002, bez nowego numeru eksperymentu.
**Post hoc:** pytanie wynika z wcześniejszych obserwacji, nie jest
niezależnym badaniem potwierdzającym.

## Pytanie i zakres

Czy podobieństwo wcześniej wybranych 100 par utrzymuje się po odjęciu
**całej średniej mapy po obrazach**, a nie tylko jej skalarnej średniej?
Te same 19 kanałów i 100 par wybranych wcześniej na A, bez nowego
rankingu. Osobno A i B; główna prezentacja na B. Jeden blok/checkpoint,
te same obrazy train. Bez zmiany modelu, inferencji, danych dev/test/rezerwy.

Duży udział szablonu oraz znaczny spadek podobieństwa odchyleń byłyby
zgodne z dominacją wspólnego szablonu. Duży udział odpowiedzi zmiennej
i utrzymanie jej podobieństwa przemawiałyby przeciw temu wyjaśnieniu.
Wynik może być mieszany. Próg 1/2 służy wyłącznie opisowi większości
energii, nie stanowi kryterium skuteczności scalenia. Nie dobieramy po
wyniku progu podobieństwa lub minimalnej interesującej korzyści kompresji.

## Wejścia i niezmienność

- Źródło: `runs/E002/20261001T051649Z_e0b3121_seed42_a8l9uuks/`.
- SHA-256 źródłowego artifacts.sha256:
  `c7c162e76ed54f38941b278a6ba298af4da234baf204896aa5f21a9b98b0483f`.
- SAMPLES.json: `338eb31862ab3d35495c22c8ae58109aed280a8218f1c32e234c941c3d1bae1d`.
- selected_pairs.csv: `f1d166ae88144e6f45edcc078a5415b2b210111e058421b190b38b616b4218b8`.
- Cztery rank*.npz z hashami w [config.json](config.json). Zapisane h/z,
  FP32, obrazy 96×96, blok features.4.1, 480 map 6×6.
- Kontrola wcześniejszego rozkładu:
  `runs/E002/analyses/20261002T055900Z_9dc2720_constant_kc4jiyub/`,
  artifacts.sha256 `ee780ee5e3326bb0990827d257de4b5d9da9cd645efe1a80955d556d907e82ae`.

Przed analizą sprawdzamy cały źródłowy manifest i manifest odniesienia,
ID/etykiety rank::4, 1000 unikalnych obrazów, rozłączność A/B (po 500,
5/klasę). Nie otwieramy CIFAR ani checkpointu. Wcześniejszy kod kontroli
hashy i ID jest używany tylko do odczytu, bez jego modyfikowania;
zależności i ich hashe będą zapisane w manifestach i kopii źródeł.

## Rozkład map i miary — ustalone przed obliczeniem

Dla U(x,p), czyli h albo z danego kanału, N=500 i P=36:

- mu=mean_xp(U);
- I(x)=mean_p(U(x,p))-mu — zmienna średnia obrazu;
- S(p)=mean_x(U(x,p))-mu — stały, niejednorodny wzór przestrzenny;
- R(x,p)=U(x,p)-mu-I(x)-S(p) — zmiany wzoru zależne od obrazu.

W zbalansowanej tabeli N×P te cztery składowe są ortogonalne.
E=sum(U²)=E_const+E_image+E_spatial_template+E_interaction, gdzie
E_const=NPmu², E_image=P sum(I²), E_spatial_template=N sum(S²),
E_interaction=sum(R²).

**T(p)=mean_x(U)=mu+S(p)** to średnia mapa, szablon tej próby.
**D(x,p)=U(x,p)-T(p)=I(x)+R(x,p)** to odpowiedź odmienna od szablonu.
Szablon i centrowanie szacujemy **osobno na A i B**. Są opisem zapisanej
próby, nie uniwersalnym wzorem ani transformacją uczoną do oceny modelu.

Na kanał: normy U, powtórzonego T, D i R oraz udziały czterech energii,
E_template/E (const+spatial_template), E_dynamic/E (image+interaction),
E_interaction/(E_spatial_template+E_interaction). Ostatnia miara określa,
ile dawnej składowej przestrzennej zmienia się między obrazami.
Odnosimy 19 kanałów do wszystkich 480, bez zmiany wyboru.

Na tych samych parach: cosinus U (s_raw), szablonu T (s_template), D
(s_dynamic) i R (s_interaction), zmiany s_dynamic-s_raw i s_interaction-s_raw.
Sprawdzamy również rozkład surowego cosinusa:
C_template=N dot(Tc,Td)/(norm(Uc)norm(Ud)),
C_dynamic=dot(Dc,Dd)/(norm(Uc)norm(Ud)), s_raw=C_template+C_dynamic.
Wkłady mają znak, nie są udziałami procentowymi. Nie dzielimy ich przez
s_raw, który może być mały.

## Kontrole numeryczne

FP64, bezpośrednie odejmowanie średnich od zapisanych map. Normy <=1e-12:
odpowiedni cosinus lub udział nieokreślony, NaN w CSV i licznik w JSON,
nie zastępujemy zerem. D/R o normie <=1e-6 normy U oznaczamy jako małe
względnie, ale nie usuwamy z tabel. NaN/Inf w wejściach przerywa analizę.

Energia całkowita i template+dynamic: błąd względny <=1e-10 przy E>1e-24,
poniżej błąd bezwzględny <=1e-24. Rozkład surowego cosinusa i zgodność
z oryginalnym E002: atol=1e-10. Udziały const/image i suma
spatial_template+interaction muszą zgadzać się z wcześniejszą kontrolą
stałej średniej (atol=1e-10). Cosinusy przycinamy tylko przy błędzie
zaokrąglenia, z licznikiem korekt. Nie stosujemy testów istotności na
zależnych parach.

Przed analizą: sztuczne mapy ze wspólnym silnym szablonem i przeciwnymi
zmianami, same szablony (zerowe D/R), sam efekt obrazu (zerowe R),
losowe mapy dla kontroli bezpośrednich wzorów, zerowe kanały, niezmienność
wejść. [TESTY_CPU.json](TESTY_CPU.json) zapisze dowód i hashe kodu/configu.

## Wykonanie i artefakty

Conda attention, Python 3.11.15, NumPy 2.4.6. Wyłącznie CPU, jeden wątek
OMP/MKL/OpenBLAS, FP64. Analiza deterministyczna, bez losowania danych/par;
seed testów 20261007. Źródłowy odczyt E002 wykorzystał 4 GPU po 250
obrazów. Nie deklarujemy nowej pracy GPU ani niezależnych powtórzeń.

Z czystego drzewa po lokalnym commicie kodu/protokołu i dowodu kontroli:

```bash
conda run --no-capture-output -n attention python experiments/2026-10-01_E002_obserwacja_se/analizy/2026-10-07_szablon_przestrzenny/launch.py
```

Nowy katalog: `runs/E002/analyses/<UTC>_<git>_template_<unikalny>/`.
Konfiguracja, dokładna komenda, commit, środowisko/pip/Conda, źródła
i zależności, logi, kontrola wejść, kanały_A/B.csv (480) i pary_A/B.csv (100),
summary.json, manifest i artifacts.sha256 po zamknięciu logów.

Nowe wykresy w `../../wykresy/<analysis-ID>/`, każdy PNG/PDF:
surowy vs D/R (h/z), udział szablonu na tle 480 kanałów, cztery energie
dla 19 kanałów. Małe CSV/JSON także w folderze wykresów; raport WYNIKI.md
tutaj. Nie nadpisujemy zakończonych wyników. Weryfikacja: sha256sum -c
artifacts.sha256 i hashe wykresów z manifestu. Kod i małe wyniki w Git;
surowe aktywacje poza Git, nadal bez zewnętrznej kopii zapasowej.

## Granice wnioskowania i kolejny etap

Podobieństwo D/R pokazuje współzmienność na badanych obrazach, nie
ważność dla klasyfikacji ani bezpieczeństwo połączenia. Szablon także
może być użyteczny. Jeden blok/model, dane train, przeuczenie E001,
analiza post hoc, 19 kanałów i zależne pary. Odczyt po SE jest niezbędną
kontrolą, ale osobna informacja o wagach wymaga później wykazania korzyści.

Ta analiza domyka zaplanowane kontrole średniej i szablonu. Po wyniku
zatrzymujemy się; proponowany następny krok będzie dotyczył **przygotowania
kontrolowanej próby scalenia jednej pary** i pomiaru skutków, nie tworzenia
nowego kryterium na podstawie samych cosinusów. Nie uruchamiamy tutaj etapu 4.
