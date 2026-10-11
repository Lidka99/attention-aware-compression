# 2026-10-11_E006_kryterium_bramkowe_predykcja

Wersja protokołu: **0.1-draft**. Status: projekt; nic nieuruchomione.
Nowa próba badawcza: badanie **predykcyjne** (bez pełnej kompresji), czy
zachowanie bramek SE przewiduje rzeczywistą szkodę scalenia pary lepiej niż
podobieństwo aktywacji po SE (kontrola główna).

## Pytanie i hipoteza

Czy sygnały z bramek SE, zależne od obrazu, pozwalają przewidzieć szkodę
scalenia pary kanałów **ponad** to, co już widać w podobieństwie aktywacji
po SE (z)? Hipoteza: dla par o podobnym z, ale różnym zachowaniu bramek
(a_c(x)/a_d(x) zmienia się inaczej między obrazami), rzeczywista szkoda
scalenia różni się, a sygnał bramkowy to przewiduje.

To jest etap „wykrycie zjawiska przed projektowaniem metody" (etap C planu).
Sukces tu nie dowodzi jeszcze przewagi nowej metody w pełnej kompresji.

## Baseline i dane

- Model: `last.pt` z E005 (Tiny ImageNet, 96×96, EfficientNet-B0 z SE),
  po zakończeniu treningu; SHA-256 i ścieżka zostaną wpisane przy freeze.
- Blok: `features.4.1` (480 kanałów, mapy 6×6, jak w E002).
- Kalibracja: rozłączne połowy A/B z train E005 (np. po 500 obrazów,
  stratyfikowane po klasach, seed ustalony przed uruchomieniem).
  A: dopasowanie alpha/beta i wszystkie sygnały; B: pomiar szkody.
- Zero rezerwy, testu i dev w tym badaniu. Dev E005 nie jest dotykany.

## Pary kandydackie (bez selekcji po wyniku)

Pula par: losowanie deterministyczne K par (np. K=300) z podziałem na
kubełki podobieństwa h (np. 5 kubełków × 60 par), aby mieć spektrum —
ale **nie** wybieramy par według bramek ani szkody. Pary rozłączne lub
nie — do ustalenia; przy nie rozłącznych raportujemy pokrycie kanałów.

## Szkoda rzeczywista (ground truth)

Dla pary (c,d), reprezentant c, usuwane d: dopasowanie alpha/beta na A
(jak E003), scalenie kopii modelu, pomiar na B **bez dostrajania**:
- względny MSE wyjścia bloku na wejściach nauczyciela (ta sama miara
  co w diagnostyce E003),
- dodatkowo zmiana logitów całej sieci na B (tańsza kontrola korelacji).
Szkoda mierzona na B, sygnały na A — brak przecieku dopasowania.

## Sygnały-kandydaci (obliczane na A)

1. `s_post` — cosinus wektorów z (kontrola główna, dotychczas najlepsza).
2. `s_pre` — cosinus wektorów h.
3. `g_corr` — cosinus profili bramek a_c(x), a_d(x) po obrazach.
4. `g_lstd` — std log-stosunku wag (miara z E002).
5. `g_flip` — częstość zmiany znaku różnicy wag (a_c−a_d) między obrazami.
6. `g_amp` — korelacja a_c(x) z a_d(x) ważona siłą aktywacji h.
7. `g_sens` — przewidywane zaburzenie bramek po usunięciu d: z wag fc1
   można policzyć zmianę wejścia fc1 (średnia z g) i oszacować zmianę
   bramek wszystkich kanałów; norma tej zmiany na A. To sygnał „jak
   bardzo usunięcie d przestawia atencję", różny od podobieństwa z.
8. Kontrola losowa z ustalonym seedem.

Kombinacje (np. `s_post` + `g_sens`) raportujemy jako eksplorację,
wyraźnie oznaczoną jako wtórną, bez dostrajania wag kombinacji po wyniku
(stała reguła, np. rangi uśrednione).

## Kryterium oceny (ustalone przed uruchomieniem)

Na B liczymy Spearmana między każdym sygnałem a szkodą rzeczywistą oraz
jakość wyboru: ze puli K par wskaż najbezpieczniejszą 1/4 według sygnału
i porównaj średnią szkodę wybranej części z częścią wskazaną przez
`s_post` (różnica sparowana, bootstrap po obrazach B i po parach).
Sukces zjawiska: `g_*` lepsze od `s_post` o ustalony w protokole próg
(np. poprawa korelacji ≥0,05 i wybór bezpieczniejszej 1/4 w ≥70%
powtórzeń bootstrapu). Progi i seed bootstrapu wpiszemy przed freeze.

## Ograniczenia z góry

Jeden model i blok; szkoda lokalna bloku bez dostrajania; pary zależą od
siebie; brak niezależnego testu. Wynik pozytywny upoważnia do E007
(pełna kompresja 30%: nowe kryterium vs POST/PRE/RANDOM, równe budżety,
dostrajanie jak w E004). Wynik negatywny zamyka wątek bramek jako sygnału
ponad z — też zostanie zaraportowany.

## Kolejne kroki

1. Po zakończeniu E005: raport E005 i wpisanie hashy baseline'u do configu.
2. Implementacja pomiarów + kontrole CPU, freeze protokołu, uruchomienie
   na 4 GPU (obliczenia lekkie: forwardy kalibracji i K×2 scalonych kopii).
3. Raport: tabela korelacji, wybór 1/4, wniosek o dalszym kierunku.
