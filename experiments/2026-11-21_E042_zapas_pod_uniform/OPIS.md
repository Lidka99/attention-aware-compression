# 2026-11-21_E042_zapas_pod_uniform — czy jest zapas jakości pod pułapami UNIFORM?

**Pytanie.** Czy choć jeden profil z małej, z góry ustalonej puli zachowuje
więcej trafności niż dwie referencje przy twardych pułapach liczby parametrów
i MAC z UNIFORM? E041 dopasowywał do większego MAC TILT; E040 sprawdzał model
zastępczy i zatrzymał się przed dev. Tu mierzymy bezpośrednio jakość skończonej
puli profili, aby zdecydować, czy warto budować selektor.

## Zakres i pułapy

ResNet18, nominalnie 50% redukcji **parametrów**, własne checkpointy E011/E015,
seedy 42–45. CIFAR-100 jest przypadkiem głównym (sygnał w poprzednich próbach),
CIFAR-10 kontrolnym. Pułapy pochodzą z historycznego U/b50 (E035/E037):

| Zbiór | Maks. P | Maks. MAC/obraz 32×32 |
|---|---:|---:|
| CIFAR-100 | 5 581 542 | 275 335 732 |
| CIFAR-10 | 5 584 580 | 276 383 716 |

P liczy wszystkie parametry (nie bufory); MAC obejmuje Conv/Linear. Każdy
profil kompresji ma 99,9–100% **obu** pułapów, bez przekroczeń. Szerokości są
całkowitoliczbowe dla 12 powiązanych grup w kolejności upstream, z granicami
`floor(0.1n)` i `floor(0.9n)` jak w kontroli E041. Teacher służy tylko do
odniesienia jakości i jest poza pułapami.

## A. Zamrożenie puli bez wyników jakości

U to historyczny UNIFORM. `T-proj` to deterministyczna projekcja historycznego
TILT z E035 na oba pułapy U. Dla każdej propozycji używamy tej samej procedury
geometrycznej co E041: SLSQP z punktem startowym U, minimalizacja odległości
kwadratowej udziałów `k/n`, cele `0,9995 × (P_U, MAC_U)`; zaokrąglenie i
deterministyczna naprawa najwyżej dwóch szerokości. Końcowe koszty, a nie
status solvera, rozstrzygają wykonalność. Nie gwarantujemy globalnie najbliższej
projekcji. `T-proj` musi być wykonalny i odległy od U o RMS ≥0,025 w `k/n`.

Na zbiór do 24 nowych profili. Zaczynamy od U i trzy rodzaje propozycji
budujemy odchyleniami udziałów kanałów:

- 8 wymian w obrębie etapu: w parze szerokości ukrytych dwóch bloków
  każdego etapu `(+0,12, −0,12)` i odwrotnie;
- 4 między etapami: wszystkim trzem grupom jednego etapu dodajemy 0,16,
  odejmując 0,16 dla drugiego (pary etapów 1↔4 i 2↔3; oba kierunki);
- pierwsze 12 wykonalnych i odrębnych z maks. 48 losowych propozycji
  (PCG64 `20261121`, te same odchylenia w obu zbiorach, niezależnie dla
  grup losowane całkowite pp od −18 do +18 włącznie).

Propozycje zaokrąglamy do kanałów i ograniczamy do legalnego zakresu przed
projekcją. Przyjmujemy w zapisanej kolejności, gdy wynik końcowy jest wykonalny
i ma RMS ≥0,025 względem każdej przyjętej szerokości, U i T-proj. Zachowujemy
także odrzucone próby. Bramka geometrii na każdym zbiorze: **≥6** wewnątrz etapu,
**≥2** między etapami i **12** losowych; brak spełnienia zatrzymuje pilot.
Nie jest to jednostajna próbka wszystkich geometrii. Zamrożone szerokości
są wspólne dla czterech seedów modelu na danym zbiorze.

Kontrola CPU przed GPU: porównanie grup i kosztów z geo/hookami operatora,
na seedzie 42 każdego zbioru dla U, T-proj i pierwszego profilu każdej rodziny
(syntetyczny obraz, bez odczytu obrazów lub wyniku klasyfikacji). Na GPU
przy **każdym** profilu ponownie kontrolujemy zrealizowane P i MAC, szerokości,
niepuste klastry, bias FC oraz brak gradientów.

## B. Ocena i kryterium decyzji

Operator upstream `6946915` jak w E035: bias klasyfikatora zachowany,
bez aktualizacji gradientowych. Jedna procedura BN: reset + CMA,
zamrożone A4096 z train, crop/flip identyczne do E035. Oceniany jest tylko
dev5000; role A/dev sprawdzamy względem splitów checkpointów, z wykluczeniem
indeksów rezerwy. Testy E035 pozostają poza tym pomiarem.

Cztery procesy: seed 42/43/44/45 → GPU 0/1/2/3, na każdym oba zbiory
po kolei, batch 128 na GPU dla A i oceny; FP32, deterministycznie, TF32 off.
Wspólna bramka przed nowymi pomiarami: wszystkie procesy odtwarzają teacher
i U dla obu zbiorów z dev E035 (identyczne top-1, stan i widoki A, NLL w
tolerancji 1e−5). Maksimum 216 ocen po CMA (192 kandydatów, 8 T-proj,
8 U i 8 teacherów), limit 30 min na 4 GPU, bez nowych plików wag.

Per zbiór wybieramy **jeden** profil o najwyższej średniej trafności po
czterech sparowanych seedach z puli kandydatów oraz mocniejszą z referencji
U/T-proj po tej samej średniej. Porównujemy liczby trafień na dev5000;
remisy rozstrzyga mniejszy MAC, mniejsze P, następnie ID. Nie wybieramy
innego zwycięzcy per seed. Raportujemy wszystkie profile/seedy, top‑1 i NLL,
sparowane średnie i SD, koszty oraz stratę względem teacher.

**Bramka dla dalszych prac:** tylko na głównym CIFAR-100 średnia różnica
≥1,00 pp (≥200 dodatkowych trafień przez cztery sparowane dev5000) i wynik
ściśle dodatni na ≥3/4 seedów, przy spełnieniu obu pułapów. Wyniku kontrolnego
CIFAR-10 nie łączymy z głównym. To próg decyzyjny, nie test istotności.
Ujemny wynik zamyka pilot; dodatni uzasadnia osobny test reguły selekcji.
Wybranie profilu na tym samym dev może zawyżać ocenę zapasu. Wynik dotyczy
tylko tej puli, bez wniosku o maksimum globalnym, przewadze na nowym zbiorze
ani rzeczywistej latencji. Nie zmieniamy puli ani progu na podstawie jakości.
