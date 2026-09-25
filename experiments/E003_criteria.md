# E003 — czy atencja pomaga wybrać scalenie?

Status: draft. Wersja: 0.1. Wymaga E000, E001, E002. Konfiguracja: `configs/E003_criteria.draft.json`.

## Hipoteza
Informacja o zachowaniu atencji pomaga wybrać grupę o mniejszym błędzie niż proste podobieństwo aktywacji po atencji przy tej samej szerokości i koszcie wyboru. Może się okazać redundantna — to dopuszczalny wynik.

## Porównania
Identyczna zamrożona pula legalnych kandydatów dla wszystkich selektorów: podobieństwo wag, aktywacji h, aktywacji z=a*h, samych a (diagnostyka), wybór losowy, proponowane kryterium. Dokładne odległości, normalizacja, obsługa zer i rozstrzyganie remisów wymagają specyfikacji przed uruchomieniem. Proponowane kryterium nie jest jeszcze zdefiniowane.

## Procedura
1. Zamrozić bloki, grupy kandydatów, sposoby naprawy i budżety. Zapisać manifest grup; podzielić bloki lub pule na rozwojowe i potwierdzające. Przeciek współdzielonych kanałów między zbiorami grup opisać i ograniczyć; split całych bloków daje mocniejszy test przenoszenia.
2. Obliczyć oceny selektorów na kalibracji. Kompensację wykonać według wspólnej procedury. Kandydatów oceniać na rzeczywistym mniejszym grafie z przeliczoną atencją.
3. Na części rozwojowej ustalić metodę. Zamrozić kod/konfigurację przed otwarciem części potwierdzającej. Każde późniejsze strojenie wymaga nowej wersji i nie może wykorzystywać tego samego potwierdzenia jako nietkniętego testu.
4. Ocenić wybranych kandydatów na rozłącznych danych. Dla małej puli ocenić wszystkich kandydatów jako punkt odniesienia; koszt tej diagnostyki raportować oddzielnie od kosztu selekcji. Etykiety i wyniki oceny nie trafiają do selektora.
5. Powtórzyć dla zamrożonych seedów kalibracji z tym samym checkpointem; interpretować to jako zmienność kalibracji.

## Metryka pierwotna
Regret wyboru: strata wybranego kandydata minus minimalna strata spośród tej samej ocenionej puli, w danym bloku i warunku naprawy. Strata: NLL dla klasyfikacji, jeśli zadanie zostanie tak ustalone. To optimum w skończonej puli, nie globalne optimum. Główne porównanie: proponowany selektor względem podobieństwa z.

Dodatkowo: błąd wyjścia bloku, accuracy, korelacja rankingu, koszt kalibracji i selekcji. Local block MSE nie zastępuje jakości zadania.

## Analiza i decyzja
Wyrównać szerokości, precyzję, zasoby naprawy i budżet selekcji. Raportować sparowane różnice osobno dla bloków/seedów; nie traktować kandydatów jako niezależnych modeli. Dokładna agregacja, próg praktycznej korzyści, liczba powtórzeń, tolerancje i limity zasobów: UNSET. Zamrozić przed potwierdzeniem.

Jeżeli nowa metoda nie poprawia wyboru ponad z lub korzyść wynika wyłącznie z dodatkowych ocen, brak wsparcia dla hipotezy. Przy szerokich przedziałach wynik `inconclusive` i kosztowo ograniczona decyzja o uzupełnieniu. Dopiero dodatni wynik uzasadnia plan E004: kompresja całej sieci i druga architektura.

Komenda odtworzenia: UNSET — brak implementacji. Raport: `reports/E003_criteria.md` (jeszcze brak). Wymagane manifesty, oceny wszystkich kandydatów, predykcje i hashe zgodnie z kontraktem reprodukowalności.
