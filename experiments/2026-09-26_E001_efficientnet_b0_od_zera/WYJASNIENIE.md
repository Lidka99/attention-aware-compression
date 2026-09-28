# E001 — co robimy i po co?

Najpierw uczymy sieć rozpoznawać obrazy. Dopiero później spróbujemy ją zmniejszyć.

1. **Model:** EfficientNet-B0 z atencją SE. Zaczyna z losowymi wagami; nie korzysta z cudzej wytrenowanej sieci.
2. **Dane:** roboczo proponujemy CIFAR-100, czyli obrazy należące do 100 klas. To pilot. Powiększenie obrazu z 32×32 do 96×96 nie dodaje szczegółów, ale dostosowuje wejście do wybranego sposobu użycia sieci.
3. **Podział:** sieć uczy się na 40 000 obrazów. Na 5 000 innych sprawdzamy postęp i wybieramy checkpoint. Kolejne 5 000 oraz oficjalny test pozostawiamy nietknięte.
4. **Cztery GPU:** każda karta przetwarza 32 inne obrazy, a następnie procesy uzgadniają aktualizację wag. Razem daje to 128 obrazów na krok. To jeden wspólnie uczony model.
5. **Epoka:** jedno przejście przez treningowe dane, z pominięciem niepełnej końcówki batcha. Planujemy 200 epok; jakość i czas nie są jeszcze znane.
6. **Checkpoint:** plik zapisujący model i stan treningu. Pozwala wrócić do modelu lub kontynuować od zakończonej epoki.
7. **Reprodukowalność:** zapisujemy kod, środowisko Conda attention, wszystkie ustawienia, konkretne ID danych, losowość, logi, metryki i wagi. Osobny test sprawdza, czy powtórzony krótki przebieg daje identyczny model.

Test syntetyczny sprawdza wyłącznie kod, nie zdolność rozpoznawania rzeczywistych obrazów. Nie uznajemy go za wykonanie właściwego eksperymentu. Nie badamy jeszcze kompresji ani innowacyjności. Następny krok po zatwierdzeniu danych to przygotowanie ich podziału i krótka weryfikacja działania na rzeczywistych obrazach przed długim treningiem.

## Decyzja o pełnym treningu

2026-09-26: wybrano CIFAR-100 i zlecono pełne 200 epok. Uruchomienie na 4 GPU tworzy jeden model. Postęp śledzimy w stdout.log i metrics.jsonl, a zapis best.pt/last.pt pozwala zachować wynik i kontynuować trening po awarii.

## Co oznacza wynik? — 2026-09-28

Model zakończył 200 epok. W najlepszej epoce poprawnie rozpoznał 3342 z 5000 obrazów walidacyjnych, czyli 66,84%. Walidacja służyła do wyboru checkpointu, więc nie jest niezależnym testem końcowym. Na danych treningowych pod koniec trafność wynosiła około 99,95%: model znacznie lepiej radzi sobie z obrazami używanymi do nauki niż z nowymi. To oznaka przeuczenia. Mamy wytrenowany punkt odniesienia, ale nie dowód skuteczności kompresji ani powtarzalności pełnego treningu na innych seedach. Kolejny krok to przegląd krzywych uczenia i sprawdzenie zapisanego najlepszego modelu; nie rozpoczynamy automatycznie nowych treningów.
