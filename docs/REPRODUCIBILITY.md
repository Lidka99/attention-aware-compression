# Kontrakt reprodukowalności

## 1. Co musi być ustalone przed uruchomieniem

Każdy eksperyment ma ID, opis w Markdown, numer wersji, pytanie, hipotezę, pierwotną metrykę, kontrole, procedurę oraz zamrożoną konfigurację. Stan `draft` nie zezwala na interpretację wyników jako potwierdzających hipotezę. Etapy: `draft`, `frozen`, `running`, `completed`, `failed`, `inconclusive`, `stopped`.

Przed stanem `frozen` wszystkie pola oznaczone UNSET lub null muszą być rozstrzygnięte albo jawnie opisane jako niepotrzebne. W opisie musi być dokładna komenda, wersja kodu, lokalizacja zasobów i sposób ich uzyskania. Początkowe JSON-y są checklistą ustawień, nie kontraktem nieistniejącego programu.

## 2. Kod i środowisko

- Uruchamiać zamrożone eksperymenty z czystego drzewa Git; zapisać SHA commita, SHA-256 konfiguracji i pełną konfigurację po rozwinięciu wartości domyślnych.
- Zapisać dokładną komendę, katalog roboczy oraz niezbędne, jawnie wybrane zmienne środowiska. Nie zrzucać wszystkich zmiennych: mogą zawierać sekrety.
- Po wyborze stosu ML przygotować lockfile z dokładnymi wersjami, wersję Pythona oraz instrukcję instalacji. `pip freeze` jest dodatkowym zapisem stanu, nie pełną specyfikacją sterowników i sprzętu.
- Zapisać system, CPU/GPU, pamięć, sterownik GPU, CUDA/cuDNN, wersję frameworka, precision, ustawienia TF32/AMP i backend. Jeśli używamy obrazu kontenera, zapisać digest.
- Osobno ustawić seedy: trening, podział danych, kalibracja, wybór kandydatów, naprawa. Zapisać RNG Pythona, NumPy, CPU/GPU frameworka i inicjalizację workerów loadera, jeśli są używane.
- Ustalić algorytmy deterministyczne, wątki, liczbę workerów i kolejność danych. Każdy wyjątek od deterministyczności udokumentować. Ten sam seed nie gwarantuje identyczności między różnymi urządzeniami i wersjami bibliotek.

## 3. Dane i checkpointy

Zapisać źródło, wersję, checksum archiwum lub manifest plików, etykiety, identyfikatory próbek, transformacje, kolejność i politykę duplikatów. Zapisać rzeczywiste listy splitów wraz z SHA-256; sam seed losowania nie wystarcza.

Wydzielić role danych: kalibracja selektora, kompensacja/rekalibracja BN, walidacja rozwojowa, potwierdzenie zamrożonego kryterium i końcowy test. Budżetowanie lub współdzielenie zbiorów dla selektora i naprawy określić jawnie; wszystkie metody mają takie same zasoby. Zbiory oceny pozostają rozłączne z doborem i naprawą. Nie wykorzystywać wielokrotnie oglądanego testu jako niezależnego potwierdzenia.

Zapisać checkpoint SHA-256, dokładną architekturę i pochodzenie. Oddzielić niezależne modele od powtórzeń na jednym checkpointcie.

## 4. Struktura pojedynczego uruchomienia

`runs/<E-ID>/<UTC>_<git-short-sha>_<seed>_<unique-id>/`

Minimalne pliki:

- `manifest.json`: run-ID, E-ID, wersja protokołu, SHA kodu, czas UTC, status, exit code, komenda, powiązane uruchomienie przy wznowieniu, hashe wejść.
- `config.resolved.json`: pełna konfiguracja.
- `environment.json` i zapis pakietów.
- `stdout.log`, `stderr.log`: pełne logi.
- `metrics.json` oraz predykcje z ID próbek potrzebne do sparowanych porównań.
- `candidate_manifest.json`: dokładne grupy kanałów, kolejność zmian, konfiguracja kompensacji; nie tylko RNG seed.
- `artifacts.sha256`: lista hashy plików wynikowych.

Wyniki zapisywać do nowego katalogu; nie nadpisywać wcześniejszych. Status `completed` ustawiać dopiero po zakończeniu i zapisaniu wszystkich wymaganych artefaktów. Przy awarii zapisać błąd i status `failed`. Wznowienie ma osobny manifest i wskazuje poprzednie run-ID oraz zapisane stany modelu, optymalizatora i RNG, jeśli są potrzebne.

## 5. Raportowanie

Raport zawiera wszystkie zaplanowane warunki, także porażki, odchylenia od protokołu i koszt wykonania. Podawać licznik/mianownik accuracy, jednostki, rozmiar zbioru, paired differences oraz zmienność. Oddzielić niepewność wynikającą z obrazów, losowania kalibracji i treningu checkpointów. Powtórzenia kalibracji nie są niezależnymi modelami; kandydaci z jednego bloku nie są niezależnymi replikami całego badania.

Przed wynikiem zamrozić próg praktycznej korzyści, tolerancje numeryczne, sposób agregacji, budżet, kryteria przerwania i zasady wykluczania. Ustalić je po niezależnym sprawdzeniu szumu i kosztu, bez dobierania pod przewagę proponowanej metody. Raport `inconclusive` jest dopuszczalnym wynikiem.

## 6. Archiwizacja i odtworzenie

Każdy raport w Git wskazuje run-ID i trwały URI artefaktów z checksumami. Ustalić magazyn przed kosztownymi uruchomieniami. Nie publikować danych, których nie wolno redystrybuować; zapisać instrukcję ich uzyskania i odtworzenia splitów.

Przed uznaniem pipeline za odtwarzalny powtórzyć bazowy przebieg z nowego checkoutu i środowiska. Zdefiniować dwa poziomy: identyczność bitowa w kontrolowanym środowisku oraz zgodność metryk w zadeklarowanej tolerancji. Repozytorium z opisami dopiero przygotowuje tę możliwość — nie jest dowodem odtwarzalności wyników.
