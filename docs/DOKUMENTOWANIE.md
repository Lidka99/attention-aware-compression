# Jak dokumentujemy przebieg eksperymentów

Ustalenie z rozmowy, 2026-09-29: oddzielamy **eksperyment naukowy**, jego
**uruchomienia** oraz **analizy**. Wykresy istniejącego treningu i audyt jego
checkpointu należą do tego samego eksperymentu, nie dostają nowego numeru E.

## Proste zasady

- `experiments/RRRR-MM-DD_E001_nazwa/`: pytanie, protokół, kod, wyniki,
  późniejsze analizy i wykresy. Nowa próba badawcza z innym pytaniem lub
  zmienionymi warunkami dostaje kolejny numer; powtórzenia tego samego
  protokołu są osobnymi uruchomieniami tego eksperymentu.
- `runs/E001/<run-ID>/`: duże artefakty konkretnego uruchomienia poza Git.
  Audyty: `runs/E001/audits/<run-ID>/`. Każde uruchomienie ma nowy katalog,
  dokładną komendę, wersję kodu/środowiska, wejścia z hashami, status i logi.
- Wykresy: `experiments/.../wykresy/<run-ID>/`, wraz z kodem tworzenia,
  identyfikatorem źródłowych metryk i ich hashem. Małe wykresy i raporty
  zapisujemy w Git. Checkpointy, dane i środowiska pozostają poza Git.
- Nie zmieniamy zakończonego protokołu ani źródłowych wyników. Dopisujemy
  datowany raport lub osobny wersjonowany protokół analizy.
- Status protokołu (`draft`/`frozen`) to co innego niż status wykonania
  (`running`/`completed`/`failed`). Samo przygotowanie skryptu nie oznacza
  wykonania pomiaru. Raport podaje dowody: run-ID, wynik, logi i hashe.

Każdy raport odpowiada na pięć pytań: **co sprawdzaliśmy, na czym, jak,
co wyszło i co z tego wynika**. Dodatkowo podaje ograniczenia i jeden
proponowany kolejny krok. Oddzielamy obserwacje od hipotez o ich przyczynach.

Na poziomie projektu: [AKTUALNY_PLAN.md](../AKTUALNY_PLAN.md) pokazuje, gdzie
jesteśmy, a [spis eksperymentów](../experiments/README.md) prowadzi do szczegółów.
Dokumenty historyczne pozostają oznaczone jako historyczne, nie konkurują
z aktualnym planem. Nie jest potrzebna osobna platforma do śledzenia eksperymentów
na obecnym etapie.

## Uwaga o alphaXiv

W tej sesji nie wywoływano alphaXiv. W `PLAN_DZIALANIA.md` znajduje się
historyczna informacja, że nie udało się zapisać pracy ADSC przez MCP pod
adresem wydawcy. Nie ma tu logu błędu pozwalającego ustalić jego przyczynę.
Nie jest to błąd treningu ani oceny modelu. Jeżeli chodzi o inny komunikat
widoczny w interfejsie, potrzebna jest jego treść, aby go rozpoznać.
