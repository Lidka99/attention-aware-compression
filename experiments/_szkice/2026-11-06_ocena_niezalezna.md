# Niezależna ocena po E034 — propozycja zakresu

Status: **historyczny szkic z 2026-11-06, zastąpiony przez ukończony E035**.
Zatwierdzony zakres rozszerzono o Tiny ImageNet. Obowiązują protokół i wyniki
w `experiments/2026-11-07_E035_ocena_niezalezna/`; poniżej zachowana propozycja.
Przed otwarciem oficjalnego testu uzgodnić zakres z użytkowniczką,
przenieść zatwierdzony protokół do osobnego eksperymentu i zapisać commit.

## Pytanie

Czy ustalona przewaga TILT+0.30 nad UNIFORM na dev utrzymuje się na
nieużywanym oficjalnym teście, przy dopasowanym budżecie parametrów,
bez fine-tuningu? To potwierdzenie wcześniej wybranej reguły, nie search.

## Proponowany zakres

- ResNet18 i PreActResNet18; CIFAR-10 i CIFAR-100; seedy42–45.
- Istniejące checkpointy E011/E015/E034, bez nowego treningu i selekcji seedów.
- Budżety30/50/70% redukcji parametrów; ramiona UNIFORM i TILT+0.30.
- Dokładne profile z E031 (ResNet18/30/70), E029 (ResNet18/50) i E034
  (PreActResNet18). Przed runem sprawdzić zgodność checkpointów i zapisać
  źródłowe ścieżki, szerokości, liczby parametrów oraz SHA256.
- Niezmieniony operator upstream6946915 i zachowany bias klasyfikatora.
- Kalibracja tylko na dotychczasowym train A4096 crop/flip, jak w audytach.
  Primary CMA/reset; secondary oryginalne repair_bn. PRE jako diagnostyka.
- Oficjalne testy CIFAR po10000 obrazów. Dane testowe wyłącznie do oceny,
  bez kalibracji, wyboru profili, progów, checkpointów czy wariantu BN.
- Tiny i rezerwa CIFAR-100 poza tym proponowanym zakresem.

## Wykonanie

4GPU, jeden seed na GPU, konfiguracje sekwencyjnie wewnątrz workera.
2 architektury×2 zbiory×3 budżety×4 seedy×2 profile×3 stany =288 ocen,
plus48 kontroli teacher. Przed testem kontrola CPU oraz odtworzenie dev
na istniejącym protokole, bez dostrajania. Błąd wykonania wymaga jawnego
raportu; zachować przerwane artefakty, nie wybierać korzystniejszego runu.
Koszt i limity doprecyzować przed zamrożeniem po audycie implementacji.

## Proponowane kryteria ustalone przed testem

1. Integralność: pełna siatka,4GPU/exit0, zgodność profili i hashy,
   niepuste klastry, skończone metryki, brak gradientowych aktualizacji.
2. Kierunek: średnia sparowana TILT−UNIFORM>0 i≥3/4 dodatnich seedów
   w każdej z12 komórek architektura×zbiór×budżet po CMA.
3. Wielkość efektu raportowana osobno: przy50/70% ≥+2pp i≥3/4,
   bez zastępowania negatywnych komórek średnią po zbiorach.
   E034 zachowuje swoje wcześniejsze kryteria; E031 nie zostaje
   reinterpretowany jako spełniający mocniejszy próg.
4. Secondary: te same kontrasty po repair_bn, NLL, bezwzględne top1
   i strata względem teacher po tej samej BN. PRE bez werdyktu primary.
5. Raport: per-seed, średnia i SD sparowanych różnic; 4 seedy to4
   powtórzenia treningu, nie liczba obrazów testowych. Progi opisowe
   nie są testami istotności. Wyniki dev i test pokazać obok siebie.

## Decyzja po ocenie

Raportować każdą komórkę, w tym ewentualną porażkę CIFAR-100/ResNet18.
Nie dostrajać reguły na wynikach testowych. Po obejrzeniu wyników test nie
jest już niewidzianą walidacją nowych wariantów. Ten eksperyment nie
rozstrzyga optymalności alokacji, mechanizmu preaktywacji ani przewagi
nad wszystkimi metodami kompresji; służy ocenie zakresu twierdzenia
o TILT względem UNIFORM. Nowy trening wymaga osobnej wyraźnej zgody.
