# 2026-10-24_E012_folding_referencja

Status: **draft — nieuruchomiony**, oczekuje na E011 i testy operatora.

## Cel

Odtworzyć referencyjny folding dla CIFAR-10/ResNet18 bez treningu po
kompresji. Najpierw zgodność operatora i kontrola działania, dopiero później
alternatywne miary w osobnej próbie badawczej.

## Planowane wejście

Seed42, ostatnia epoka 200 z E011. Nie wybieramy najlepszego seeda po dev.
Konkretny hash checkpointu zostanie wpisany do zamrożonego configu po
zakończeniu i audycie E011. Seedy43–45 pozostają do późniejszych powtórzeń.

## Zakres do zamrożenia

- BASE oraz oryginalne grupowanie HKMeans/L2 na powiązanych parametrach.
- Budżety proponowane: 0/10/30/50% kanałów per grupa. Rzeczywista redukcja
  parametrów/FLOPs mierzona osobno; identyczne szerokości dla porównań.
- Wyniki przed korektą BN i po identycznej korekcie BN; BASE także po BN.
- Brak gradientów i optymalizatora. Dane do BN tylko z train E011,
  ocena na dev E011. Test nietknięty.
- Cztery niezależne zadania GPU0–3 dla czterech budżetów; nie są to seedy.
- Bramki: singleton/zero compression, residual shapes, niepuste klastry,
  skończone parametry i logity, zachowanie predykcji po zapisie modelu.

## Aktualny brak

Operator nie jest jeszcze zaimplementowany ani sprawdzony runtime.
Audyt opisuje konwencję macierzy sum/średnich i przebudowę BN:
[audyt](../../literature/AUDYT_CUT_LESS_FOLD_MORE.md).
E012 nie jest automatycznie uruchamiany po treningu. Przed wykonaniem:
implementacja, testy zgodności, zamrożenie configu/źródeł i commit.
