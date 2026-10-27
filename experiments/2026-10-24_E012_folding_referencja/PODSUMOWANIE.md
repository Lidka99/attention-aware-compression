# E012 — krótkie podsumowanie

**Cel:** odtworzyć referencyjny folding bez fine-tuningu na CIFAR-10/ResNet18.
**Co zrobiono:** operator per grupy kanałów, 10/30/50% usuniętych kanałów, BN bez gradientów, kontrola biasu fc.
**Wynik:** po BN **94,20 / 89,30 / 70,50%** dla 10/30/50%; BASE po BN 95,48%.
**Wniosek:** działający punkt odniesienia; adaptacja reguł, nie reprodukcja artykułu 1:1.
**Szczegóły:** [WYNIKI.md](WYNIKI.md), [OPIS.md](OPIS.md).
