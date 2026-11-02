# E017 — plan etapów

## Pytanie
Czy przy **identycznym operatorze foldingu i identycznych szerokościach
warstw** inna reprezentacja/miara grupowania kanałów poprawia jakość
bez fine-tuningu?

## Etap 1 — benchmark miar (ten eksperyment)
- 2 datasety (CIFAR-10, CIFAR-100), seed42, 10/30/50% kanałów.
- 7 wariantów: reference (E012), cosine, standardized, bn_effective,
  random1–3.
- 42 zadania, 4 GPU, brak treningu wag.
- Kryterium: kandydat ≥ +0,50 pp vs reference **i** vs średnia random,
  przy 30% i 50% kanałów na obu zbiorach.
- Wynik spełniający kryterium = **rekomendacja replikacji**, nie publikacja.

## Bramka G1 — rozstrzygnięta
- **Brak spełnienia:** zapisano wynik negatywny. Jedyny trop (cosine,
  CIFAR-10/50%) sprawdzono w E018 na seedach 43–45: **nie replikuje się**
  (+0,47 pp, 2/3 seedów; przy 30% gorsza). Trop zamknięty.
- Nie stroić miar po wynikach. Nowa miara wymaga mechanizmu i nowego
  protokołu (np. atencja/SE), a nie kolejnego wariantu tej rodziny.

## Etap 2 — miara uwzględniająca atencję (osobny eksperyment)
Po Etapie 1 i audycie literatury:
- model z SE (np. ResNet18+SE) trenowany od zera w tym repo;
- grupowanie według efektywnego działania kanału z bramką atencji;
- porównanie z reference, cosine i random na tych samych szerokościach;
- jawne rozdzielenie „bez fine-tuningu” od „z danymi kalibracyjnymi”.
To nowy eksperyment z własnym protokołem; nie część E017.

## Etap 3 — skala
Dopiero po potwierdzeniu na wielu seedach: Tiny ImageNet, potem ImageNet.
Publiczne wagi (np. Keras Applications) wymagają świadomej zmiany ustalenia
„bez publicznych wag” i osobnej decyzji; nie wprowadzamy ich po cichu.

## Czego ten plan nie obiecuje
Publikacji ani nowości metody. Najpierw musi istnieć powtarzalna korzyść
w kontrolowanym porównaniu.
