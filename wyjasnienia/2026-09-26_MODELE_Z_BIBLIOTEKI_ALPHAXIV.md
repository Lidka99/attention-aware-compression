# Modele poza ResNetem w zapisanych artykułach — 2026-09-26

Sprawdzono aktualną bibliotekę alphaXiv: folder Atencja w CNN — kompresja (5 prac) oraz Want to read (6 prac). Poniższe zestawienie obejmuje 7 prac bezpośrednio związanych z architekturami wizyjnymi; nie jest pełnym przeglądem czterech pozostałych artykułów. Oddzielamy własne eksperymenty autorów od modeli jedynie cytowanych.

| Artykuł | Modele poza ResNetem użyte w badaniach | Znaczenie dla nas |
|---|---|---|
| ECA-Net | MobileNetV2 z ECA oraz wariant z SE; ImageNet-1K | Bezpośredni przykład atencji kanałowej; §4.1–4.2, tabela 3. DenseNet i Inception w tabeli 4 to wyniki z innych publikacji. |
| Pick-or-Mix | MobileNet-v1, VGG-16, EfficientViT-M5 z PiX | Dynamiczne próbkowanie i mieszanie kanałów, nie zwykłe SE/ECA; m.in. §4.4 i tabele 6–7. |
| Cross-Layer Retrospective Retrieving via Layer Attention | EfficientNet-B0/B1; również DeiT, CeiT i PVTv2 | EfficientNet jest kandydatem CNN; dodawane MRLA to atencja między warstwami. §5.1, tabela 1, dodatek B.1.1. |
| Revisiting Dynamic Convolution via Matrix Decomposition | MobileNetV2, mnożniki szerokości 1.0, 0.5, 0.35 | Dynamiczna konwolucja i mieszanie kanałów; §5, tabele 1–4. Inny mechanizm niż samo ważenie map. |
| DANCE | C3D oraz R(2+1)D | Modele wideo, nie klasyfikacja pojedynczych obrazów; §6. Autorzy korzystają z pretreningu Kinetics przy UCF101/HMDB51. |
| Cut Less, Fold More | ViT-B/32, CLIP ViT-B/32, modele rodziny LLaMA 60M/130M | Alternatywy poza CNN. Część splotowa obejmuje ResNet18 i PreActResNet18. §4 i dodatek D. |
| Unified Response Geometry | ConvNeXt-B, EfficientNet-B3, MobileNetV2, ViT-S/16, Swin-T | Rozszerzone badanie przesiewowe sześciu rodzin; dodatek E, tabele 7–10. Potwierdzenie na wielu seedach dotyczy ResNet-50; nie przypisujemy go wszystkim rodzinom. |

Źródła:

- [ECA-Net](https://www.alphaxiv.org/abs/1910.03151)
- [Pick-or-Mix](https://www.alphaxiv.org/abs/2406.10935)
- [MRLA](https://www.alphaxiv.org/abs/2302.03985)
- [Dynamic Convolution](https://www.alphaxiv.org/abs/2103.08756)
- [DANCE](https://www.alphaxiv.org/abs/2603.17275)
- [Cut Less, Fold More](https://www.alphaxiv.org/abs/2602.18116)
- [Unified Response Geometry](https://www.alphaxiv.org/abs/2609.18239)

## Jak to przekłada się na nasz wybór?

Propozycja do dalszego doprecyzowania: EfficientNet-B0 jako pierwszy model. To mniejszy wariant rodziny z atencją SE, obecny w zapisanej pracy MRLA. Badalibyśmy jego atencję kanałową SE; nie dodajemy automatycznie MRLA, które zmienia pytanie na interakcje między warstwami. EfficientNet-B3 w pracy o geometrii daje dodatkowy punkt odniesienia dla kompresji tej rodziny, ale nie oznacza bezpośrednio tych samych wyników dla B0.

Alternatywa: MobileNetV2 z ECA, bezpośrednio zgodny z konstrukcją badaną w ECA-Net. Ułatwia badanie ECA, ale mieszanie kanałów i ich kolejność trzeba uwzględnić przy kompresji. Obie rodziny mają konwolucje depthwise, więc folding musi zachować zależności między kanałami i warstwami; nie wystarczy dowolnie usunąć kolumn macierzy wag.

ConvNeXt-B występuje w zapisanej pracy z 2026 roku, ale standardowy model nie ma modułu SE/ECA. Jest kandydatem do późniejszego sprawdzenia szerszej stosowalności metody, a nie automatycznie właściwym pierwszym modelem do pytania o atencję kanałową.

Data publikacji pracy o kompresji nie jest datą powstania użytej architektury. Modele takie jak EfficientNet czy MobileNetV2 nie są nowościami z 2026 roku. Nowość naszej pracy musi dotyczyć pytania i metody.

Wszystkie nasze modele trenujemy od losowych wag na 4 GPU. Nie odtwarzamy automatycznie protokołów z gotowymi checkpointami. Dobór danych i rozmiaru modelu wymaga jeszcze potwierdzenia zasobów. Nie uruchomiono treningu ani nie zmieniono biblioteki alphaXiv.
