# Czy ResNet jest przestarzały? — 2026-09-26

ResNet jest starszą architekturą: publikacja pochodzi z CVPR 2016. Nadal może być użytecznym modelem kontrolnym, ale w mojej ocenie oparcie szerokich wniosków o nowej metodzie kompresji wyłącznie na ResNecie byłoby zbyt wąskim sprawdzeniem.

Źródło: [Deep Residual Learning](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).

W naszych badaniach ważniejsza od daty jest zgodność modelu z pytaniem o atencję kanałową. Standardowy ResNet nie zawiera modułów SE ani ECA; wariant z taką atencją trzeba jawnie określić.

Nowsze od ResNet rodziny to m.in. EfficientNetV2 (2021), projektowany z myślą o efektywnym treningu, oraz ConvNeXt (2022) i ConvNeXt V2 (2023). To nie jest lista najnowszych architektur z 2026 roku. ConvNeXt V2 zawiera GRN, którego nie należy utożsamiać z atencją SE/ECA. Przeniesienie naszych pytań na ten model wymagałoby doprecyzowania mechanizmu.

Źródła: [EfficientNetV2](https://proceedings.mlr.press/v139/tan21a.html), [ConvNeXt](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf), [ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2).

Propozycja: dobierzmy główny model do mechanizmu atencji, który chcemy badać. ResNet z SE/ECA może pełnić rolę prostszego modelu kontrolnego; nie przesądzamy, że będzie modelem głównym. Przed kosztownym treningiem sprawdzimy budowę kandydata, dostępne dane i dokładny sprzęt. Sama liczba czterech GPU nie określa czasu treningu ani dostępnej pamięci.

Nowość doktoratu ma wynikać z metody i wyników, nie z wieku wybranej architektury. Wszystkie wybrane modele trenujemy samodzielnie od losowych wag na 4 GPU. Nie pobieramy gotowych checkpointów. Decyzja o konkretnej architekturze pozostaje otwarta.
