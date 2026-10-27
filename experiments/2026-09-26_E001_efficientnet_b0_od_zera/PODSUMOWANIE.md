# E001 — krótkie podsumowanie

**Cel:** baseline EfficientNet-B0 z SE, od zera na CIFAR-100.
**Co zrobiono:** 200 epok, 4 GPU, seed42; audyt checkpointu i predykcji.
**Wynik:** dev top-1 **66,84%** (3342/5000), najlepsza epoka 177; silne przeuczenie.
**Wniosek:** baseline odniesienia dla E002–E004; nie jest wynikiem kompresji.
**Szczegóły:** [WYNIKI.md](WYNIKI.md), [OPIS.md](OPIS.md).
