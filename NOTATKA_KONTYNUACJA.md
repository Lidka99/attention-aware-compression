# Notatka do kontynuacji — 2026-10-27

## Jednym zdaniem

Mamy obiecujący, zreplikowany wynik: **nierówna alokacja budżetu foldingu**
(TILT+0.30) bije jednolity folding bez fine-tuningu na CIFAR-10, ale to
nadal jeden dataset i adaptacja operatora — nie pełna reprodukcja artykułu.

## Co jest gotowe

- **E011:** 4 checkpointy ResNet18/CIFAR-10 od zera (seedy42–45),
  dev top-1 95,28/95,60/95,46/95,50%.
- **E012:** referencyjny folding bez fine-tuningu; po BN 94,20/89,30/70,50%
  dla 10/30/50% usuniętych kanałów; BASE po BN 95,48%.
- **E013:** przesiew alokacji na seed42; TILT+0.30 vs UNIFORM:
  **+0,58 / +4,34 / +10,70 pp** przy 30/50/70%.
- **E014:** replikacja na 4 seedach; TILT+0.30 vs UNIFORM:
  **+0,87 / +3,77 / +11,27 pp**; wszystkie 12 różnic seed×budżet dodatnie;
  MAG2 znacznie gorzej. To aktualny najmocniejszy wynik.

## Co wolno twierdzić

- Przy **tym samym globalnym budżecie parametrów**, na **CIFAR-10/ResNet18**,
  z **4 checkpointami** i oceną na **dev**, alokacja depth-aware poprawia
  jakość po korekcie BN bez aktualizacji wag.
- Efekt jest duży i spójny między seedami; nie wygląda na pojedynczy traf.

## Czego nie wolno jeszcze twierdzić

- Że jesteśmy lepsi od *Cut Less, Fold More* „1:1” — używamy **adaptacji**
  ich operatora, nie ich oficjalnego kodu/liczb.
- Że metoda działa na innych datasetach/architekturach — brak CIFAR-100,
  Tiny ImageNet, ImageNet.
- Że jest w pełni „data-free” — korekta BN korzysta z 45k train (bez etykiet,
  bez gradientów). „Bez fine-tuningu” tak; „bez danych” nie.
- Że TILT+0.30 jest optymalny — to jedna z zamrożonych hipotez; przy 30%
  TILT+0.15 wypadł minimalnie lepiej.

## Najbliższy krok (rekomendacja)

1. **E015: drugi dataset.** Najprościej: ResNet18/CIFAR-100 (ten sam operator,
   nowy trening 4 seedów) albo adaptacja operatora do EfficientNet z E001/E007.
   To jest decyzja do podjęcia na początku następnego czatu.
2. Dopiero potem: analiza mechanizmu (dlaczego wczesne warstwy są wrażliwsze)
   i ewentualne ulepszenie reguły alokacji.
3. Oficjalny test zostaje nietknięty do zamrożenia metody.

## Gdzie wszystko jest

- `STATUS_BADAN.md` — jedna strona statusu.
- `PODSUMOWANIE.md` w folderach E001–E014 — krótkie streszczenia.
- Pełne raporty: `WYNIKI.md`, `OPIS.md` w tych folderach.
- Kluczowe runy:
  - E011: `runs/E011/20261024T192205Z_018f3f2_seeds42-45_b2da2012`
  - E012: `runs/E012/20261024T235929Z_cca3cb2_a0b9b3f2`
  - E013: `runs/E013/20261027T093342Z_b60fd73_ca4f6700`
  - E014: `runs/E014/20261027T094707Z_267904d_6a0e93eb`
- Commity: E014 code `267904d`; dokumentacja `5647b6d`.

## Stan maszyny

- Conda `attention`; 4× RTX 2080 Ti wolne; ok. 8,7 GB wolnego dysku.
- E011–E014 completed; brak aktywnych zadań GPU.
- Nowe eksperymenty: nowy folder E0xx, nowy run, bez nadpisywania.
