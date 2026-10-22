# Cut Less, Fold More — audyt źródeł i plan odtworzenia

Data: 2026-10-22. Status: audyt statyczny kodu; nie wykonano eksperymentu.

## Cel ustalony w rozmowie

Sprawdzić, czy inne miary podobieństwa i grupowania poprawiają accuracy
po foldingu przy tym samym rozmiarze sieci, bez fine-tuningu.
Najpierw odtwarzamy referencję, potem zmieniamy grupowanie.
Potwierdzenie TAYLORA z E010 nie jest obecnym następnym krokiem.

## Źródła

- Artykuł: https://arxiv.org/abs/2602.18116 — *Cut Less, Fold More:
  Model Compression through the Lens of Projection Geometry*.
- Repo: https://github.com/osaukh/folding_as_projection
- Audytowany HEAD: `694691556e8314a835178a741938dd3d4d80c171`.
- Pliki poniżej odnoszą się do tego repo i wersji; przed wykonaniem należy
  pobrać tę wersję i zapisać lokalne hashe. Audyt nie zastępuje testu runtime.

## Co rzeczywiście robi implementacja CNN

| Element | Plik | Ustalenie |
|---|---|---|
| Architektura | `models/resnet.py` | Wariant ResNet18 dla CIFAR: wejściowe 3×3, stride 1, bez maxpool; nie zakładamy zgodności z wagami ImageNet |
| Pipeline | `pipelines/test_resnet_compression.py` | Przetwarza checkpointy i stopnie kompresji; metody fold, mag-l1, mag-l2 |
| Folding | `compression/fold.py` | `ResNet18_ModelFolding`, grupy osi z `get_axis_to_perm_ResNet18(override=False)` |
| Cechy | `utils/weight_clustering.py` | `concat_weights`: konkatenacja spłaszczonych parametrów związanych wspólną osią kanałów, z pominięciem running statistics i identity_transform |
| Grupowanie | te same pliki | HKMeans, seed 42, n_init=1, max_iter=10; wywołanie ResNet jawnie ustawia PCA=True, normalize=False |
| Korekta BN | `utils/tune_utils.py` | `repair_bn`: train mode i forward na danych bez etykiet pod `torch.no_grad()`; brak optymalizatora |
| Trening źródłowy | `train/cifar10_learn_configurable.py`, `train/run.sh` | Receptury i siatka treningu dostępne, checkpointów ResNet18 brak w repo |

### Istotne szczegóły

1. Cechy grupowania obejmują powiązania między warstwami. Zastąpienie ich
   pojedynczym filtrem zmienia reprezentację, nie tylko miarę odległości.
2. `normalize=True` oznacza w tej implementacji RobustScaler, nie normalizację
   każdego filtra do jednostkowej długości. Nie jest to gotowy wariant cosine.
3. W `merge_channel_clustering` macierz przypisań ma wiersze znormalizowane
   do średniej. Kod dla osi 0 stosuje `diag(1/diag(M M^T)) M`, a dla osi 1
   stosuje `M`. Dla niepustego klastra daje to odpowiednio sumę i średnią.
   Trzeba sprawdzić cały przebieg razem z przebudową BN w `base_resnet.py`;
   nie wolno zastąpić go intuicyjnym operatorem mean/sum bez testu zgodności.
4. Obsługa mniejszej liczby klastrów niż żądana wymaga kontroli: pusty klaster
   może powodować problem z odwrotnością przekątnej `M M^T`. Sprawdzamy
   niepuste klastry, skończone parametry i faktyczne szerokości warstw.
5. Funkcja nazywana `repair_bn` nie zeruje jawnie buforów BN i nie ustawia
   momentum. Odtwarzamy jej rzeczywiste zachowanie. Nie utożsamiamy bez testu
   tej funkcji z pełną implementacją algorytmu REPAIR z innego repo.
6. Bez fine-tuningu oznacza tutaj brak aktualizacji gradientowej. Korekta BN
   nadal korzysta z danych. Raportujemy osobno przed i po korekcie.

## Dostępność zasobów i ograniczenia odtworzenia

- Repo nie zawiera checkpointów zwykłego ResNet18/CIFAR-10 ani release z nimi.
  Skrypt pojedynczej oceny wskazuje nieobecny lokalny plik `.pth`.
- Artykuł wskazuje również zewnętrzne checkpointy PreActResNet18. To inna
  architektura: nie zamieniamy jej po cichu na ResNet18.
- CIFAR-10 jest pobierany przez torchvision. Wagi ImageNet nie zastępują
  checkpointu CIFAR-10 z jego głowicą i strukturą wejściową.
- `requirements.txt` jest szerokim zrzutem środowiska (m.in. torch 2.2.2,
  torchvision 0.17.2, scikit-learn 1.5.0, hartigan-kmeans 0.0.6, dodatkowe
  zależności ViT). Minimalne środowisko CNN trzeba sprawdzić osobno.
- W kodzie występują ścieżki względne i domyślne `device='cuda'`.
  Uruchomienie czterech zadań wymaga jawnego mapowania urządzeń.
- W HEAD część diagnostyki jest zakomentowana, a historyczne logi ją zawierają.
  Nie obiecujemy odtworzenia logów 1:1 bez checkpointu i wersji ich wykonania.

## Najbliższy etap — odtworzenie referencji (szkic, nie frozen)

1. Ustalić konkretny checkpoint ResNet18/CIFAR-10: źródło, licencja,
   SHA-256, architektura, preprocessing i oczekiwana accuracy.
   Jeśli wag autorów nie ma, odtworzyć trening według ich receptury;
   raportować to jako reimplementację protokołu na nowym checkpointcie,
   nie odtworzenie dokładnej liczby z artykułu.
2. Uruchomić testy operatora: zerowa kompresja, singletony, klastry
   identycznych kanałów, zgodność residuali, niepuste klastry, skończone
   parametry, zgodność liczby kanałów i predykcji przed/po serializacji.
3. Wykonać referencję na jednym checkpointcie, bez fine-tuningu.
   Proponowana pierwsza siatka: 0/10/30/50% usuwanych kanałów per grupa.
   To nie są procenty redukcji parametrów całej sieci — mierzymy je osobno.
4. Każdy stopień kompresji uruchomić jako niezależne zadanie na jednym z
   czterech GPU, ze wspólnego oryginalnego checkpointu. Zapisać mapowanie
   ratio→GPU; klasteryzacja może wykonywać się na CPU. To nie są 4 seedy.
5. Dla każdego modelu zapisać accuracy/NLL przed i po identycznej korekcie
   BN, liczbę parametrów, FLOPs, czas kompresji, predykcje, grupy i hashe.
   Dodać kontrolę BASE po samej korekcie BN, aby oddzielić jej efekt.
6. Zamrozić protokół dopiero po ustaleniu wag, danych, środowiska i kontroli.
   Obecny audyt nie tworzy jeszcze wykonanego ani zamrożonego E011.

## Następny etap — alternatywne grupowanie

Referencja musi najpierw przejść kontrole. Następnie porównujemy mały zestaw:
oryginalne HKMeans/L2, wariant kierunkowy, wariant korelacyjny i losowe
grupowanie. Dobór algorytmu i definicji odległości wymaga osobnego protokołu.

- Główne porównanie: kandydat minus oryginalny folding, nie minus RANDOM.
- Te same cechy wejściowe, szerokości warstw, operator scalenia i procedura BN.
- Zmiany miary, reprezentacji i algorytmu traktujemy jako osobne ablacje.
- Cosine i L2 na jednostkowo znormalizowanych wektorach mogą prowadzić do
  równoważnych odległości; nie przedstawiamy takich wariantów jako niezależnej
  nowości. Korelacja wymaga jawnego centrowania i obsługi zerowych norm.
- Dane do korekty BN i wyboru wariantów oddzielamy od końcowej oceny.
  Oficjalnego testu nie używamy do dobierania miar; ustalamy dev przed biegiem.
  Porównanie do artykułu na oficjalnym teście wymaga osobnej, zamrożonej oceny.
- Dopiero po przesiewie: powtórzenia na niezależnych checkpointach, następnie
  CIFAR-100, Tiny ImageNet i ImageNet. Zmienność klasteryzacji oraz treningu
  źródłowego raportujemy oddzielnie.

## Stan na koniec audytu

Kod referencyjny i miejsca modyfikacji rozpoznane. Brak zweryfikowanego
checkpointu ResNet18/CIFAR-10 i uruchomienia runtime. Następna czynność:
ustalenie źródła checkpointu i minimalnego środowiska, następnie techniczne
testy operatora. Nie mamy jeszcze wyniku naukowego dla nowego kierunku.
