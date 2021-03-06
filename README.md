# DEM-waterlevel

## Informacje ogólne
Repozytorium zawiera aktualnie rozwijany projekt, w ramach którego tworzony jest model uczenia maszynowego, pozwalający na odszumianie obszarów rzek na fotogrametrycznych cyfrowych modelach terenu oraz bezpośrednie uzyskanie wartości wysokości powierzchni wody wyrażonej w metrach nad poziomem morza.

## Publikacja
Wyniki prac zostaną opublikowane na konferencji EGU21. Abstrakt wystąpienia można znaleźć pod poniższym adresem:

https://meetingorganizer.copernicus.org/EGU21/EGU21-10266.html

## Dataset
Poniżej znajduje się wizualizacja pojedynczej próbki datasetu. Pierwszy obraz to zdjęcie true color (3 kanały RGB), drugi to uzyskany fotogrametrycznie cyfrowy model terenu (1 kanał zawierający wysokość wyrażoną w mnpm), natomiast trzeci to odszumiony cyfrowy model terenu (również 1 kanał zawierający wysokość wyrażoną w mnpm).

[![dataset.png](https://i.postimg.cc/bYQM3wDB/Microsoft-Teams-image.png)](https://postimg.cc/HjkBLHtw)

Dataset został stworzony automatycznie przy pomocy skryptu napisanego w języku Python odszumiającego powierzchnie wody na cyfrowych mapach terenu przy użyciu geodezyjnych danych wysokościowych uzyskanych z pomiarów in situ oraz dzielącego ortofotomapy i cyfrowe mapy terenu obszarów rzecznych na kwadraty przedstawiające obszary 10x10m. Skrypt korzysta z biblioteki ArcPy służącej do przetwarzania danych geoprzestrzennych dostarczonej wraz z pakietem oprogramowania ArcGis Desktop.

Ortofotomapy, cyfrowe mapy terenu oraz dane wysokościowe in situ zostały pozyskane w ramach kampanii pomiarowych prowadzonych w ramach trwającego doktoratu.

## Uruchomienie
Uruchomienie kodu na własnym komputerze wymaga wykonania następujących kroków przygotowujących:

1. Wprowadzenie danych neptune w pliku [config.cfg](ml/config.cfg).
2. Modyfikacja ścieżki do folderu roboczego w pliku [train.ipynb](ml/train.ipynb):
    ```Python
    #set workdir
    os.chdir("/content/drive/MyDrive/DEM-waterlevel/ml/")
    ```
3. Modyfikacja ścieżki do zbioru danych w pliku [train.ipynb](ml/train.ipynb):
    ```Python
    #dataset configuration
    dataset_dir = os.path.normpath("/content/drive/MyDrive/DEM-waterlevel/dataset")
    ```
