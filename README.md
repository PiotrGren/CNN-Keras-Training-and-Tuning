**Choose your language / Wybierz język**

[EN](#english) / [PL](#polski)

#### English

# CNN Keras Natural Spaces Classification Training and Tuning

## Table of Contents

1. [Description](#description)
2. [Dataset](#dataset)
3. [Usage](#usage)
4. [Notes](#notes)
5. [License](#license)
6. [Contributing](#contributing)
7. [Author](#author)

## Description

This repository contains two scripts for training and tuning a Convolutional Neural Network (CNN) using Keras. The scripts are:

1. `CNN_Natural_Space_Images_Classification.py`
2. `CNN Model Tuning.py`

### Code Analysis

#### `CNN_Natural_Space_Images_Classification.py`

This script focuses on training a CNN model from scratch using Keras. The main steps involve:

1. Loading and preprocessing the dataset.
2. Defining the CNN architecture.
3. Compiling the model with appropriate loss function and optimizer.
4. Training the model with the training data.
5. Evaluating the model with the test data.
6. Saving the trained model for future use.

#### `CNN Model Tuning.py`

This script is designed for improving CNN model created in previous script using VGG16 convolutional layers. The main steps include:

1. Loading preprocessed data from `.npy` files.
2. Loading an VGG16 model.
3. Freezing certain layers of the VGG16 model to retain previously learned features (see the freeze.txt file for details).
4. Adding output layers based on the model from the first script.
5. Compiling the model with a suitable optimizer and loss function.
6. Training the model.
7. Evaluating the model to ensure improved performance.
8. Saving the fine-tuned/improved model.

**Both scripts require the dataset to be correctly organized and preprocessed as per the instructions to function correctly.**

## Dataset

The dataset used in these scripts is compressed in a zip file named **Dataset.zip** located in the repository. If you want to use the dataset as in the provided code, you need to unzip the file first. Also in the second zip file you will find dataset encoded into numpy arrays and exported into .npy files. If you want to run the first script and create the encoded dataset files yourself, you don't need to download the **TransformedData.zip** file. You can also skip the code related to exporting the encoded dataset in the first script and test the second script created to tune/improve the model, using the one provided in the repository.

## Usage

### 1. `CNN_Natural_Space_Images_Classification.py`

This script is used to train a CNN model from scratch using provided dataset. CNN model is created using tensorflow keras model.

#### Steps to run the script:

1. Unzip the dataset (**Dataset.zip**).
2. Ensure the dataset is in the correct directory as expected by the script.
3. Run the script using Python.

```bash
python CNN_Natural_Space_Images_Classification.py
```

### 2. `CNN Model Tuning.py`

This script is used to fine-tune or, as one prefers to say, improve the CNN model. The dataset, in this case, is transformed into numpy arrays and exported to .npy files, which are contained in the aforementioned zip folder **TransformedData.zip**.

#### Steps to run the script:

1. Unzip the dataset, including the .npy files (**TransformedData.zip**) if you did not create it yourself in the first script.
2. Ensure the **.npy** files are in the correct directory as expected by the script.
3. Run the script using Python.

```bash
python `CNN Model Tuning.py`
```

## Notes

1. Make sure you have the necessary dependencies installed. You can install the required packages using:

```bash
pip install -r requirements.txt
```

2. The structure of the dataset and how it should be organized after unzipping should match the expected format in the scripts. Make sure that dataset is unziped correctly, creating a properly organized dataset folder and adjust the paths in the scripts if necessary.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Feel free to submit issues or pull requests if you have suggestions or improvements.

## Author

Piotr Greń - Main developer - https://github.com/PiotrGren

#### Polski

# CNN Keras Natural Spaces Classification Training and Tuning

## Spis treści

1. [Opis](#opis)
2. [Zbiór danych](#zbiór-danych)
3. [Wykorzystanie](#wykorzystanie)
4. [Uwagi](#uwagi)
5. [Licencja](#licencja)
6. [Wkład](#wkład)
7. [Autor](#autor)

## Opis

To repozytorium zawiera dwa skrypty do trenowania i dostrajania konwolucyjnej sieci neuronowej (CNN) przy użyciu Keras. Skrypty są następujące:

1. `CNN_Natural_Space_Images_Classification.py`
2. `CNN Model Tuning.py`

### Analiza kodu

#### `CNN_Natural_Space_Images_Classification.py`

Ten skrypt skupia się na trenowaniu modelu CNN od podstaw przy użyciu Keras. Główne kroki obejmują:

1. Ładowanie i wstępne przetwarzanie zbioru danych.
2. Definiowanie architektury CNN.
3. Kompilacja modelu z odpowiednią funkcją straty i optymalizatorem.
4. Trenowanie modelu z wykorzystaniem danych treningowych.
5. Ocena modelu na podstawie danych testowych.
6. Zapisanie wytrenowanego modelu do wykorzystania w przyszłości.

#### `CNN Model Tuning.py`

Ten skrypt jest przeznaczony do ulepszania modelu CNN utworzonego w poprzednim skrypcie przy użyciu warstw konwolucyjnych modelu VGG16. Główne kroki obejmują:

1. Ładowanie wstępnie przetworzonych danych z plików `.npy`.
2. Ładowanie modelu VGG16.
3. Zamrożenie niektórych warstw modelu VGG16 w celu zachowania wcześniej nauczonych funkcji (szczegóły w pliku freeze.txt).
4. Dodanie warstw wyjściowych na podstawie modelu z pierwszego skryptu.
5. Kompilacja modelu z odpowiednim optymalizatorem i funkcją straty.
6. Trenowanie modelu.
7. Ocena modelu w celu zapewnienia lepszej wydajności.
8. Zapisanie dopracowanego/ulepszonego modelu.

**Oba skrypty do poprawnego działania wymagają prawidłowego zorganizowania i wstępnego przetworzenia zbioru danych zgodnie z instrukcjami.**

## Zbiór danych

Zestaw danych używany w tych skryptach jest skompresowany w pliku zip o nazwie **Dataset.zip** znajdującym się w repozytorium. Jeśli chcesz użyć zestawu danych w dostarczonym kodzie, musisz najpierw rozpakować plik. Również w drugim pliku zip znajdziesz zestaw danych zakodowany w tablicach numpy i wyeksportowany do plików .npy. Jeśli chcesz uruchomić pierwszy skrypt i samodzielnie utworzyć zakodowane pliki zbiorów danych, nie musisz pobierać pliku **TransformedData.zip**. Możesz również pominąć kod związany z eksportowaniem zakodowanego zestawu danych w pierwszym skrypcie i przetestować drugi skrypt utworzony w celu dostrojenia / ulepszenia modelu, używając tego dostarczonego w repozytorium.

## Wykorzystanie

### 1. `CNN_Natural_Space_Images_Classification.py`

Ten skrypt służy do trenowania modelu CNN od podstaw przy użyciu dostarczonego zestawu danych. Model CNN jest tworzony przy użyciu modelu keras tensorflow.

#### Kroki uruchamiania skryptu:

1. Rozpakuj zestaw danych (**Dataset.zip**).
2. Upewnij się, że zestaw danych znajduje się we właściwym katalogu, zgodnie z oczekiwaniami skryptu.
3. Uruchom skrypt przy użyciu języka Python.

```bash
python CNN_Natural_Space_Images_Classification.py
```

### 2. `CNN Model Tuning.py`

Skrypt ten służy do dostrajania lub też, ulepszania modelu CNN. W tym przypadku zbiór danych jest przekształcany w tablice numpy i eksportowany do plików .npy, które znajdują się we wspomnianym folderze zip **TransformedData.zip**.

#### Kroki uruchamiania skryptu:

1. Rozpakuj zestaw danych, w tym pliki .npy (**TransformedData.zip**), jeśli nie utworzyłeś ich samodzielnie w pierwszym skrypcie.
2. Upewnij się, że pliki **.npy** znajdują się we właściwym katalogu, zgodnie z oczekiwaniami skryptu.
3. Uruchom skrypt przy użyciu języka Python.

```bash
python `CNN Model Tuning.py`
```

## Uwagi

1. Upewnij się, że masz zainstalowane niezbędne zależności. Wymagane pakiety można zainstalować za pomocą

```bash
pip install -r requirements.txt
```

2. Struktura zbioru danych i sposób jego organizacji po rozpakowaniu powinny być zgodne z oczekiwanym formatem w skryptach. Upewnij się, że zestaw danych został prawidłowo rozpakowany, tworząc odpowiednio zorganizowany folder zestawu danych i w razie potrzeby dostosuj ścieżki w skryptach.

## Licencja

Ten projekt jest dostępny na licencji MIT.Szczegóły można znaleźć w pliku LICENSE.

## Wkład

Zachęcamy do przesyłania zgłoszeń lub pull requestów, jeśli masz sugestie lub ulepszenia.

## Autor

Piotr Greń - Main developer - https://github.com/PiotrGren
