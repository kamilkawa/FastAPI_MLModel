# System Klasyfikacji Irysów (Iris Prediction API)

## Opis projektu

Projekt to usługa webowa (API) oparta na modelu uczenia maszynowego, służąca do klasyfikacji gatunków kwiatów (irysów) na podstawie ich cech morfologicznych. System wykorzystuje model regresji logistycznej wytrenowany na klasycznym zbiorze danych Iris.

Aplikacja rozwiązuje problem automatycznej identyfikacji gatunku irysa (*setosa*, *versicolor*, *virginica*) na podstawie czterech pomiarów:
1. Długości działki kielicha (sepal length)
2. Szerokości działki kielicha (sepal width)
3. Długości płatka (petal length)
4. Szerokości płatka (petal width)

## Wymagania i Instalacja

Projekt wykorzystuje narzędzie `uv` do zarządzania zależnościami i środowiskiem wirtualnym.

### Wymagania wstępne
- Python 3.10 lub nowszy
- Zainstalowane narzędzie `uv` (https://github.com/astral-sh/uv)

### Instalacja i przygotowanie środowiska

1. Sklonuj repozytorium:
   ```bash
   git clone <adres-repozytorium>
   cd zadanie-python
   ```

2. Zainstaluj zależności:
   ```bash
   uv sync
   ```

3. (Opcjonalnie) Wytrenuj model od nowa:
   Model jest już wytrenowany i znajduje się w katalogu `models/`. Jeśli chcesz go przeliczyć na nowo:
   ```bash
   uv run scripts/train_model.py
   ```
   Skrypt ten wygeneruje również wykres macierzy pomyłek (`models/confusion_matrix.png`) oraz logi treningu (`logs/training.log`).

## Uruchomienie serwera

Aby uruchomić serwer API lokalnie:

```bash
uv run uvicorn main:app --reload
```

Serwer domyślnie wystartuje pod adresem: `http://127.0.0.1:8000`.

## Instrukcja użycia

### Dokumentacja API
Po uruchomieniu serwera, pełna interaktywna dokumentacja (Swagger UI) dostępna jest pod adresem:
- **http://127.0.0.1:8000/docs**

### Endpointy

#### 1. Predykcja gatunku
- **URL:** `/predict`
- **Metoda:** `POST`
- **Opis:** Zwraca przewidywany gatunek irysa na podstawie przesłanych wymiarów.

**Przykładowe zapytanie (Request Body):**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Przykładowa odpowiedź:**
```json
{
  "species": "setosa"
}
```

#### 2. Status serwera
- **URL:** `/`
- **Metoda:** `GET`
- **Opis:** Sprawdzenie czy API działa.

## Informacje o modelu

- **Algorytm:** Logistic Regression (scikit-learn)
- **Zbiór danych:** Iris Dataset (150 próbek, 3 klasy)
- **Metryki:** Model osiąga ~95-100% dokładności na zbiorze testowym.
- **Walidacja:** Zastosowano 5-krotną walidację krzyżową (Cross-Validation) podczas treningu.

### Dane wejściowe (Input)
Model przyjmuje 4 wartości numeryczne (float):
- `sepal_length` (cm)
- `sepal_width` (cm)
- `petal_length` (cm)
- `petal_width` (cm)

### Dane wyjściowe (Output)
Model zwraca jedną z trzech klas (string):
- `setosa`
- `versicolor`
- `virginica`
