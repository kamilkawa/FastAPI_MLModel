import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Tuple, List
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator
from pathlib import Path
from logger_config import setup_logger

# Ustawienie ścieżek
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "iris_model.joblib"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"

logger = setup_logger(__name__)


def ensure_directories() -> None:
    """Tworzy niezbędne katalogi, jeśli nie istnieją."""
    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Utworzono katalog: {MODELS_DIR}")


def load_and_split_data() -> (
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]
):
    """Wczytuje zbiór Iris i dzieli go na zestawy treningowe i testowe."""
    logger.info("Wczytywanie zbioru Iris...")
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = list(iris.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info(f"Wymiar danych treningowych: {X_train.shape}")
    logger.info(f"Wymiar danych testowych: {X_test.shape}")

    return X_train, X_test, y_train, y_test, class_names


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> BaseEstimator:
    """Przeprowadza walidację krzyżową i trenuje model na pełnym zbiorze."""
    model = LogisticRegression(max_iter=200)

    # Walidacja krzyżowa (Cross-Validation)
    logger.info("Rozpoczynanie walidacji krzyżowej 5-fold...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

    logger.info(f"Wyniki walidacji krzyżowej: {cv_scores}")
    logger.info(
        f"Średnia dokładność (CV): {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})"
    )

    # Trenowanie modelu na całym zbiorze treningowym
    logger.info("Trenowanie modelu LogisticRegression na pełnym zbiorze treningowym...")
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: BaseEstimator, X_test: np.ndarray, y_test: np.ndarray, class_names: List[str]
) -> np.ndarray:
    """Ewaluuje model na zbiorze testowym i loguje wyniki."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Dokładność modelu na zbiorze testowym: {accuracy:.2f}")

    report = classification_report(y_test, y_pred, target_names=class_names)
    logger.info("Raport klasyfikacji:\n" + report)

    cm = confusion_matrix(y_test, y_pred)
    return cm


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
) -> None:
    """Generuje i zapisuje wykres macierzy pomyłek."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Macierz Pomyłek (Confusion Matrix)")
    plt.ylabel("Prawdziwa klasa")
    plt.xlabel("Przewidziana klasa")

    plt.savefig(CONFUSION_MATRIX_PATH)
    logger.info(f"Wykres macierzy pomyłek zapisany do {CONFUSION_MATRIX_PATH}")
    plt.close()


def save_model(model: BaseEstimator) -> None:
    """Zapisuje wytrenowany model do pliku."""
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Model zapisany do {MODEL_PATH}")


def main() -> None:
    ensure_directories()

    # 1. Przygotowanie danych
    X_train, X_test, y_train, y_test, class_names = load_and_split_data()
    # 2. Trenowanie
    model = train_model(X_train, y_train)
    # 3. Ewaluacja
    cm = evaluate_model(model, X_test, y_test, class_names)

    # 4. Zapis wyników
    save_confusion_matrix(cm, class_names)
    save_model(model)


if __name__ == "__main__":
    main()
