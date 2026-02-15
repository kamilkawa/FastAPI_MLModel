import os
from pathlib import Path

# Ścieżka bazowa projektu (poziom, gdzie jest pyproject.toml)
BASE_DIR = Path(__file__).resolve().parent.parent

# Ścieżka do katalogu z modelami
MODELS_DIR = BASE_DIR / "models"

# Nazwy plików
MODEL_FILENAME = "iris_model.joblib"
CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"

# Pełne ścieżki do plików
MODEL_PATH = MODELS_DIR / MODEL_FILENAME
CONFUSION_MATRIX_PATH = MODELS_DIR / CONFUSION_MATRIX_FILENAME

# Konfiguracja logowania
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"
