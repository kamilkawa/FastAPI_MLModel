import logging
import sys
from pathlib import Path

# Ustalanie ścieżek
# TODO: przygotuj plik config z konfiguracja
BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
LOG_FILE_PATH = LOGS_DIR / "training.log"


def setup_logger(name: str = __name__) -> logging.Logger:
    """Konfiguruje loggera do zapisu w konsoli i pliku."""
    # Tworzenie katalogu na logi
    if not LOGS_DIR.exists():
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Handler konsolowy
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler plikowy
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
