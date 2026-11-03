"""Shared paths and settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
TARGET_COLUMN = "Class"
RANDOM_STATE = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
