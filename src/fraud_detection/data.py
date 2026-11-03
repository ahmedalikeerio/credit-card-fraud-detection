"""Dataset loading, validation, and chronological splitting."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fraud_detection.config import (
    DEFAULT_DATA_PATH,
    TARGET_COLUMN,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)


@dataclass
class DataSplit:
    """Chronologically ordered train, validation, and test partitions."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def load_transactions(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and validate the Kaggle credit-card transaction CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download creditcard.csv from the Kaggle "
            "dataset linked in the README and place it in data/raw/."
        )

    data = pd.read_csv(path)
    required = {"Time", "Amount", TARGET_COLUMN}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if data.empty:
        raise ValueError("The dataset is empty.")

    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="raise").astype(int)
    labels = set(data[TARGET_COLUMN].unique())
    if not labels.issubset({0, 1}) or labels != {0, 1}:
        raise ValueError("The Class column must contain both labels: 0 and 1.")

    feature_columns = data.columns.difference([TARGET_COLUMN])
    non_numeric = data[feature_columns].select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"Expected numeric model features; found: {non_numeric}")

    return data.sort_values("Time", kind="stable").reset_index(drop=True)


def chronological_split(
    data: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
) -> DataSplit:
    """Split rows in time order, reserving the latest rows for final testing."""
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Train and validation fractions must be greater than zero.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than one.")
    if "Time" not in data.columns or TARGET_COLUMN not in data.columns:
        raise ValueError("Input data must contain Time and Class columns.")

    n_rows = len(data)
    train_end = int(n_rows * train_fraction)
    validation_end = train_end + int(n_rows * validation_fraction)
    if train_end == 0 or validation_end >= n_rows:
        raise ValueError("Not enough rows for the requested split fractions.")

    train = data.iloc[:train_end]
    validation = data.iloc[train_end:validation_end]
    test = data.iloc[validation_end:]

    for name, part in (("training", train), ("validation", validation), ("test", test)):
        if part[TARGET_COLUMN].nunique() < 2:
            raise ValueError(
                f"The chronological {name} split has only one class. "
                "Review the split boundaries before training."
            )

    def features_and_target(part: pd.DataFrame):
        return part.drop(columns=[TARGET_COLUMN]), part[TARGET_COLUMN]

    X_train, y_train = features_and_target(train)
    X_validation, y_validation = features_and_target(validation)
    X_test, y_test = features_and_target(test)

    return DataSplit(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
    )
