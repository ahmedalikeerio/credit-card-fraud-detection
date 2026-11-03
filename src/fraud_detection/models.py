"""Preprocessing and baseline classifier definitions."""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.config import RANDOM_STATE


SCALED_COLUMNS = ["Time", "Amount"]


def build_pipeline(estimator, feature_columns) -> Pipeline:
    """Scale time and amount inside the pipeline; preserve other numeric inputs."""
    scaled = [column for column in SCALED_COLUMNS if column in feature_columns]
    passthrough = [column for column in feature_columns if column not in scaled]

    preprocessing = ColumnTransformer(
        transformers=[
            ("scale_time_amount", StandardScaler(), scaled),
            ("keep_anonymized_features", "passthrough", passthrough),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("preprocess", preprocessing),
        ("classifier", estimator),
    ])


def create_models(feature_columns) -> dict[str, Pipeline]:
    """Build a naive baseline and three supervised classification models."""
    return {
        "Prior baseline": build_pipeline(
            DummyClassifier(strategy="prior"), feature_columns
        ),
        "Logistic regression": build_pipeline(
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
            feature_columns,
        ),
        "Random forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=150,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            feature_columns,
        ),
        "Extra trees": build_pipeline(
            ExtraTreesClassifier(
                n_estimators=150,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            feature_columns,
        ),
    }
