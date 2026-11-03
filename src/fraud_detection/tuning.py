"""Time-ordered randomized hyperparameter search."""

from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from fraud_detection.config import RANDOM_STATE


PARAMETER_DISTRIBUTIONS = {
    "Logistic regression": {
        "classifier__C": loguniform(0.01, 10.0),
        "classifier__class_weight": [None, "balanced"],
    },
    "Random forest": {
        "classifier__n_estimators": [100, 150],
        "classifier__max_depth": [None, 12, 20],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", 0.7],
    },
    "Extra trees": {
        "classifier__n_estimators": [100, 150],
        "classifier__max_depth": [None, 12, 20],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", 0.7],
    },
}


def tune_model(model_name, estimator, X_train, y_train, n_iter: int = 4):
    """Tune one selected model with expanding, time-ordered CV folds."""
    if model_name not in PARAMETER_DISTRIBUTIONS:
        raise ValueError(f"No tuning search space is configured for {model_name!r}.")

    search = RandomizedSearchCV(
        estimator=clone(estimator),
        param_distributions=PARAMETER_DISTRIBUTIONS[model_name],
        n_iter=n_iter,
        scoring="average_precision",
        cv=TimeSeriesSplit(n_splits=3),
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, float(search.best_score_), search.best_params_
