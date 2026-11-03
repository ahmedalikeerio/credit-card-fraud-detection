"""Command-line entry point for training and evaluating the fraud models."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_detection.config import DEFAULT_DATA_PATH, MODEL_DIR, OUTPUT_DIR
from fraud_detection.data import chronological_split, load_transactions
from fraud_detection.evaluation import (
    calculate_metrics,
    choose_f1_threshold,
    save_evaluation_plots,
)
from fraud_detection.models import create_models
from fraud_detection.tuning import tune_model


def run(data_path: Path) -> None:
    """Train candidate models, tune the leader, and evaluate the test period once."""
    data = load_transactions(data_path)
    split = chronological_split(data)
    print(f"Loaded {len(data):,} transactions from {data_path}")
    print(f"Fraud prevalence: {data['Class'].mean():.4%}")
    print(
        "Split sizes — "
        f"train: {len(split.X_train):,}, "
        f"validation: {len(split.X_validation):,}, "
        f"test: {len(split.X_test):,}"
    )

    models = create_models(split.X_train.columns.tolist())
    comparison_rows = []
    fitted_models = {}

    for name, estimator in models.items():
        fitted = clone(estimator)
        fitted.fit(split.X_train, split.y_train)
        validation_probabilities = fitted.predict_proba(split.X_validation)[:, 1]
        comparison_rows.append({
            "model": name,
            "validation_pr_auc": float(
                average_precision_score(split.y_validation, validation_probabilities)
            ),
            "validation_roc_auc": float(
                roc_auc_score(split.y_validation, validation_probabilities)
            ),
        })
        fitted_models[name] = fitted

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "validation_pr_auc", ascending=False
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_DIR / "validation_model_comparison.csv", index=False)
    print("\nValidation model comparison (ranked by PR-AUC):")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    candidate_names = [name for name in models if name != "Prior baseline"]
    best_name = max(
        candidate_names,
        key=lambda name: next(
            row["validation_pr_auc"] for row in comparison_rows if row["model"] == name
        ),
    )
    print(f"\nTuning selected candidate: {best_name}")
    tuned_model, best_cv_score, best_parameters = tune_model(
        best_name, models[best_name], split.X_train, split.y_train
    )
    validation_probabilities = tuned_model.predict_proba(split.X_validation)[:, 1]
    threshold = choose_f1_threshold(split.y_validation, validation_probabilities)
    pd.DataFrame({
        "true_label": split.y_validation.to_numpy(dtype=int),
        "fraud_probability": validation_probabilities,
    }).to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)

    test_probabilities = tuned_model.predict_proba(split.X_test)[:, 1]
    test_metrics = calculate_metrics(split.y_test, test_probabilities, threshold)
    test_metrics["selected_model"] = best_name
    test_metrics["validation_selected_threshold"] = threshold

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuned_model, MODEL_DIR / "fraud_detector.joblib")
    save_evaluation_plots(
        split.y_test, test_probabilities, threshold, OUTPUT_DIR
    )
    with (OUTPUT_DIR / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(test_metrics, file, indent=2)
    metadata = {
        "selected_model": best_name,
        "best_time_cv_pr_auc": best_cv_score,
        "best_parameters": best_parameters,
        "threshold_selected_on": "validation period; maximizes F1",
        "test_period_used_for_selection": False,
    }
    with (OUTPUT_DIR / "model_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nBest time-series CV PR-AUC:", f"{best_cv_score:.4f}")
    print("Selected threshold from validation:", f"{threshold:.6f}")
    print("\nFinal test metrics:")
    for metric, value in test_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    print(f"\nSaved metrics and plots in {OUTPUT_DIR}")
    print(f"Saved model in {MODEL_DIR / 'fraud_detector.joblib'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to Kaggle's creditcard.csv file.",
    )
    args = parser.parse_args()
    run(args.data)


if __name__ == "__main__":
    main()
