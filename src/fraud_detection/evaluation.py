"""Threshold selection, classification metrics, and evaluation plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import ConfusionMatrixDisplay


def choose_f1_threshold(y_true, probabilities) -> float:
    """Choose the threshold that maximizes F1 on validation data."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        return 0.5

    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    return float(thresholds[int(np.argmax(f1_values))])


def calculate_metrics(y_true, probabilities, threshold: float) -> dict[str, float | int]:
    """Calculate threshold-free and threshold-dependent binary metrics."""
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "average_precision_pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def save_evaluation_plots(y_true, probabilities, threshold: float, output_dir: Path) -> None:
    """Save a precision-recall curve and a thresholded confusion matrix."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = (np.asarray(probabilities) >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y_true, probabilities, ax=ax, name="Model")
    ax.set_title("Fraud detection: test precision-recall curve")
    fig.tight_layout()
    fig.savefig(output_dir / "test_precision_recall_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        labels=[0, 1],
        display_labels=["Legitimate", "Fraud"],
        values_format=",d",
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"Test confusion matrix (threshold={threshold:.4f})")
    fig.tight_layout()
    fig.savefig(output_dir / "test_confusion_matrix.png", dpi=160)
    plt.close(fig)
