# Credit Card Fraud Detection

An end-to-end machine learning project for identifying potentially fraudulent card transactions in a highly imbalanced dataset. The project compares classical classifiers, tunes the leading model, evaluates it on a later time period, and includes an interactive Streamlit dashboard.

## Project highlights

- Chronological train, validation, and test split to keep later transactions out of model selection.
- Logistic regression, random forest, Extra Trees, and a prior baseline compared using average precision (PR-AUC).
- Hyperparameter search with `TimeSeriesSplit` on training data only.
- Decision threshold selected on validation data, followed by one final evaluation on the held-out test period.
- Streamlit pages for test results, interactive validation-threshold exploration, and scoring a CSV with a locally trained model.

## Test results

The saved test results below came from the project's held-out chronological test period. The threshold (`0.3400`) was selected on validation data before test evaluation.

| Metric | Test result |
| --- | ---: |
| Selected model | Extra Trees |
| Average precision (PR-AUC) | 0.7796 |
| ROC-AUC | 0.9256 |
| Precision | 95.12% |
| Recall | 75.00% |
| F1 score | 0.8387 |
| Balanced accuracy | 0.8750 |
| Accuracy | 99.9649% |
| Validation-selected threshold | 0.3400 |

At this threshold, the test set contained 39 detected fraud cases, 13 missed fraud cases, and 2 legitimate transactions flagged for review. Accuracy is high partly because legitimate transactions dominate this dataset, so PR-AUC, precision, and recall are more useful for understanding fraud detection performance.

Extra Trees led the validation comparison with PR-AUC `0.8714`; the later test period scored `0.7796`. This is a noticeable validation-to-test drop. It does not by itself identify the cause, but it is a reason to inspect how transaction patterns change over time before using the model beyond this benchmark.

## Dataset

Download `creditcard.csv` from the [Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/raw/creditcard.csv`. Review the dataset page for its terms and attribution requirements. The dataset is not included in this repository.

The target column is `Class` (`1` = fraud, `0` = legitimate). Input features are `Time`, `Amount`, and anonymized columns `V1` through `V28`. Rows are ordered by time before splitting. `Time` and `Amount` are scaled inside each model pipeline; the anonymized features pass through unchanged.

## Setup

Python 3.10 or newer is required. From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, create the environment with `py -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1`.

## Train the models

After placing the dataset at `data/raw/creditcard.csv`, run:

```bash
python -m fraud_detection.train
```

For a dataset in another location:

```bash
python -m fraud_detection.train --data /path/to/creditcard.csv
```

The training command writes evaluation summaries and charts to `outputs/`, and the fitted model to `models/fraud_detector.joblib`. It also creates `outputs/validation_predictions.csv` for the interactive threshold explorer. The model and row-level validation scores are excluded from Git and can be regenerated locally.

## Launch the Streamlit app

With the virtual environment active:

```bash
streamlit run app.py
```

The **Overview** page shows the saved test metrics, validation model comparison, confusion matrix, and precision–recall curve. The **Threshold lab** page uses local validation predictions to show how different thresholds affect validation precision, recall, F1, and confusion counts. It does not adjust or re-evaluate the held-out test results.

The **Score transactions** page needs the locally generated `models/fraud_detector.joblib`. Upload a CSV containing the same numeric feature columns used for training (`Time`, `V1`–`V28`, and `Amount`). An optional `Class` column is not used as a model input. The app adds a fraud probability and a review flag, and lets you download the scored CSV. Use anonymized or synthetic data for this demo; do not upload real card numbers or sensitive customer information.

## Repository structure

```text
credit-card-fraud-detection/
├── app.py                               # Streamlit dashboard
├── data/raw/README.md                   # Dataset placement instructions
├── models/README.md                     # Local model artifact instructions
├── outputs/
│   ├── validation_model_comparison.csv  # Validation model comparison
│   ├── test_metrics.json                # Saved holdout metrics
│   ├── test_confusion_matrix.png        # Holdout confusion matrix
│   └── test_precision_recall_curve.png  # Holdout precision–recall curve
├── src/fraud_detection/
│   ├── config.py                        # Paths and split settings
│   ├── data.py                          # Dataset validation and chronological split
│   ├── evaluation.py                    # Threshold, metrics, and evaluation plots
│   ├── models.py                        # Preprocessing and candidate estimators
│   ├── train.py                         # Training and final evaluation entry point
│   └── tuning.py                        # Time-series cross-validation search
├── pyproject.toml
└── requirements.txt
```

## Method and limitations

Fraud is a rare class, so a high accuracy score alone can be misleading. Average precision summarizes precision across recall levels, while the selected operating threshold determines the precision–recall trade-off. This demonstration selects the threshold to maximize validation F1; real fraud operations should choose thresholds based on investigation capacity and the costs of false alerts and missed fraud.

This is a historical benchmark and portfolio project, not a production payment or fraud-decision service. The anonymized benchmark features and its time period may not represent a current payment stream. The reported metrics do not guarantee future performance.

## Git history suggestion

When adding the project to a new, empty repository, create focused commits that describe real pieces of work, for example:

1. `Set up project structure and data validation`
2. `Add time-aware model training and evaluation`
3. `Build interactive Streamlit dashboard`
4. `Document benchmark results and usage`

Use the actual date and your own Git identity for every commit. Do not backdate commits or claim that work happened on dates it did not.
