"""Interactive Streamlit dashboard for the credit-card fraud project."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
MODEL_PATH = ROOT / "models" / "fraud_detector.joblib"
METRICS_PATH = OUTPUTS / "test_metrics.json"
COMPARISON_PATH = OUTPUTS / "validation_model_comparison.csv"
VALIDATION_PREDICTIONS_PATH = OUTPUTS / "validation_predictions.csv"

st.set_page_config(
    page_title="Fraud Detection Workbench",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --ink: #e9f1fa; --muted: #9aadc2; --panel: #101a29; --line: #22364a;
              --cyan: #39d8c5; --blue: #7eb9ff; }
      .stApp { background: radial-gradient(ellipse at 12% 0%, #14283a 0, #0a111c 44%, #080d15 100%); }
      [data-testid="stSidebar"] { background: #0b1420; border-right: 1px solid var(--line); }
      [data-testid="stSidebar"] * { color: var(--ink); }
      .block-container { padding-top: 2.1rem; padding-bottom: 2.7rem; max-width: 1440px; }
      h1, h2, h3 { color: var(--ink) !important; letter-spacing: -0.025em; }
      p, li, label, [data-testid="stCaptionContainer"] { color: var(--muted); }
      .hero { padding: 1.7rem 2rem; margin-bottom: 1.3rem; border: 1px solid #23445a;
              border-radius: 20px; background: linear-gradient(120deg, rgba(23,51,69,.97), rgba(14,25,40,.94));
              box-shadow: 0 16px 44px rgba(0,0,0,.22); }
      .hero .eyebrow { margin: 0 0 .45rem; color: var(--cyan); font-size: .78rem; font-weight: 700;
                       text-transform: uppercase; letter-spacing: .14em; }
      .hero h1 { margin: 0; font-size: clamp(2rem, 4vw, 3.2rem); line-height: 1.08; color: #f5f9ff !important; }
      .hero .lede { max-width: 850px; margin: .7rem 0 0; font-size: 1.02rem; line-height: 1.6; }
      .metric-card { min-height: 116px; padding: 1rem 1.1rem; border: 1px solid var(--line);
                     border-radius: 16px; background: linear-gradient(145deg, #111d2d, #0d1724); }
      .metric-label { color: #9db2c8; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
      .metric-value { margin-top: .42rem; color: #f3f8ff; font-size: 1.75rem; font-weight: 750; line-height: 1.1; }
      .metric-note { margin-top: .4rem; color: #7892aa; font-size: .78rem; }
      .section-kicker { margin: 1.5rem 0 .35rem; color: var(--cyan); font-size: .78rem; font-weight: 700;
                        letter-spacing: .12em; text-transform: uppercase; }
      .small-note { color: #8fa5ba; font-size: .88rem; line-height: 1.55; }
      .stButton > button, .stDownloadButton > button { border-radius: 10px; }
      div[data-testid="stAlert"] { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def read_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def read_comparison() -> pd.DataFrame:
    if not COMPARISON_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(COMPARISON_PATH)


@st.cache_data
def read_validation_predictions() -> pd.DataFrame:
    if not VALIDATION_PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(VALIDATION_PREDICTIONS_PATH)


@st.cache_resource
def read_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><p class="eyebrow">Machine learning · imbalanced classification</p>'
        f'<h1>{title}</h1><p class="lede">{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def styled_figure(figure: go.Figure) -> go.Figure:
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce8f5", family="Arial, sans-serif"),
        margin=dict(l=12, r=12, t=48, b=12),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    figure.update_xaxes(gridcolor="rgba(135,165,190,.13)", zerolinecolor="rgba(135,165,190,.16)")
    figure.update_yaxes(gridcolor="rgba(135,165,190,.13)", zerolinecolor="rgba(135,165,190,.16)")
    return figure


def overview(metrics: dict, comparison: pd.DataFrame) -> None:
    hero(
        "Fraud Detection Workbench",
        "An end-to-end, time-aware machine learning project for rare card-fraud detection. "
        "Explore model comparisons, validation thresholds, and transaction risk scores.",
    )

    if not metrics:
        st.warning("Evaluation summary is missing. Run the training pipeline to generate project outputs.")
        return

    threshold = float(metrics.get("threshold", 0.34))
    model_name = metrics.get("selected_model", "Extra trees")
    st.markdown('<div class="section-kicker">Held-out test results</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    with cols[0]:
        metric_card("Average precision · PR-AUC", f"{metrics.get('average_precision_pr_auc', 0):.3f}", "Primary metric for rare fraud")
    with cols[1]:
        metric_card("Precision", f"{metrics.get('precision', 0):.1%}", "Of flagged transactions, this share was fraud")
    with cols[2]:
        metric_card("Recall", f"{metrics.get('recall', 0):.1%}", "Share of fraud transactions detected")
    with cols[3]:
        metric_card("F1 score", f"{metrics.get('f1', 0):.3f}", f"{model_name} · threshold {threshold:.2f}")

    st.caption(
        f"Test ROC-AUC: {metrics.get('roc_auc', 0):.3f} · "
        f"Balanced accuracy: {metrics.get('balanced_accuracy', 0):.3f} · "
        f"Accuracy: {metrics.get('accuracy', 0):.4%}. Accuracy is shown for completeness; "
        "it is not the main success metric for this imbalanced dataset."
    )

    st.markdown('<div class="section-kicker">Validation model comparison</div>', unsafe_allow_html=True)
    if comparison.empty:
        st.info("The validation comparison file is not available yet. Run the training pipeline to create it.")
    else:
        ranked = comparison.sort_values("validation_pr_auc", ascending=True)
        color_map = {
            "Extra trees": "#39d8c5",
            "Random forest": "#7eb9ff",
            "Logistic regression": "#aa91ff",
            "Prior baseline": "#718096",
        }
        figure = px.bar(
            ranked,
            x="validation_pr_auc",
            y="model",
            color="model",
            color_discrete_map=color_map,
            orientation="h",
            text=ranked["validation_pr_auc"].map(lambda value: f"{value:.3f}"),
            hover_data={"validation_roc_auc": ":.4f", "validation_pr_auc": ":.4f"},
            labels={"validation_pr_auc": "Validation average precision (PR-AUC)", "model": "Model"},
        )
        figure.update_traces(textposition="outside", cliponaxis=False)
        figure.update_layout(height=330, showlegend=False, xaxis_range=[0, 1])
        st.plotly_chart(styled_figure(figure), use_container_width=True)

    st.markdown('<div class="section-kicker">Test-set diagnostics</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        image_path = OUTPUTS / "test_confusion_matrix.png"
        if image_path.exists():
            st.image(str(image_path), caption="Held-out test confusion matrix at the validation-selected threshold", use_container_width=True)
        else:
            st.info("Run training to generate the test confusion matrix.")
    with right:
        image_path = OUTPUTS / "test_precision_recall_curve.png"
        if image_path.exists():
            st.image(str(image_path), caption="Precision–recall curve on the held-out test period", use_container_width=True)
        else:
            st.info("Run training to generate the test precision–recall curve.")

    st.info(
        f"At the saved threshold, the holdout contained {metrics.get('true_positives', 0)} detected fraud cases, "
        f"{metrics.get('false_negatives', 0)} missed fraud cases, and {metrics.get('false_positives', 0)} false alerts. "
        "A model score supports review; it should not make a real payment decision by itself."
    )


def threshold_lab(metrics: dict) -> None:
    hero(
        "Validation Threshold Lab",
        "Move the decision threshold and see how precision, recall, F1, and validation errors change. "
        "This explorer uses validation predictions, keeping the final test results separate.",
    )
    predictions = read_validation_predictions()
    if predictions.empty or not {"true_label", "fraud_probability"}.issubset(predictions.columns):
        st.warning(
            "Validation prediction scores are not present. Download the dataset and run "
            "`python -m fraud_detection.train`; this creates the local file used by this page."
        )
        return

    y_true = predictions["true_label"].astype(int).to_numpy()
    probabilities = predictions["fraud_probability"].astype(float).to_numpy()
    default_threshold = float(metrics.get("threshold", 0.34))
    threshold = st.slider(
        "Flag a transaction when its fraud probability is at least",
        min_value=0.0,
        max_value=1.0,
        value=min(max(default_threshold, 0.0), 1.0),
        step=0.01,
        format="%.2f",
    )
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    cols = st.columns(4)
    with cols[0]:
        metric_card("Precision", f"{precision_score(y_true, predicted, zero_division=0):.1%}", "Flagged transactions that were fraud")
    with cols[1]:
        metric_card("Recall", f"{recall_score(y_true, predicted, zero_division=0):.1%}", "Fraud cases caught in validation")
    with cols[2]:
        metric_card("F1 score", f"{f1_score(y_true, predicted, zero_division=0):.3f}", "Balance of precision and recall")
    with cols[3]:
        metric_card("Validation PR-AUC", f"{average_precision_score(y_true, probabilities):.3f}", "Threshold-independent ranking quality")

    left, right = st.columns([1, 1.25])
    with left:
        st.markdown("#### Validation confusion matrix")
        matrix = [[int(tn), int(fp)], [int(fn), int(tp)]]
        figure = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=["Predicted legitimate", "Predicted fraud"],
                y=["Actual legitimate", "Actual fraud"],
                text=[[f"{value:,}" for value in row] for row in matrix],
                texttemplate="%{text}",
                colorscale=[[0, "#102236"], [0.55, "#177f91"], [1, "#42d9c8"]],
                showscale=False,
                hovertemplate="%{y}<br>%{x}: %{z:,}<extra></extra>",
            )
        )
        figure.update_layout(height=360, title=f"Threshold = {threshold:.2f}", xaxis_title="", yaxis_title="")
        st.plotly_chart(styled_figure(figure), use_container_width=True)
    with right:
        st.markdown("#### Precision–recall curve")
        precision, recall, _ = precision_recall_curve(y_true, probabilities)
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="Validation curve", line=dict(color="#39d8c5", width=3)))
        figure.add_trace(go.Scatter(
            x=[recall_score(y_true, predicted, zero_division=0)],
            y=[precision_score(y_true, predicted, zero_division=0)],
            mode="markers",
            name=f"Threshold {threshold:.2f}",
            marker=dict(color="#ffbc66", size=12, line=dict(color="#fff2dd", width=1)),
        ))
        figure.update_layout(height=360, xaxis_title="Recall", yaxis_title="Precision", xaxis_range=[0, 1], yaxis_range=[0, 1])
        st.plotly_chart(styled_figure(figure), use_container_width=True)
    st.caption(f"Validation balanced accuracy at this threshold: {balanced_accuracy_score(y_true, predicted):.3f}.")


def score_transactions(metrics: dict) -> None:
    hero(
        "Score Transactions",
        "Upload a CSV with the same numeric feature columns used during training. "
        "The app returns fraud probabilities and a review flag; it does not change or overwrite your file.",
    )
    st.warning("Use anonymized or synthetic examples only. Do not upload real card numbers or sensitive customer data.")
    model = read_model()
    if model is None:
        st.info(
            "The trained model is intentionally kept out of Git. To enable scoring, download the dataset and run "
            "`python -m fraud_detection.train` locally first. The model will be created at `models/fraud_detector.joblib`."
        )
        return

    feature_names = list(getattr(model, "feature_names_in_", []))
    if not feature_names:
        feature_names = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
    st.caption("Required columns: " + ", ".join(feature_names))
    uploaded = st.file_uploader("Choose a transaction CSV", type=["csv"], help="An optional Class column is not used as a model input.")
    if uploaded is None:
        return

    try:
        transactions = pd.read_csv(uploaded)
    except Exception as error:
        st.error(f"Could not read the CSV: {error}")
        return
    if transactions.empty:
        st.error("The uploaded CSV has no transaction rows.")
        return
    missing = [column for column in feature_names if column not in transactions.columns]
    if missing:
        st.error("Missing required feature columns: " + ", ".join(missing))
        return
    features = transactions.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any() or not np.isfinite(features.to_numpy(dtype=float)).all():
        bad_columns = features.columns[features.isna().any()].tolist()
        if not bad_columns:
            bad_columns = features.columns[~np.isfinite(features.to_numpy(dtype=float)).all(axis=0)].tolist()
        st.error("These columns contain missing or nonnumeric values: " + ", ".join(bad_columns))
        return

    with st.spinner("Scoring uploaded transactions…"):
        probabilities = model.predict_proba(features)[:, 1]
    threshold = float(metrics.get("threshold", 0.34))
    results = transactions.copy()
    results["fraud_probability"] = probabilities
    results["review_flag"] = (probabilities >= threshold).astype(int)
    results["risk_status"] = results["review_flag"].map({1: "Review", 0: "No review"})

    flagged = int(results["review_flag"].sum())
    cols = st.columns(3)
    with cols[0]:
        metric_card("Rows scored", f"{len(results):,}", "Uploaded transactions")
    with cols[1]:
        metric_card("Flagged for review", f"{flagged:,}", f"Threshold {threshold:.2f}")
    with cols[2]:
        metric_card("Highest score", f"{results['fraud_probability'].max():.1%}", "Model probability, not certainty")

    histogram = px.histogram(
        results,
        x="fraud_probability",
        color="risk_status",
        nbins=35,
        color_discrete_map={"Review": "#ffbc66", "No review": "#39d8c5"},
        labels={"fraud_probability": "Predicted fraud probability", "risk_status": "Status"},
    )
    histogram.add_vline(x=threshold, line_dash="dash", line_color="#fff2dd", annotation_text=f"Threshold {threshold:.2f}")
    st.plotly_chart(styled_figure(histogram), use_container_width=True)

    st.markdown("#### Highest-risk rows")
    st.dataframe(results.sort_values("fraud_probability", ascending=False).head(25), use_container_width=True, hide_index=True)
    st.download_button(
        "Download scored CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="fraud_scored_transactions.csv",
        mime="text/csv",
        use_container_width=True,
    )


with st.sidebar:
    st.markdown("## 🛡️ Fraud Lab")
    st.caption("Credit-card fraud detection portfolio project")
    page = st.radio("Explore", ["Overview", "Threshold lab", "Score transactions"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        '<p class="small-note">Historical benchmark · time-ordered evaluation · '
        "risk scores are for demonstration and review.</p>",
        unsafe_allow_html=True,
    )

metrics = read_metrics()
comparison = read_comparison()
if page == "Overview":
    overview(metrics, comparison)
elif page == "Threshold lab":
    threshold_lab(metrics)
else:
    score_transactions(metrics)

st.markdown("---")
st.caption("Built with Python, scikit-learn, Streamlit, and Plotly · See README.md for data source and methodology.")
