"""
evaluate.py — Model Evaluation and Metrics

Why not just use accuracy?

If 98.7% of transactions are legit, a model that predicts "not fraud" for EVERYTHING
gets 98.7% accuracy but catches exactly 0 frauds. Useless.

Better metrics for imbalanced problems like fraud detection:

  Precision = of all transactions flagged as fraud, what % actually are fraud?
              (Minimize false alarms — costly for customer experience)

  Recall    = of all actual frauds, what % did we catch?
              (Minimize missed fraud — costly for the bank)

  F1-Score  = harmonic mean of Precision and Recall. Balances both.

  AUC-ROC   = Area Under the ROC Curve. Measures how well the model separates
              fraud from non-fraud across ALL decision thresholds.
              0.5 = random guessing, 1.0 = perfect.

In fraud detection, high Recall is usually prioritised over Precision —
missing a fraud is worse than a false alarm.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
import plotly.graph_objects as go
import plotly.express as px


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Compute all evaluation metrics for a single model.

    Args:
        y_true:  Ground truth labels (0/1)
        y_pred:  Model's binary predictions (0/1)
        y_proba: Model's fraud probability scores (float in [0,1])

    Returns:
        Dictionary of metric names → values
    """
    return {
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1':        round(f1_score(y_true, y_pred, zero_division=0), 4),
        'auc_roc':   round(roc_auc_score(y_true, y_proba), 4),
    }


def evaluate_all_models(results: dict) -> dict:
    """
    Compute metrics for all three models and return a combined results dict.

    Args:
        results: Dictionary returned by train.train_all(), plus prediction arrays

    Returns:
        Dictionary: model_name → metrics dict
    """
    from src.train import (
        predict_isolation_forest, predict_xgboost, predict_lstm
    )

    y_test        = results['y_test']
    X_test        = results['X_test']
    X_test_scaled = results['X_test_scaled']
    feature_names = results.get('feature_names')

    # Get predictions from each model
    # Isolation Forest uses RAW (unscaled) features, matching how it was trained.
    iso_pred,  iso_proba  = predict_isolation_forest(results['iso_forest'], X_test)
    xgb_pred,  xgb_proba  = predict_xgboost(results['xgboost'], X_test, feature_names=feature_names)
    out = {
        'Isolation Forest': {
            'metrics':  compute_metrics(y_test, iso_pred, iso_proba),
            'y_pred':   iso_pred,
            'y_proba':  iso_proba,
            'y_test':   y_test,
        },
        'XGBoost': {
            'metrics':  compute_metrics(y_test, xgb_pred, xgb_proba),
            'y_pred':   xgb_pred,
            'y_proba':  xgb_proba,
            'y_test':   y_test,
        },
    }

    # LSTM is optional (requires torch). Only evaluate it if it was trained.
    if results.get('lstm') is not None:
        lstm_pred, lstm_proba = predict_lstm(results['lstm'], X_test_scaled)
        out['LSTM'] = {
            'metrics':  compute_metrics(y_test, lstm_pred, lstm_proba),
            'y_pred':   lstm_pred,
            'y_proba':  lstm_proba,
            'y_test':   y_test,
        }

    return out


# ─────────────────────────────────────────────
# PLOTLY VISUALISATIONS (returned to Streamlit)
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    """
    Create a clean, readable Plotly confusion matrix.

    Design notes:
      - Colors are normalised PER ROW. With 1.3% fraud, raw counts make the
        fraud row invisible (3,900 vs 48). Row percentages keep every cell
        readable and meaningful ("of all actual frauds, what share went where").
      - Each cell shows the count, the row percentage, and what the cell means
        (Caught, Missed, False Alarm, Cleared).
      - No colorbar: it added clutter without information.
    """
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_pct  = cm / np.maximum(row_sums, 1)

    # Meaning of each cell: rows = actual [not fraud, fraud], cols = predicted
    meaning = [['Cleared', 'False Alarm'],
               ['Missed',  'Caught']]

    cell_text = [
        [f"<b>{cm[r][c]:,}</b><br>{row_pct[r][c]:.1%}<br>{meaning[r][c]}"
         for c in range(2)]
        for r in range(2)
    ]

    fig = go.Figure(data=go.Heatmap(
        z=row_pct,
        x=['Predicted<br>Not Fraud', 'Predicted<br>Fraud'],
        y=['Actual<br>Not Fraud', 'Actual<br>Fraud'],
        colorscale='Blues',
        zmin=0.0, zmax=1.0,
        text=cell_text,
        texttemplate='%{text}',
        textfont={'size': 13},
        showscale=False,
        xgap=4, ygap=4,
        hovertemplate='%{y} / %{x}<br>count: %{customdata:,}<extra></extra>',
        customdata=cm,
    ))
    fig.update_layout(
        title={'text': f'<b>{model_name}</b>', 'x': 0.5, 'xanchor': 'center',
               'font': {'size': 15}},
        height=320,
        margin=dict(t=45, b=10, l=10, r=10),
        xaxis=dict(side='bottom', tickfont={'size': 11}, automargin=True),
        yaxis=dict(tickfont={'size': 11}, automargin=True, autorange='reversed'),
    )
    return fig


def plot_metrics_comparison(all_metrics: dict):
    """
    Bar chart comparing Precision, Recall, F1, and AUC-ROC across all three models.
    """
    metric_names = ['precision', 'recall', 'f1', 'auc_roc']
    model_names  = list(all_metrics.keys())

    fig = go.Figure()
    for metric in metric_names:
        values = [all_metrics[m]['metrics'][metric] for m in model_names]
        fig.add_trace(go.Bar(name=metric.upper(), x=model_names, y=values))

    fig.update_layout(
        barmode='group',
        title='Model Performance Comparison',
        yaxis=dict(range=[0, 1.05], title='Score'),
        xaxis_title='Model',
        legend_title='Metric',
        height=400,
    )
    return fig


def plot_feature_importance(xgb_model, feature_names: list):
    """
    Horizontal bar chart of XGBoost feature importance.

    XGBoost tracks how often each feature is used to split data across all trees.
    Features used more = more important for the model's decisions.
    """
    importance = xgb_model.feature_importances_
    df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    df = df.sort_values('importance', ascending=True)

    fig = px.bar(
        df, x='importance', y='feature', orientation='h',
        title='XGBoost Feature Importance',
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Blues',
    )
    fig.update_layout(height=450, showlegend=False)
    return fig


def plot_class_distribution(df):
    """
    Pie + bar chart showing the fraud/non-fraud class split.
    """
    counts = df['isFraud'].value_counts().reset_index()
    counts.columns = ['Class', 'Count']
    counts['Class'] = counts['Class'].map({0: 'Not Fraud', 1: 'Fraud'})

    fig = px.pie(
        counts, values='Count', names='Class',
        title='Class Distribution (Fraud vs. Not Fraud)',
        color='Class',
        color_discrete_map={'Not Fraud': '#4C9BE8', 'Fraud': '#E84C4C'},
        hole=0.4,
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def plot_roc_curves(all_results: dict):
    """
    ROC curves for all three models on the same chart.

    ROC curve plots True Positive Rate vs. False Positive Rate at every threshold.
    The more the curve bows toward the top-left corner, the better.
    AUC = area under that curve (higher = better).
    """
    from sklearn.metrics import roc_curve

    fig = go.Figure()
    colors = {'Isolation Forest': '#FF6B6B', 'XGBoost': '#4ECDC4', 'LSTM': '#45B7D1'}

    for name, result in all_results.items():
        fpr, tpr, _ = roc_curve(result['y_test'], result['y_proba'])
        auc = result['metrics']['auc_roc']
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"{name} (AUC={auc:.3f})",
            line=dict(color=colors.get(name, '#888'), width=2),
        ))

    # Random baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random (AUC=0.5)',
        line=dict(color='gray', width=1, dash='dash'),
    ))

    fig.update_layout(
        title='ROC Curves',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=400,
        legend=dict(x=0.6, y=0.1),
    )
    return fig
