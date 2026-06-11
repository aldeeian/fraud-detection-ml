"""
test_pipeline.py — tests for the fraud detection data and feature pipeline.

These tests run fast (no model training, no torch needed) and guard the two
things that matter most:

  1. The synthetic data has the right shape, schema, and fraud rate.
  2. There is no data leakage — the engineered features must NOT perfectly
     separate fraud from non-fraud. A quick logistic-regression probe that
     scores a perfect 1.0 AUC would mean the leak is back.

Run with:  pytest tests/ -v
"""

import numpy as np
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.etl import load_data, generate_synthetic_data, get_summary_stats
from src.features import engineer_features, get_X_y, FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def test_synthetic_data_has_correct_row_count():
    df = generate_synthetic_data(n_rows=5_000, random_state=0)
    assert len(df) == 5_000


def test_synthetic_data_has_paysim_schema():
    df = generate_synthetic_data(n_rows=1_000, random_state=0)
    expected = {
        "step", "type", "amount", "nameOrig", "oldbalanceOrg",
        "newbalanceOrig", "nameDest", "oldbalanceDest",
        "newbalanceDest", "isFraud", "isFlaggedFraud",
    }
    assert expected.issubset(set(df.columns))


def test_fraud_rate_is_realistic():
    df = generate_synthetic_data(n_rows=50_000, random_state=0)
    fraud_rate = df["isFraud"].mean()
    # Around 1.3%, allow a small tolerance band
    assert 0.008 < fraud_rate < 0.020


def test_fraud_only_in_transfer_and_cashout():
    df = generate_synthetic_data(n_rows=20_000, random_state=0)
    fraud_types = set(df.loc[df["isFraud"] == 1, "type"].unique())
    assert fraud_types.issubset({"TRANSFER", "CASH_OUT"})


def test_same_seed_gives_same_data():
    a = generate_synthetic_data(n_rows=1_000, random_state=7)
    b = generate_synthetic_data(n_rows=1_000, random_state=7)
    assert a.equals(b)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def test_engineered_columns_exist():
    df = generate_synthetic_data(n_rows=1_000, random_state=0)
    eng = engineer_features(df)
    for col in FEATURE_COLUMNS:
        assert col in eng.columns, f"missing engineered column: {col}"


def test_flagged_fraud_is_not_a_feature():
    # Guard against re-introducing the label-leak feature.
    assert "is_flagged" not in FEATURE_COLUMNS
    assert "isFlaggedFraud" not in FEATURE_COLUMNS


def test_get_X_y_shapes_match():
    df = generate_synthetic_data(n_rows=2_000, random_state=0)
    X, y = get_X_y(df)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == len(FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Data-leakage guard — the important one
# ---------------------------------------------------------------------------

def test_no_perfect_separation_leak():
    """
    If a simple linear model scores a perfect AUC, the features are leaking the
    label. Real fraud data is not perfectly separable. We require the probe AUC
    to be high (the signal is real) but strictly below 1.0 (no leak).
    """
    df = load_data(n_rows=30_000)
    X, y = get_X_y(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
    probe = LogisticRegression(max_iter=2_000, class_weight="balanced")
    probe.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, probe.predict_proba(X_te)[:, 1])
    # Must have real signal, but must not be a perfect (leaky) separator.
    assert 0.70 < auc < 0.9999, f"suspicious probe AUC={auc:.4f} (possible leak)"


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def test_summary_stats_keys():
    df = generate_synthetic_data(n_rows=1_000, random_state=0)
    stats = get_summary_stats(df)
    for key in ("total_transactions", "fraud_count", "fraud_rate_pct"):
        assert key in stats
    assert stats["total_transactions"] == 1_000
