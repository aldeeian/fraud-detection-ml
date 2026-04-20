"""
features.py — Feature Engineering Pipeline

Raw PaySim columns are a starting point, but ML models learn better when we
give them *derived* features that highlight fraud patterns directly.

Key fraud signals we engineer:
  - Balance discrepancies (fraudsters often zero out accounts)
  - Amount-to-balance ratio (how much of the account is being moved)
  - Transaction type encoded as numbers (models can't read strings)
  - Time features (fraud clusters at certain hours)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


# The exact set of numeric features we pass to models.
# Defined here as a constant so every file uses the same list.
FEATURE_COLUMNS = [
    'step',
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest',
    'type_encoded',
    'balance_diff_orig',
    'balance_diff_dest',
    'amount_to_balance_ratio',
    'is_high_amount',
    'hour_of_day',
    'is_flagged',
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw transaction DataFrame into a feature-rich DataFrame
    ready for model training or inference.

    Args:
        df: Output of etl.load_data() — cleaned raw transactions

    Returns:
        df with additional engineered columns appended
    """
    df = df.copy()  # never mutate the original

    # ── 1. Encode transaction type ────────────────────────────────────────────
    # ML models need numbers, not strings. We map each type to an integer.
    # Ordinal encoding is fine here because tree models (XGBoost) don't care about order.
    type_map = {
        'PAYMENT':  0,
        'TRANSFER': 1,
        'CASH_OUT': 2,
        'DEBIT':    3,
        'CASH_IN':  4,
    }
    df['type_encoded'] = df['type'].map(type_map).fillna(-1).astype(int)

    # ── 2. Balance difference for sender ─────────────────────────────────────
    # In a legitimate transaction: newbalanceOrig = oldbalanceOrg - amount
    # For fraud: newbalanceOrig is forced to 0, so this diff will be non-zero.
    # This is one of the STRONGEST fraud signals in PaySim.
    df['balance_diff_orig'] = (df['oldbalanceOrg'] - df['amount']) - df['newbalanceOrig']

    # ── 3. Balance difference for recipient ───────────────────────────────────
    # Legitimate: newbalanceDest = oldbalanceDest + amount
    # Fraudulent cash-outs often show anomalous destination balance changes.
    df['balance_diff_dest'] = (df['oldbalanceDest'] + df['amount']) - df['newbalanceDest']

    # ── 4. Amount-to-balance ratio ───────────────────────────────────────────
    # "Is this person moving a large fraction of their wealth?"
    # Ratio close to 1.0 = draining the account (fraud signal)
    # Add tiny epsilon to avoid division-by-zero
    df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1e-9)
    df['amount_to_balance_ratio'] = df['amount_to_balance_ratio'].clip(0, 10)  # cap at 10x

    # ── 5. High-amount flag ───────────────────────────────────────────────────
    # Transactions in the top 10% by amount deserve extra attention.
    high_amount_threshold = df['amount'].quantile(0.90)
    df['is_high_amount'] = (df['amount'] > high_amount_threshold).astype(int)

    # ── 6. Hour of day ────────────────────────────────────────────────────────
    # step is hours since simulation start; step % 24 gives hour-of-day.
    # Fraudsters often operate late at night when monitoring is reduced.
    df['hour_of_day'] = df['step'] % 24

    # ── 7. System flag ───────────────────────────────────────────────────────
    # isFlaggedFraud is the rule-based system's own guess — useful as a feature.
    df['is_flagged'] = df['isFlaggedFraud'].astype(int)

    return df


def get_X_y(df: pd.DataFrame):
    """
    Split an engineered DataFrame into features (X) and labels (y).

    Returns:
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,) with 0/1 fraud labels
    """
    df_eng = engineer_features(df)
    X = df_eng[FEATURE_COLUMNS].values.astype(np.float32)
    y = df_eng['isFraud'].values.astype(np.int32)
    return X, y


def scale_features(X_train: np.ndarray, X_test: np.ndarray, save_path: str | None = None):
    """
    Fit a StandardScaler on training data and transform both train and test sets.

    Why scale? Tree models (XGBoost, Isolation Forest) don't need scaling, but
    neural networks (LSTM) are very sensitive to feature magnitudes.

    Args:
        X_train:   Training feature matrix
        X_test:    Test feature matrix
        save_path: Optional path to save the fitted scaler (for later inference)

    Returns:
        X_train_scaled, X_test_scaled, fitted_scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit on train only!
    X_test_scaled  = scaler.transform(X_test)        # apply same transform to test

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_path)

    return X_train_scaled, X_test_scaled, scaler


def scale_single(x: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    Scale a single transaction's feature vector using an already-fitted scaler.
    Used during inference (predict.py).
    """
    return scaler.transform(x.reshape(1, -1))
