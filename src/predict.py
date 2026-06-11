"""
predict.py — Inference Functions

This module handles making predictions on:
  1. Single transactions (for the dashboard's "predict one transaction" form)
  2. Batches of new transactions (for bulk scoring)

Loaded models come from the /models directory. The scaler that was fitted
during training is also loaded so new data is transformed identically.
"""

import numpy as np
import pandas as pd
import joblib
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
from pathlib import Path

from src.features import engineer_features, FEATURE_COLUMNS
from src.train import FraudLSTM


MODELS_DIR = Path('models')


def load_models():
    """
    Load all saved models and the scaler from the /models directory.

    Returns None for any model that hasn't been trained yet.
    """
    models = {}

    iso_path = MODELS_DIR / 'isolation_forest.joblib'
    models['iso_forest'] = joblib.load(iso_path) if iso_path.exists() else None

    xgb_path = MODELS_DIR / 'xgboost.joblib'
    models['xgboost'] = joblib.load(xgb_path) if xgb_path.exists() else None

    lstm_path   = MODELS_DIR / 'lstm.pt'
    config_path = MODELS_DIR / 'lstm_config.joblib'
    if TORCH_AVAILABLE and lstm_path.exists() and config_path.exists():
        cfg  = joblib.load(config_path)
        lstm = FraudLSTM(
            input_size=cfg['input_size'],
            hidden_size=cfg['hidden_size'],
            num_layers=cfg['num_layers'],
            dropout=cfg['dropout'],
        )
        lstm.load_state_dict(torch.load(lstm_path, map_location='cpu'))
        lstm.eval()
        models['lstm'] = lstm
    else:
        models['lstm'] = None

    scaler_path = MODELS_DIR / 'scaler.joblib'
    models['scaler'] = joblib.load(scaler_path) if scaler_path.exists() else None

    return models


def predict_single(transaction: dict, models: dict) -> dict:
    """
    Predict fraud probability for a single transaction entered via the dashboard form.

    Args:
        transaction: Dictionary of raw transaction fields, e.g.:
            {
                'step': 100, 'type': 'TRANSFER', 'amount': 5000.0,
                'nameOrig': 'C123', 'oldbalanceOrg': 5000.0, 'newbalanceOrig': 0.0,
                'nameDest': 'C456', 'oldbalanceDest': 0.0, 'newbalanceDest': 5000.0,
                'isFlaggedFraud': 0,
            }
        models: Dict returned by load_models()

    Returns:
        Dictionary: model_name → {'probability': float, 'prediction': str}
    """
    # Build a one-row DataFrame so engineer_features() works identically to training
    df = pd.DataFrame([transaction])
    df_eng = engineer_features(df)
    X = df_eng[FEATURE_COLUMNS].values.astype(np.float32)  # shape: (1, n_features)

    results = {}

    # ── Isolation Forest ──────────────────────────────────────────────────────
    if models.get('iso_forest') is not None:
        # Isolation Forest was trained on RAW (unscaled) features
        raw      = models['iso_forest'].predict(X)
        score    = -models['iso_forest'].decision_function(X)[0]
        # Normalise score to a rough probability (not perfectly calibrated)
        prob_iso = float(np.clip((score + 0.5) / 1.0, 0, 1))
        results['Isolation Forest'] = {
            'probability': round(prob_iso, 4),
            'prediction':  'FRAUD' if raw[0] == -1 else 'NOT FRAUD',
        }

    # ── XGBoost ───────────────────────────────────────────────────────────────
    if models.get('xgboost'):
        xgb = models['xgboost']
        try:
            X_df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
            prob_xgb = float(xgb.predict_proba(X_df)[0, 1])
        except Exception:
            prob_xgb = float(xgb.predict_proba(X)[0, 1])
        results['XGBoost'] = {
            'probability': round(prob_xgb, 4),
            'prediction':  'FRAUD' if prob_xgb >= 0.5 else 'NOT FRAUD',
        }

    # ── LSTM ──────────────────────────────────────────────────────────────────
    if TORCH_AVAILABLE and models.get('lstm') is not None and models.get('scaler') is not None:
        X_scaled  = models['scaler'].transform(X)
        X_tensor  = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(2)  # (1, features, 1)
        with torch.no_grad():
            prob_lstm = float(models['lstm'](X_tensor).item())
        results['LSTM'] = {
            'probability': round(prob_lstm, 4),
            'prediction':  'FRAUD' if prob_lstm >= 0.5 else 'NOT FRAUD',
        }

    return results


def predict_batch(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """
    Score a batch of transactions with all available models.

    Args:
        df:     DataFrame of raw transactions (same schema as training data)
        models: Dict returned by load_models()

    Returns:
        Original DataFrame with added columns:
            iso_forest_prob, xgboost_prob, lstm_prob, ensemble_prob, ensemble_pred
    """
    df_eng = engineer_features(df)
    X = df_eng[FEATURE_COLUMNS].values.astype(np.float32)
    result_df = df.copy()

    probas = {}

    if models.get('iso_forest') is not None:
        # Isolation Forest was trained on RAW (unscaled) features
        scores = -models['iso_forest'].decision_function(X)
        probas['iso_forest_prob'] = np.clip((scores + 0.5), 0, 1)

    if models.get('xgboost'):
        try:
            X_df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
            probas['xgboost_prob'] = models['xgboost'].predict_proba(X_df)[:, 1]
        except Exception:
            probas['xgboost_prob'] = models['xgboost'].predict_proba(X)[:, 1]

    if TORCH_AVAILABLE and models.get('lstm') is not None and models.get('scaler') is not None:
        X_s      = models['scaler'].transform(X)
        X_tensor = torch.tensor(X_s, dtype=torch.float32).unsqueeze(2)
        with torch.no_grad():
            probas['lstm_prob'] = models['lstm'](X_tensor).numpy()

    for col, vals in probas.items():
        result_df[col] = vals

    # Ensemble: average of all available probabilities
    prob_cols = [c for c in probas]
    if prob_cols:
        result_df['ensemble_prob'] = result_df[prob_cols].mean(axis=1)
        result_df['ensemble_pred'] = (result_df['ensemble_prob'] >= 0.5).astype(int)

    return result_df
