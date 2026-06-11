"""
train.py — Model Training

Three models of increasing sophistication:

1. Isolation Forest   — Unsupervised. No labels needed. Detects "weird" transactions.
2. XGBoost + SMOTE    — Supervised. The industry standard for tabular fraud detection.
3. LSTM (PyTorch)     — Supervised. A neural network that learns patterns in sequences.

The models are saved to the /models folder so the dashboard can reload them
without retraining every time.
"""

import numpy as np
import joblib
from pathlib import Path

# ── Scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# ── XGBoost ───────────────────────────────────────────────────────────────────
from xgboost import XGBClassifier

# ── SMOTE: Synthetic Minority Oversampling TEchnique ──────────────────────────
# Fraud is ~1.3% of data. Without correction, models learn to just say "not fraud"
# for everything and still get 98.7% accuracy — useless for real fraud detection.
# SMOTE creates synthetic fraud examples so the model sees a balanced dataset.
from imblearn.over_sampling import SMOTE

# ── PyTorch (optional) ───────────────────────────────────────────────────────
# The LSTM is the only model that needs torch. If torch is not installed the
# rest of the system (Isolation Forest, XGBoost, dashboard) still works, and
# the LSTM is simply skipped. This keeps the install light for people who only
# want the tabular models.
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:  # ImportError, or OSError from a broken install
    TORCH_AVAILABLE = False


MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# 1. ISOLATION FOREST  (unsupervised baseline)
# ═════════════════════════════════════════════════════════════════════════════

def train_isolation_forest(X: np.ndarray, y: np.ndarray | None = None,
                           contamination: float = 0.013):
    """
    Train an Isolation Forest anomaly detector (unsupervised baseline).

    How it works: Build many random decision trees. Normal points need many
    splits to isolate; anomalous points are isolated quickly with few splits.

    Two important details for an honest baseline:
      - Isolation Forest does NOT need feature scaling (it splits on raw values),
        so we train it on the unscaled matrix.
      - If labels are available we fit on NORMAL transactions only. This is the
        proper "novelty detection" setup: the model learns what normal looks
        like, then flags anything that deviates.

    This model is expected to be the WEAKEST of the three. Fraud here is
    designed to mimic normal transactions, so a purely unsupervised detector
    catches only the obvious cases. That gap is exactly why banks rely on
    supervised models (XGBoost) trained on labelled fraud.

    Args:
        X:             Unscaled feature matrix
        y:             Optional labels. If given, fit on normal rows only.
        contamination: Expected fraction of anomalies

    Returns:
        Fitted IsolationForest model
    """
    print("Training Isolation Forest (unsupervised baseline)...")
    X_fit = X if y is None else X[y == 0]
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_fit)

    joblib.dump(model, MODELS_DIR / 'isolation_forest.joblib')
    print("  Saved -> models/isolation_forest.joblib")
    return model


def predict_isolation_forest(model: IsolationForest, X: np.ndarray):
    """
    Run inference with the Isolation Forest.

    Returns:
        y_pred: binary labels (1=fraud, 0=normal)
        scores: anomaly score (higher = more anomalous). Used for AUC-ROC.
    """
    # IsolationForest returns -1 (anomaly) or +1 (normal)
    raw = model.predict(X)
    y_pred = (raw == -1).astype(int)  # convert to 0/1

    # decision_function returns negative scores; flip so higher = more anomalous
    scores = -model.decision_function(X)
    # Normalise to [0, 1] range so it acts like a probability
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return y_pred, scores


# ═════════════════════════════════════════════════════════════════════════════
# 2. XGBOOST WITH SMOTE  (main supervised model)
# ═════════════════════════════════════════════════════════════════════════════

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, feature_names: list | None = None):
    """
    Train an XGBoost classifier on SMOTE-balanced data.

    XGBoost (eXtreme Gradient Boosting) builds an ensemble of decision trees
    where each new tree corrects the mistakes of all previous trees. It's the
    most commonly used model in financial fraud detection competitions.

    SMOTE flow:
        Original: 98,700 non-fraud + 1,300 fraud
        After SMOTE: 98,700 non-fraud + 98,700 synthetic fraud → balanced!

    Args:
        X_train:       Training features (scaled or unscaled — XGBoost is robust)
        y_train:       Training labels (0 or 1)
        feature_names: Optional list of feature names for importance plots

    Returns:
        Fitted XGBClassifier
    """
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"  Before SMOTE: {y_train.sum()} fraud / {len(y_train)} total")
    print(f"  After  SMOTE: {y_resampled.sum()} fraud / {len(y_resampled)} total")

    print("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=200,          # 200 trees in the ensemble
        max_depth=6,               # each tree can be 6 levels deep
        learning_rate=0.1,         # how much each tree contributes
        subsample=0.8,             # use 80% of rows per tree (prevents overfitting)
        colsample_bytree=0.8,      # use 80% of features per tree
        scale_pos_weight=1,        # already balanced via SMOTE, so 1:1
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    if feature_names:
        import pandas as pd
        X_df = pd.DataFrame(X_resampled, columns=feature_names)
        model.fit(X_df, y_resampled)
    else:
        model.fit(X_resampled, y_resampled)

    joblib.dump(model, MODELS_DIR / 'xgboost.joblib')
    print("  Saved -> models/xgboost.joblib")
    return model


def predict_xgboost(model: XGBClassifier, X: np.ndarray, feature_names: list | None = None):
    """
    Run XGBoost inference.

    Returns:
        y_pred:  binary predictions
        y_proba: fraud probability for each transaction (used for AUC-ROC)
    """
    if feature_names:
        import pandas as pd
        X = pd.DataFrame(X, columns=feature_names)
    y_proba = model.predict_proba(X)[:, 1]  # column 1 = probability of fraud
    y_pred  = (y_proba >= 0.5).astype(int)
    return y_pred, y_proba


# ═════════════════════════════════════════════════════════════════════════════
# 3. LSTM (PyTorch deep learning model)
# ═════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class FraudLSTM(nn.Module):
        """
        A simple LSTM neural network for fraud classification.

        Architecture:
            Input  → LSTM layers → Dropout → Fully Connected → Sigmoid output

        We treat each transaction's feature vector as a sequence of length = n_features,
        where each feature is one time step. This is a common educational simplification
        that lets us demonstrate LSTM without needing account-level transaction histories.

        In production, you'd group transactions by account and feed real sequences.
        """

        def __init__(self, input_size: int = 1, hidden_size: int = 64,
                     num_layers: int = 2, dropout: float = 0.3):
            super().__init__()

            self.hidden_size = hidden_size
            self.num_layers  = num_layers

            # LSTM: processes the sequence step-by-step, remembering important past steps
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,      # shape: (batch, seq_len, features)
                dropout=dropout if num_layers > 1 else 0,
            )

            # Dropout randomly zeroes some neurons during training → prevents overfitting
            self.dropout = nn.Dropout(dropout)

            # Fully connected layer: maps LSTM output to a single fraud probability
            self.fc = nn.Linear(hidden_size, 1)

            # Sigmoid squashes output to [0, 1] so we can interpret it as a probability
            self.sigmoid = nn.Sigmoid()

        def forward(self, x, return_logits: bool = False):
            # x shape: (batch_size, seq_len, input_size)
            lstm_out, _ = self.lstm(x)
            # Take only the last time step's output
            last_hidden = lstm_out[:, -1, :]
            out = self.dropout(last_hidden)
            logits = self.fc(out).squeeze(1)  # shape: (batch_size,)
            # During training we return raw logits for BCEWithLogitsLoss (stable +
            # supports pos_weight). During inference we apply sigmoid to get a
            # probability in [0, 1].
            if return_logits:
                return logits
            return self.sigmoid(logits)


else:
    class FraudLSTM:  # stub so type hints and imports do not crash
        pass


def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               epochs: int = 15, batch_size: int = 512, lr: float = 0.001):
    """
    Train the LSTM model.

    The feature matrix X is reshaped from (n, features) to (n, features, 1)
    — treating each feature as one timestep with a single value.

    Args:
        X_train:    Scaled training features (scaling is important for neural nets!)
        y_train:    Training labels
        epochs:     Number of full passes through the training data
        batch_size: How many transactions to process at once
        lr:         Learning rate (how big a step the optimizer takes each update)

    Returns:
        Fitted FraudLSTM model
    """
    if not TORCH_AVAILABLE:
        print("PyTorch is not installed. Skipping LSTM (install torch to enable it).")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training LSTM on: {device}")

    # ── Handle class imbalance with pos_weight ────────────────────────────────
    # Instead of SMOTE, PyTorch lets us penalise wrong fraud predictions more heavily.
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-9)], dtype=torch.float32).to(device)

    # ── Reshape X: (n, features) → (n, features, 1) ──────────────────────────
    seq_len = X_train.shape[1]
    X_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(2)  # add channel dim
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # ── Build model ───────────────────────────────────────────────────────────
    model = FraudLSTM(input_size=1, hidden_size=64, num_layers=2, dropout=0.3).to(device)

    # Class imbalance handling: weight the positive (fraud) class by how rare it
    # is, so the network is penalised heavily for missing fraud. We compute the
    # logits version (model outputs a probability via sigmoid, so we strip the
    # final sigmoid for training and use BCEWithLogitsLoss, which is both
    # numerically stable AND supports pos_weight).
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Adam optimiser: adapts the learning rate per-parameter (works well in practice)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Reduce learning rate if loss plateaus (helps convergence)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()          # clear previous gradients
            preds = model(X_batch, return_logits=True)  # raw logits for stable loss
            loss  = criterion(preds, y_batch)  # compute loss
            loss.backward()                # backpropagation: compute gradients
            optimizer.step()               # update weights
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        print(f"  Epoch {epoch+1:02d}/{epochs} — loss: {avg_loss:.4f}")

    # ── Save model ────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), MODELS_DIR / 'lstm.pt')
    # Also save architecture params so we can reconstruct the model for inference
    joblib.dump({'input_size': 1, 'hidden_size': 64, 'num_layers': 2, 'dropout': 0.3,
                 'seq_len': seq_len}, MODELS_DIR / 'lstm_config.joblib')
    print("  Saved -> models/lstm.pt and models/lstm_config.joblib")
    return model


def predict_lstm(model: FraudLSTM, X: np.ndarray, batch_size: int = 512):
    """
    Run LSTM inference.

    Returns:
        y_pred:  binary predictions
        y_proba: fraud probability for each transaction
    """
    if not TORCH_AVAILABLE or model is None:
        raise RuntimeError("LSTM unavailable: PyTorch is not installed or model not trained.")

    device = next(model.parameters()).device
    model.eval()  # disable dropout during inference

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(2)
    dataset   = TensorDataset(X_tensor)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs = []
    with torch.no_grad():  # no need to track gradients during inference
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            probs   = model(X_batch).cpu().numpy()
            all_probs.append(probs)

    y_proba = np.concatenate(all_probs)
    y_pred  = (y_proba >= 0.5).astype(int)
    return y_pred, y_proba


# ═════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: train all three models at once
# ═════════════════════════════════════════════════════════════════════════════

def train_all(X: np.ndarray, y: np.ndarray, feature_names: list | None = None):
    """
    Train all three models and return them in a dictionary.

    This is what the Streamlit dashboard calls with a single button click.

    Args:
        X:             Full (unscaled) feature matrix
        y:             Labels
        feature_names: Column names for XGBoost importance plots

    Returns:
        dict with keys: 'iso_forest', 'xgboost', 'lstm', 'scaler',
                        'X_test', 'y_test', 'X_test_scaled'
    """
    from src.features import scale_features

    # Train/test split — 80% train, 20% test, stratified so fraud rate is same in both
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (required for LSTM, optional for others)
    X_train_s, X_test_s, scaler = scale_features(
        X_train, X_test, save_path='models/scaler.joblib'
    )

    # Train all three.
    # Isolation Forest: raw (unscaled) features, fit on normal rows only.
    # XGBoost: raw features (tree model, robust to scale).
    # LSTM: scaled features (neural nets need it).
    iso   = train_isolation_forest(X_train, y_train)
    xgb   = train_xgboost(X_train, y_train, feature_names=feature_names)
    lstm  = train_lstm(X_train_s, y_train)

    return {
        'iso_forest':   iso,
        'xgboost':      xgb,
        'lstm':         lstm,
        'scaler':       scaler,
        'X_test':       X_test,
        'X_test_scaled': X_test_s,
        'y_test':       y_test,
        'feature_names': feature_names,
    }
