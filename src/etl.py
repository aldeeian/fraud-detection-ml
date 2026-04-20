"""
etl.py — Extract, Transform, Load

This module is responsible for:
1. Generating a realistic synthetic PaySim-style dataset (so we don't need Kaggle)
2. Loading real CSV files if the user uploads one
3. Cleaning and validating the data before feature engineering begins

PaySim simulates mobile money transactions. Fraud occurs almost exclusively in
TRANSFER and CASH_OUT transactions, where a fraudster drains an account to zero
then immediately moves the money to another account they control.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────

def generate_synthetic_data(n_rows: int = 100_000, random_state: int = 42) -> pd.DataFrame:
    """
    Create a synthetic dataset that looks like real PaySim data.

    Why synthetic? The real PaySim CSV is ~470 MB and requires a Kaggle account.
    Our synthetic version captures the same statistical patterns so all the ML
    code works identically.

    Args:
        n_rows:       How many transaction rows to generate (default 100k)
        random_state: Seed for reproducibility — same seed = same data every time

    Returns:
        A pandas DataFrame with 11 columns matching the PaySim schema.
    """
    rng = np.random.default_rng(random_state)  # modern numpy random generator

    # ── 1. Transaction type ───────────────────────────────────────────────────
    # PaySim has 5 types. Their real-world frequencies are roughly:
    types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    probs = [0.339,     0.081,      0.353,      0.130,   0.097]
    transaction_types = rng.choice(types, size=n_rows, p=probs)

    # ── 2. Time step (1 hour = 1 step, simulating 31 days) ───────────────────
    steps = rng.integers(1, 745, size=n_rows)

    # ── 3. Transaction amounts (log-normal = realistic skewed distribution) ───
    # Most transactions are small; a few are very large. Log-normal captures this.
    amounts = np.exp(rng.normal(loc=6.0, scale=2.0, size=n_rows))
    amounts = np.clip(amounts, 1.0, 10_000_000.0).round(2)

    # ── 4. Account names ──────────────────────────────────────────────────────
    # In PaySim: 'C' prefix = customer, 'M' prefix = merchant
    n_customers = n_rows // 5  # pool of repeat customers (realistic)
    customer_pool = ['C' + str(rng.integers(100_000_000, 999_999_999)) for _ in range(n_customers)]

    orig_names = [customer_pool[i % n_customers] for i in range(n_rows)]

    # Destination: merchants for PAYMENT/DEBIT, customers for TRANSFER/CASH_OUT/CASH_IN
    merchant_pool = ['M' + str(rng.integers(100_000_000, 999_999_999)) for _ in range(1000)]
    dest_names = []
    for t in transaction_types:
        if t in ['TRANSFER', 'CASH_OUT', 'CASH_IN']:
            dest_names.append(customer_pool[rng.integers(0, n_customers)])
        else:
            dest_names.append(merchant_pool[rng.integers(0, 1000)])

    # ── 5. Balances ───────────────────────────────────────────────────────────
    old_balance_orig = np.exp(rng.normal(loc=7.5, scale=2.5, size=n_rows)).round(2)
    old_balance_dest = np.exp(rng.normal(loc=6.0, scale=2.5, size=n_rows)).round(2)

    # Normal transactions reduce sender balance by amount (can't go negative)
    new_balance_orig = np.maximum(0, old_balance_orig - amounts).round(2)
    new_balance_dest = (old_balance_dest + amounts).round(2)

    # ── 6. Fraud labels ───────────────────────────────────────────────────────
    # ~1.3% fraud rate, but ONLY in TRANSFER and CASH_OUT (PaySim finding)
    is_fraud = np.zeros(n_rows, dtype=np.int8)
    eligible_mask = np.isin(transaction_types, ['TRANSFER', 'CASH_OUT'])
    eligible_idx  = np.where(eligible_mask)[0]

    n_fraud = int(n_rows * 0.013)
    n_fraud = min(n_fraud, len(eligible_idx))  # can't have more fraud than eligible rows
    fraud_idx = rng.choice(eligible_idx, size=n_fraud, replace=False)
    is_fraud[fraud_idx] = 1

    # ── 7. Apply fraud balance pattern ───────────────────────────────────────
    # KEY INSIGHT: fraudsters drain accounts to 0 before moving money.
    # This "balance zeroing" is one of the strongest fraud signals.
    new_balance_orig[fraud_idx] = 0.0
    old_balance_orig[fraud_idx] = amounts[fraud_idx]  # old balance ≈ the amount stolen

    # ── 8. isFlaggedFraud — system's rule-based flag ──────────────────────────
    # In real PaySim, only transfers > 200,000 are flagged. Most fraud slips through.
    is_flagged_fraud = np.zeros(n_rows, dtype=np.int8)
    large_fraud_mask = (is_fraud == 1) & (amounts > 200_000)
    is_flagged_fraud[large_fraud_mask] = 1

    # ── 9. Assemble DataFrame ─────────────────────────────────────────────────
    df = pd.DataFrame({
        'step':            steps,
        'type':            transaction_types,
        'amount':          amounts,
        'nameOrig':        orig_names,
        'oldbalanceOrg':   old_balance_orig,
        'newbalanceOrig':  new_balance_orig,
        'nameDest':        dest_names,
        'oldbalanceDest':  old_balance_dest,
        'newbalanceDest':  new_balance_dest,
        'isFraud':         is_fraud,
        'isFlaggedFraud':  is_flagged_fraud,
    })

    return df


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_data(filepath: str | None = None, n_rows: int = 100_000) -> pd.DataFrame:
    """
    Load transaction data from a CSV file or generate synthetic data.

    Args:
        filepath: Path to a CSV file. If None, synthetic data is generated.
        n_rows:   Rows to generate (only used when filepath is None).

    Returns:
        A cleaned pandas DataFrame ready for feature engineering.
    """
    if filepath is not None:
        df = pd.read_csv(filepath)
        df = _validate_schema(df)
    else:
        df = generate_synthetic_data(n_rows=n_rows)

    df = _clean(df)
    return df


def _validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check that required columns exist. Rename 'isFlaggedFraud' if missing.
    This makes the code tolerant of slight column name variations.
    """
    required = {'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg',
                'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Some exports use 'isFlaggedFraud', others 'isFraudster' — normalise
    if 'isFlaggedFraud' not in df.columns:
        df['isFlaggedFraud'] = 0

    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data cleaning:
    - Drop duplicate rows
    - Ensure numeric columns are the right dtype
    - Cap extreme outlier amounts (above 99.9th percentile) to reduce noise
    """
    df = df.drop_duplicates()

    numeric_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                    'oldbalanceDest', 'newbalanceDest']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where numeric conversion failed (NaN)
    df = df.dropna(subset=numeric_cols)

    # Cap extreme amounts — amounts above 99.9th percentile are likely data errors
    cap = df['amount'].quantile(0.999)
    df['amount'] = df['amount'].clip(upper=cap)

    df = df.reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# QUICK SUMMARY STATS (used by the dashboard)
# ─────────────────────────────────────────────

def get_summary_stats(df: pd.DataFrame) -> dict:
    """
    Return a dictionary of key statistics for display in the Streamlit dashboard.
    """
    fraud_count = int(df['isFraud'].sum())
    total = len(df)
    return {
        'total_transactions':  total,
        'fraud_count':         fraud_count,
        'non_fraud_count':     total - fraud_count,
        'fraud_rate_pct':      round(fraud_count / total * 100, 3),
        'avg_amount':          round(df['amount'].mean(), 2),
        'median_amount':       round(df['amount'].median(), 2),
        'transaction_types':   df['type'].value_counts().to_dict(),
    }
