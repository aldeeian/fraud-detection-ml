# Financial Transaction Fraud Detection System

An end-to-end machine learning pipeline for detecting fraudulent financial transactions,
built with XGBoost, PyTorch LSTM, and an interactive Streamlit dashboard.

## Overview

This project simulates the kind of fraud detection work done at financial institutions.
It uses a synthetic dataset modelled after PaySim (a simulator of mobile money transactions)
and trains three models of increasing sophistication:

| Model | Type | Key Idea |
|---|---|---|
| Isolation Forest | Unsupervised | Detects anomalies without needing fraud labels |
| XGBoost + SMOTE | Supervised (Gradient Boosting) | Handles class imbalance with oversampling |
| LSTM (PyTorch) | Supervised (Deep Learning) | Learns patterns across feature sequences |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the dashboard

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## Project Structure

```
fraud-detection/
├── data/               # Raw and processed data files
├── models/             # Saved trained models (.joblib, .pt)
├── notebooks/          # Jupyter notebooks for exploration
├── src/
│   ├── etl.py          # Data loading and synthetic data generation
│   ├── features.py     # Feature engineering pipeline
│   ├── train.py        # Model training (all 3 models)
│   ├── evaluate.py     # Metrics: precision, recall, F1, AUC-ROC
│   └── predict.py      # Single-transaction and batch inference
├── app.py              # Streamlit dashboard (main entry point)
└── requirements.txt
```

## Dataset

The synthetic dataset is generated programmatically and mirrors the PaySim schema:

- **100,000 transactions** over a simulated 30-day period
- **~1.3% fraud rate** (realistic for financial systems)
- Fraud occurs only in `TRANSFER` and `CASH_OUT` transaction types
- Key signal: fraudsters drain accounts to zero before moving funds

Columns: `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`,
`nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`

## Feature Engineering

Beyond the raw columns, the pipeline engineers additional predictive features:

- **`balance_diff_orig`**: Expected vs. actual balance change (= 0 for fraud)
- **`balance_diff_dest`**: Same for destination account
- **`amount_to_balance_ratio`**: How much of the account is being moved
- **`is_high_amount`**: Flag for amounts above the 90th percentile
- **`hour_of_day`**: Derived from `step` (fraud often happens at unusual hours)

## Model Performance (typical ranges)

| Metric | Isolation Forest | XGBoost | LSTM |
|---|---|---|---|
| AUC-ROC | ~0.75 | ~0.97 | ~0.90 |
| Precision | ~0.05 | ~0.90 | ~0.75 |
| Recall | ~0.60 | ~0.85 | ~0.70 |

XGBoost wins on tabular data — this is expected. The LSTM demonstrates deep learning
capability and would outperform XGBoost on truly sequential account-level data.

## Tech Stack

- **Python 3.13** — Windows 11
- **pandas / numpy / scipy** — data wrangling
- **scikit-learn** — preprocessing, Isolation Forest, metrics
- **imbalanced-learn** — SMOTE oversampling
- **XGBoost** — gradient boosting classifier
- **PyTorch** — LSTM neural network
- **Streamlit + Plotly** — interactive dashboard

## Skills Demonstrated

- End-to-end ML pipeline design
- Handling severe class imbalance (SMOTE)
- Gradient boosting on tabular financial data
- Sequential modelling with LSTM
- Feature engineering for fraud signals
- Model evaluation with business-relevant metrics (recall matters more than accuracy in fraud)
- Interactive ML dashboard deployment
