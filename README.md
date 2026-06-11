# Financial Transaction Fraud Detection

[![tests](https://github.com/aldeeian/fraud-detection-ml/actions/workflows/tests.yml/badge.svg)](https://github.com/aldeeian/fraud-detection-ml/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Models](https://img.shields.io/badge/models-XGBoost%20%7C%20LSTM%20%7C%20Isolation%20Forest-green)

An end to end machine learning pipeline that detects fraudulent financial transactions, with an interactive Streamlit dashboard. Built on a synthetic dataset modelled after PaySim (mobile money transactions).

<!-- TODO: add a screenshot of the dashboard here after running it:
![dashboard](docs/dashboard.png) -->

## What this project does

1. Generates 100,000 realistic transactions with about 1.3% fraud
2. Engineers fraud signal features (balance discrepancies, drain ratios, time features)
3. Trains three models and compares them honestly
4. Serves everything in a dashboard where you can explore the data, tune the decision threshold, see the business impact in dollars, and score individual transactions

## The models

| Model | Type | Test AUC | Precision | Recall |
|---|---|---|---|---|
| Isolation Forest | Unsupervised baseline | ~0.85 | ~0.30 | ~0.44 |
| XGBoost + SMOTE | Supervised, gradient boosting | ~0.99 | ~0.87 | ~0.88 |
| LSTM (PyTorch) | Supervised, deep learning | run locally | | |

Numbers vary a little between runs because the dataset is regenerated each time. They come from a 20% held out test set the models never saw during training.

The ordering is the real lesson here. The unsupervised Isolation Forest only catches the obvious fraud (big anomalous transfers). XGBoost, which learns from labelled examples, does far better. That gap is exactly why banks invest in labelled fraud data and supervised models.

The LSTM needs PyTorch installed. If torch is missing the project still works and just skips the LSTM.

## The data leakage story (the part I am most proud of)

An earlier version of this project reported a perfect 1.0 AUC. That looked impressive until I dug into why. The synthetic data generator was setting fraud rows to have the old balance exactly equal to the amount and the new balance exactly 0. The model was not learning fraud patterns, it was reading an arithmetic fingerprint that only existed because of how I generated the data. A second leak was feeding the isFlaggedFraud column to the model as a feature, even though that column was computed from the fraud label itself.

How I fixed it:

1. Fraud balances are now noisy and partial (drains of 85 to 100%, amounts close to but not equal to the balance)
2. About 25% of frauds are "stealthy" and look like completely normal transactions, so no model can score 100% recall
3. Normal transactions also got realistic noise (balances that do not reconcile to the cent, some legitimate full account drains, and zero balance records like real PaySim has)
4. isFlaggedFraud was removed from the feature list
5. I added a unit test that trains a simple probe model and fails the build if AUC ever comes back as a perfect separator, so the leak cannot quietly return

After the fix, XGBoost dropped from a fake 1.0 to an honest ~0.99 AUC with a real precision and recall tradeoff. The metrics look slightly worse and are worth far more.

## Dashboard features

- Load synthetic data or upload your own PaySim style CSV
- Train all models with one click and compare them on a held out test set
- Decision threshold explorer: drag a slider and watch precision, recall, false alarms, and frauds caught update live
- Business impact card: estimated fraud losses prevented, losses missed, and analyst review cost in dollars at your chosen threshold
- ROC curves, metric comparison charts, confusion matrices, and XGBoost feature importance
- Single transaction scoring with one click example scenarios (typical fraud, normal payment, stealthy fraud)

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501, click Load Data in the sidebar, then Train Models.

PyTorch note: torch is only needed for the LSTM. If you skip it, install everything else and the app trains the other two models. For the CPU version of torch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Running the tests

```bash
pytest tests/ -v
```

10 tests cover the data generator (schema, fraud rate, reproducibility), the feature pipeline, and the data leakage guard described above. They run automatically in GitHub Actions on every push.

## Project structure

```
fraud-detection-ml/
├── app.py              Streamlit dashboard (entry point)
├── src/
│   ├── etl.py          Synthetic data generation and CSV loading
│   ├── features.py     Feature engineering
│   ├── train.py        All three models
│   ├── evaluate.py     Metrics and Plotly charts
│   └── predict.py      Single and batch inference
├── tests/
│   └── test_pipeline.py
└── .github/workflows/tests.yml
```

## Feature engineering

| Feature | Why it helps |
|---|---|
| balance_diff_orig | Whether the sender balance actually reconciles with the amount |
| balance_diff_dest | Same check for the recipient |
| amount_to_balance_ratio | How much of the account is being moved (drains are suspicious) |
| is_high_amount | Top 10% transaction amounts |
| hour_of_day | Fraud clusters at quiet hours |
| type_encoded | Fraud only occurs in TRANSFER and CASH_OUT |

isFlaggedFraud is deliberately excluded. See the data leakage section.

## Tech stack

Python, pandas, numpy, scikit-learn, imbalanced-learn (SMOTE), XGBoost, PyTorch (optional), Streamlit, Plotly, pytest, GitHub Actions.

## What I learned

- Perfect metrics are a warning sign, not a win. Always ask why a model is doing too well.
- Synthetic data design is as important as model design. If the classes are trivially separable, every model looks like a genius.
- Class imbalance needs deliberate handling: SMOTE for XGBoost, pos_weight in the loss for the LSTM.
- Precision and recall are a business decision, not just a metric. The threshold slider in the dashboard makes that tradeoff visible in dollars.
