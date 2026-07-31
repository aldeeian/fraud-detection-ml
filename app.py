"""
app.py — Streamlit Dashboard for Fraud Detection

Run with:  streamlit run app.py

Streamlit works by re-running this entire file top-to-bottom every time the
user clicks a button or changes an input. We use st.session_state (a dictionary
that persists between reruns) to store trained models, data, and results so we
don't have to retrain every time.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the project root to Python's path so `from src.xxx import` works
sys.path.insert(0, str(Path(__file__).parent))

from src.etl      import load_data, get_summary_stats
from src.features import get_X_y, FEATURE_COLUMNS
from src.train    import train_all
from src.evaluate import (
    evaluate_all_models, plot_confusion_matrix,
    plot_metrics_comparison, plot_feature_importance,
    plot_class_distribution, plot_roc_curves,
)
from src.predict  import predict_single


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  (dark header bar, metric cards)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 6px 0;
    }
    .metric-label { font-size: 0.85rem; color: #aaa; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #fff; }
    .fraud-badge {
        background: #e84c4c; color: white;
        padding: 6px 16px; border-radius: 20px;
        font-weight: bold; font-size: 1.1rem;
    }
    .safe-badge {
        background: #4caf50; color: white;
        padding: 6px 16px; border-radius: 20px;
        font-weight: bold; font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Data source selection
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Fraud Detector")
    st.markdown("---")
    st.subheader("1. Data Source")

    data_source = st.radio(
        "Choose data:",
        ["Generate Synthetic Data", "Upload CSV File"],
        help="Synthetic data matches the PaySim schema with 100,000 rows and ~1.3% fraud rate."
    )

    if data_source == "Upload CSV File":
        uploaded = st.file_uploader(
            "Upload PaySim-format CSV",
            type=['csv'],
            help="Must have columns: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud"
        )
    else:
        uploaded = None
        n_rows = st.slider(
            "Number of rows to generate",
            min_value=10_000, max_value=200_000,
            value=100_000, step=10_000,
        )

    load_btn = st.button("Load Data", type="primary", use_container_width=True)

    st.markdown("---")
    st.subheader("2. Train Models")
    train_btn = st.button(
        "Train Models",
        type="primary",
        use_container_width=True,
        disabled=('df' not in st.session_state),
        help="Load data first, then train.",
    )

    st.markdown("---")
    st.caption("Built with XGBoost · PyTorch LSTM · Isolation Forest")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
if load_btn:
    with st.spinner("Loading data..."):
        if uploaded is not None:
            df = load_data(filepath=uploaded)
        else:
            df = load_data(n_rows=n_rows)

        st.session_state['df'] = df
        st.session_state['stats'] = get_summary_stats(df)
        # Clear previous training results when new data is loaded
        for key in ['train_results', 'eval_results']:
            st.session_state.pop(key, None)

    # The sidebar was rendered BEFORE the data existed, so the Train button is
    # still disabled on this run. Rerun so it re-renders enabled immediately.
    st.rerun()

if 'df' in st.session_state and 'train_results' not in st.session_state and not train_btn:
    st.success(f"Loaded {len(st.session_state['df']):,} transactions. Ready to train.")


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN ALL MODELS
# ─────────────────────────────────────────────────────────────────────────────
if train_btn and 'df' in st.session_state:
    df = st.session_state['df']

    with st.spinner("Engineering features..."):
        X, y = get_X_y(df)

    progress_bar = st.progress(0, text="Starting training...")

    # train_all reports real progress (including per-epoch LSTM updates)
    results = train_all(
        X, y, feature_names=FEATURE_COLUMNS,
        progress_callback=lambda frac, text: progress_bar.progress(frac, text=text),
    )

    progress_bar.progress(0.92, text="Evaluating on test set...")
    eval_results = evaluate_all_models(results)
    progress_bar.progress(1.0, text="Done!")

    st.session_state['train_results'] = results
    st.session_state['eval_results']  = eval_results
    st.success("All three models trained and evaluated!")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT — Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Overview",
    "🤖 Model Results",
    "📈 Charts & Analysis",
    "🔮 Predict Transaction",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if 'df' not in st.session_state:
        st.info("👈 Load data using the sidebar to get started.")
    else:
        df    = st.session_state['df']
        stats = st.session_state['stats']

        st.header("Data Overview")

        # ── KPI metric cards ─────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transactions", f"{stats['total_transactions']:,}")
        c2.metric("Fraud Cases",        f"{stats['fraud_count']:,}")
        c3.metric("Fraud Rate",         f"{stats['fraud_rate_pct']:.3f}%")
        c4.metric("Avg. Amount",        f"${stats['avg_amount']:,.2f}")

        st.markdown("---")

        # ── Class distribution chart ─────────────────────────────────────────
        col_chart, col_types = st.columns(2)
        with col_chart:
            fig_dist = plot_class_distribution(df)
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_types:
            st.subheader("Transactions by Type")
            type_counts = df['type'].value_counts().reset_index()
            type_counts.columns = ['Type', 'Count']

            # Highlight fraud rate per type
            fraud_by_type = df.groupby('type')['isFraud'].mean().reset_index()
            fraud_by_type.columns = ['Type', 'Fraud Rate']
            fraud_by_type['Fraud Rate'] = (fraud_by_type['Fraud Rate'] * 100).round(2)

            merged = type_counts.merge(fraud_by_type, on='Type')
            st.dataframe(merged, use_container_width=True, hide_index=True)
            st.caption("Fraud only occurs in TRANSFER and CASH_OUT — a key signal.")

        st.markdown("---")

        # ── Raw data preview ─────────────────────────────────────────────────
        st.subheader("Sample Transactions")
        col_f, col_n = st.columns(2)
        with col_f:
            st.write("**Fraudulent Transactions (sample)**")
            fraud_sample = df[df['isFraud'] == 1].head(5)
            st.dataframe(fraud_sample[['step','type','amount','oldbalanceOrg',
                                        'newbalanceOrig','isFraud']],
                         use_container_width=True, hide_index=True)
        with col_n:
            st.write("**Legitimate Transactions (sample)**")
            legit_sample = df[df['isFraud'] == 0].head(5)
            st.dataframe(legit_sample[['step','type','amount','oldbalanceOrg',
                                        'newbalanceOrig','isFraud']],
                         use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Raw Statistics")
        st.dataframe(df.describe(), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if 'eval_results' not in st.session_state:
        if 'df' not in st.session_state:
            st.info("👈 Load data first, then train models.")
        else:
            st.info("👈 Click 'Train Models' in the sidebar.")
    else:
        eval_results = st.session_state['eval_results']

        st.header("Model Evaluation Results")
        st.caption("Evaluated on 20% held-out test set (not seen during training).")

        # ── Metrics table ────────────────────────────────────────────────────
        rows = []
        for model_name, result in eval_results.items():
            m = result['metrics']
            rows.append({
                'Model':     model_name,
                'Precision': m['precision'],
                'Recall':    m['recall'],
                'F1-Score':  m['f1'],
                'AUC-ROC':   m['auc_roc'],
            })
        metrics_df = pd.DataFrame(rows).set_index('Model')
        st.dataframe(
            metrics_df.style.highlight_max(axis=0, color='#d4edda').format("{:.4f}"),
            use_container_width=True,
        )

        st.markdown("""
        **Reading the metrics:**
        - **Precision** — When the model says "fraud", how often is it right?
        - **Recall** — Of all actual frauds, how many did it catch? *(most important for banks)*
        - **F1** — Harmonic mean of Precision and Recall (balance between the two)
        - **AUC-ROC** — Overall ranking quality; 1.0 = perfect, 0.5 = random
        """)

        st.markdown("---")

        # ── Interactive decision threshold ───────────────────────────────────
        st.subheader("Decision Threshold Explorer")
        st.caption(
            "A model outputs a fraud probability. The THRESHOLD decides when to act on it. "
            "Lower threshold = catch more fraud but more false alarms. Drag the slider to "
            "see the tradeoff live on the held-out test set (XGBoost)."
        )
        xgb_res   = eval_results['XGBoost']
        y_t       = xgb_res['y_test']
        y_p       = xgb_res['y_proba']
        threshold = st.slider("Fraud probability threshold", 0.05, 0.95, 0.50, 0.05)
        y_hat = (y_p >= threshold).astype(int)

        tp = int(((y_hat == 1) & (y_t == 1)).sum())
        fp = int(((y_hat == 1) & (y_t == 0)).sum())
        fn = int(((y_hat == 0) & (y_t == 1)).sum())
        prec_t = tp / (tp + fp) if (tp + fp) else 0.0
        rec_t  = tp / (tp + fn) if (tp + fn) else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frauds caught", f"{tp} / {tp + fn}")
        c2.metric("False alarms", f"{fp:,}")
        c3.metric("Precision", f"{prec_t:.1%}")
        c4.metric("Recall", f"{rec_t:.1%}")

        # ── Business impact estimate ─────────────────────────────────────────
        df_loaded = st.session_state.get('df')
        if df_loaded is not None:
            fraud_amts = df_loaded.loc[df_loaded['isFraud'] == 1, 'amount']
            avg_fraud  = float(fraud_amts.mean()) if len(fraud_amts) else 0.0
            review_cost = 15.0  # rough cost of an analyst reviewing one alert
            saved   = tp * avg_fraud
            missed  = fn * avg_fraud
            alerts  = (tp + fp) * review_cost

            st.markdown("##### Estimated business impact at this threshold")
            b1, b2, b3 = st.columns(3)
            b1.metric("Fraud losses prevented", f"${saved:,.0f}")
            b2.metric("Fraud losses missed", f"${missed:,.0f}")
            b3.metric("Alert review cost", f"${alerts:,.0f}",
                      help="Assumes about $15 of analyst time per flagged transaction")
            st.caption(
                "Based on the average fraud amount in the loaded dataset "
                f"(${avg_fraud:,.0f}). This is the tradeoff fraud teams tune every day: "
                "moving the threshold trades analyst time against missed fraud."
            )

        st.markdown("---")

        # ── Per-model confusion matrices ─────────────────────────────────────
        st.subheader("Confusion Matrices")
        st.caption(
            "Each cell shows the count, the share of that row, and what it means. "
            "Colors are normalised per row so the rare fraud class stays visible. "
            "The bottom row is the one that matters: Caught = fraud detected, Missed = fraud that got through."
        )
        cols = st.columns(len(eval_results))
        for i, (model_name, result) in enumerate(eval_results.items()):
            with cols[i]:
                fig = plot_confusion_matrix(result['y_test'], result['y_pred'], model_name)
                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: CHARTS & ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if 'eval_results' not in st.session_state:
        st.info("Train the models first to see analysis charts.")
    else:
        eval_results  = st.session_state['eval_results']
        train_results = st.session_state['train_results']

        st.header("Charts & Analysis")

        # ── Bar chart: all metrics side by side ──────────────────────────────
        fig_compare = plot_metrics_comparison(eval_results)
        st.plotly_chart(fig_compare, use_container_width=True)

        st.markdown("---")

        # ── ROC Curves ───────────────────────────────────────────────────────
        fig_roc = plot_roc_curves(eval_results)
        st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown("---")

        # ── XGBoost Feature Importance ───────────────────────────────────────
        st.subheader("XGBoost Feature Importance")
        st.caption("Which features does the model rely on most to spot fraud?")
        fig_imp = plot_feature_importance(train_results['xgboost'], FEATURE_COLUMNS)
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("""
        **Interpreting feature importance:**
        - `balance_diff_orig` is usually the top feature — fraudsters drain accounts to 0
        - `amount_to_balance_ratio` — moving 100% of your balance is suspicious
        - `type_encoded` — fraud only happens in TRANSFER/CASH_OUT
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: PREDICT A SINGLE TRANSACTION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Predict a Single Transaction")
    st.caption("Fill in the transaction details below and all trained models will score it.")

    if 'train_results' not in st.session_state:
        st.warning("Train the models first (sidebar) before making predictions.")
    else:
        # ── One-click example scenarios (fill the form automatically) ───────
        st.markdown("**Quick examples** (click to fill the form):")
        pcol1, pcol2, pcol3 = st.columns(3)
        if pcol1.button("🚨 Typical fraud", use_container_width=True,
                        help="Large cash-out that drains most of the sender's balance; "
                             "the mule account's recorded balance doesn't reflect the money"):
            st.session_state.update({
                "p_step": 420, "p_type": "CASH_OUT", "p_amount": 190000.0,
                "p_old_orig": 205000.0, "p_new_orig": 18500.0,
                "p_old_dest": 400.0, "p_new_dest": 750.0,
            })
        if pcol2.button("✅ Normal payment", use_container_width=True,
                        help="Small everyday payment, balances reconcile"):
            st.session_state.update({
                "p_step": 300, "p_type": "PAYMENT", "p_amount": 50.0,
                "p_old_orig": 2000.0, "p_new_orig": 1950.0,
                "p_old_dest": 500.0, "p_new_dest": 550.0,
            })
        if pcol3.button("🕵️ Stealthy fraud", use_container_width=True,
                        help="Fraud disguised as a normal transfer, the hard case"):
            st.session_state.update({
                "p_step": 660, "p_type": "CASH_OUT", "p_amount": 800.0,
                "p_old_orig": 3500.0, "p_new_orig": 2700.0,
                "p_old_dest": 100.0, "p_new_dest": 900.0,
            })

        # ── Input form ───────────────────────────────────────────────────────
        with st.form("predict_form"):
            st.subheader("Transaction Details")

            col1, col2 = st.columns(2)

            with col1:
                t_step   = st.number_input("Step (hour of simulation)", min_value=1, max_value=744, value=100, key="p_step")
                t_type   = st.selectbox("Transaction Type", ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], key="p_type")
                t_amount = st.number_input("Amount ($)", min_value=0.0, value=5000.0, step=100.0, key="p_amount")
                t_name_orig = st.text_input("Sender Account (nameOrig)", value="C123456789")

            with col2:
                t_old_orig = st.number_input("Sender Old Balance ($)", min_value=0.0, value=5000.0, step=100.0, key="p_old_orig")
                t_new_orig = st.number_input("Sender New Balance ($)", min_value=0.0, value=0.0, step=100.0,
                                              help="Fraud typically drains 85-100% of the balance, leaving a small remainder",
                                              key="p_new_orig")
                t_name_dest   = st.text_input("Recipient Account (nameDest)", value="C987654321")
                t_old_dest    = st.number_input("Recipient Old Balance ($)", min_value=0.0, value=0.0, step=100.0, key="p_old_dest")
                t_new_dest    = st.number_input("Recipient New Balance ($)", min_value=0.0, value=5000.0, step=100.0, key="p_new_dest")

            t_flagged = st.checkbox("System-flagged as suspicious (isFlaggedFraud)", value=False)

            submitted = st.form_submit_button("Run Fraud Check", type="primary", use_container_width=True)

        # ── Run prediction ────────────────────────────────────────────────────
        if submitted:
            transaction = {
                'step':           t_step,
                'type':           t_type,
                'amount':         t_amount,
                'nameOrig':       t_name_orig,
                'oldbalanceOrg':  t_old_orig,
                'newbalanceOrig': t_new_orig,
                'nameDest':       t_name_dest,
                'oldbalanceDest': t_old_dest,
                'newbalanceDest': t_new_dest,
                'isFlaggedFraud': int(t_flagged),
                'isFraud':        0,  # dummy — not used during inference
            }

            train_results = st.session_state['train_results']
            models = {
                'iso_forest': train_results.get('iso_forest'),
                'xgboost':    train_results.get('xgboost'),
                'lstm':       train_results.get('lstm'),
                'scaler':     train_results.get('scaler'),
            }

            with st.spinner("Running models..."):
                pred_results = predict_single(transaction, models)

            st.markdown("---")
            st.subheader("Prediction Results")

            result_cols = st.columns(len(pred_results))
            for i, (model_name, result) in enumerate(pred_results.items()):
                with result_cols[i]:
                    prob  = result['probability']
                    label = result['prediction']
                    badge_class = 'fraud-badge' if label == 'FRAUD' else 'safe-badge'

                    st.markdown(f"**{model_name}**")
                    st.markdown(
                        f"<div class='{badge_class}'>{label}</div>",
                        unsafe_allow_html=True
                    )
                    st.metric("Fraud Probability", f"{prob:.1%}")

                    # Visual probability bar
                    st.progress(float(prob), text=f"{prob:.1%} fraud risk")

            st.markdown("---")

            # ── Explain the key features for this transaction ─────────────────
            balance_diff = (t_old_orig - t_amount) - t_new_orig
            amount_ratio = t_amount / (t_old_orig + 1e-9)

            st.subheader("Why these predictions?")
            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                status = "🚨 High" if abs(balance_diff) > 100 else "✅ Normal"
                st.metric("Balance Discrepancy",
                          f"${balance_diff:,.2f}",
                          delta=status,
                          delta_color="inverse")
            with col_e2:
                ratio_pct = amount_ratio * 100
                status = "🚨 Suspicious" if ratio_pct > 90 else "✅ Normal"
                st.metric("Amount / Balance Ratio",
                          f"{ratio_pct:.1f}%",
                          delta=status,
                          delta_color="inverse")
            with col_e3:
                fraud_type = t_type in ['TRANSFER', 'CASH_OUT']
                st.metric("Fraud-Prone Type",
                          "YES 🚨" if fraud_type else "NO ✅",
                          delta="TRANSFER/CASH_OUT types are high risk" if fraud_type else "Low-risk type")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Financial Transaction Fraud Detection System · "
    "Built with Streamlit, XGBoost, PyTorch · "
    "Dataset: Synthetic PaySim"
)
