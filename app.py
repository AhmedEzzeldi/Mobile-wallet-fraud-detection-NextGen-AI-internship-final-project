"""
Mobile-Wallet Fraud Detection — Streamlit App
NextGen Institute Final Project · Ahmed Ezz Elaraby

Run locally with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Mobile-Wallet Fraud Detector", page_icon="🛡️", layout="wide")

# ----------------------------------------------------------------------
# Load model + supporting table (cached so it only loads once)
# ----------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load("fraud_detection_pipeline.joblib")

@st.cache_data
def load_cost_table():
    return pd.read_csv("threshold_cost_table.csv")

pipeline = load_pipeline()
cost_table = load_cost_table()

FEATURE_COLS = [
    'step', 'hour', 'type', 'amount',
    'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'destType', 'errorBalanceOrig', 'errorBalanceDest',
    'origBalanceZeroed', 'destZeroBeforeAfter'
]


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact same feature engineering used in the training notebook."""
    df = raw.copy()
    df['destType'] = df['nameDest'].astype(str).str[0] if 'nameDest' in df.columns else 'C'
    df['hour'] = df['step'] % 24
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    df['origBalanceZeroed'] = ((df['oldbalanceOrg'] > 0) & (df['newbalanceOrig'] == 0)).astype(int)
    df['destZeroBeforeAfter'] = ((df['oldbalanceDest'] == 0) & (df['newbalanceDest'] == 0) & (df['amount'] > 0)).astype(int)
    return df[FEATURE_COLS]


# ----------------------------------------------------------------------
# Sidebar: threshold slider with cost context
# ----------------------------------------------------------------------
st.sidebar.header("⚙️ Decision threshold")
threshold = st.sidebar.slider(
    "Flag as fraud when probability ≥", min_value=0.05, max_value=0.95, value=0.50, step=0.05
)

closest_row = cost_table.iloc[(cost_table['threshold'] - threshold).abs().argsort()[:1]]
st.sidebar.markdown(
    f"""**At this threshold (test-set estimate):**
- Missed fraud cases: **{int(closest_row['missed_fraud'].values[0])}**
- False alarms: **{int(closest_row['false_alarms'].values[0])}**
- Estimated total cost: **{int(closest_row['total_cost'].values[0]):,} EGP**
  *(illustrative: 5,000 EGP per missed fraud, 50 EGP per false alarm)*"""
)

best_threshold = cost_table.loc[cost_table['total_cost'].idxmin(), 'threshold']
st.sidebar.info(f"💡 Lowest-cost threshold from our test-set analysis: **{best_threshold}**")

st.sidebar.divider()
st.sidebar.caption(
    "This app scores transactions with a HistGradientBoosting model tuned on the PaySim "
    "mobile-money dataset. It is a course/portfolio project — not a production fraud system."
)

# ----------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------
st.title("🛡️ Mobile-Wallet Fraud Detector")
st.write(
    "Score a single transaction, or upload a CSV of transactions, and see the fraud risk "
    "plus the decision this threshold would make."
)

tab1, tab2 = st.tabs(["Single transaction", "Upload CSV"])

# --- Tab 1: single transaction form ---
with tab1:
    st.subheader("Transaction details")
    col1, col2, col3 = st.columns(3)

    with col1:
        step = st.number_input("Step (hour since simulation start)", min_value=1, max_value=744, value=200)
        txn_type = st.selectbox("Transaction type", ["CASH_OUT", "TRANSFER", "CASH_IN", "PAYMENT", "DEBIT"])
        amount = st.number_input("Amount", min_value=0.0, value=50000.0, step=100.0)

    with col2:
        oldbalanceOrg = st.number_input("Sender balance BEFORE", min_value=0.0, value=50000.0, step=100.0)
        newbalanceOrig = st.number_input("Sender balance AFTER", min_value=0.0, value=0.0, step=100.0)
        dest_type = st.selectbox("Receiver account type", ["Customer (C)", "Merchant (M)"])

    with col3:
        oldbalanceDest = st.number_input("Receiver balance BEFORE", min_value=0.0, value=0.0, step=100.0)
        newbalanceDest = st.number_input("Receiver balance AFTER", min_value=0.0, value=0.0, step=100.0)

    if st.button("🔍 Check this transaction", type="primary"):
        row = pd.DataFrame([{
            'step': step,
            'type': txn_type,
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest,
            'nameDest': 'C_PLACEHOLDER' if dest_type.startswith('Customer') else 'M_PLACEHOLDER',
        }])
        X_row = engineer_features(row)
        proba = pipeline.predict_proba(X_row)[0, 1]
        flagged = proba >= threshold

        st.metric("Fraud probability", f"{proba:.1%}")
        if flagged:
            st.error(f"🚨 FLAGGED — probability {proba:.1%} is at/above threshold {threshold:.0%}. "
                      "Recommended action: hold for manual review or block.")
        else:
            st.success(f"✅ Not flagged — probability {proba:.1%} is below threshold {threshold:.0%}. "
                        "Recommended action: approve.")

# --- Tab 2: batch CSV scoring ---
with tab2:
    st.subheader("Score a batch of transactions")
    st.caption(
        "CSV must have the same columns as the PaySim dataset: step, type, amount, nameOrig, "
        "oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest "
        "(isFraud/isFlaggedFraud are optional, ignored if present)."
    )
    uploaded = st.file_uploader("Upload CSV", type="csv")

    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        try:
            X_batch = engineer_features(raw)
            proba_batch = pipeline.predict_proba(X_batch)[:, 1]
            out = raw.copy()
            out['fraud_probability'] = proba_batch
            out['flagged'] = out['fraud_probability'] >= threshold

            n_flagged = int(out['flagged'].sum())
            st.write(f"**{n_flagged:,} of {len(out):,} transactions flagged** at threshold {threshold:.0%}.")
            st.dataframe(
                out.sort_values('fraud_probability', ascending=False)
                   .head(200)[['type', 'amount', 'fraud_probability', 'flagged']],
                use_container_width=True
            )
            csv_out = out.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download scored CSV", csv_out, "scored_transactions.csv", "text/csv")
        except KeyError as e:
            st.error(f"Missing required column: {e}. Please check your CSV matches the PaySim format.")

st.divider()
st.caption(
    "Model: HistGradientBoostingClassifier, tuned with RandomizedSearchCV. "
    "Built for the NextGen Institute Machine Learning final project. "
    "⚠️ Not medical/financial advice — a course portfolio project, not a production system."
)
