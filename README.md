# 🛡️ Mobile-Wallet Fraud Detection

Mobile-money services are growing fast — and so is fraud. This project detects fraudulent transfers and
cash-outs among mobile-wallet transactions, using a **HistGradientBoostingClassifier** tuned with
`RandomizedSearchCV`, deployed as an interactive **Streamlit** app.

**🔗 Live app:** `[add your Streamlit Community Cloud link here after deploying]`
**📓 Notebook:** [`Mobile_Wallet_Fraud_Detection.ipynb`](./Mobile_Wallet_Fraud_Detection.ipynb)

![App screenshot](./screenshot.png)
*(replace with a real screenshot or GIF of the running app before submitting)*

---

## The problem

A mobile-money provider processes millions of transfers and cash-outs a day. Fraud (account takeover + mule
accounts) drains customer balances and erodes trust. The goal: score every transaction's fraud risk in real
time, and route high-risk transactions to a block or a manual-review queue instead of letting them through.

## Dataset

**PaySim Mobile Money Simulation** ([Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)) — simulated from
real mobile-money transaction logs, 11 columns, originally **6,362,620 transactions**.

For this project we trained on a **well-chosen stratified sample of 308,213 transactions**: all 8,213 real
fraud cases (100% of them) + 300,000 randomly sampled genuine transactions. This keeps every fraud example
while making the project tractable to train, tune, and iterate on quickly — a standard practice when working
with multi-million-row datasets.

## Results

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Baseline (always "genuine") | – | – | 0.000 | 0.000 | 0.000 |
| Logistic Regression | 0.9957 | 0.9499 | 0.655 | 0.494 | 0.973 |
| Random Forest | 0.9999 | 0.9995 | 0.998 | 0.997 | 0.999 |
| HistGradientBoosting (untuned) | 1.0000 | 0.9996 | 0.997 | 0.995 | 0.999 |
| **HistGradientBoosting (tuned) — final model** | **1.0000** | **0.9997** | **0.997** | **0.995** | **0.999** |

On the untouched 20% test set (61,643 transactions), the final model misses only **2 fraud cases** out of
1,643 and raises only **9 false alarms** out of 59,991 genuine transactions.

> **Why the numbers look this high:** PaySim's fraud-generation logic is itself a deterministic
> balance-draining pattern, and a handful of engineered features (sender balance fully drained,
> balance-accounting mismatches on the receiving side) capture that rule almost exactly. Real-world fraud is
> noisier and adversarial, so production performance would be lower — see the notebook's Limitations section.

## What's in this repo

```
Mobile_Wallet_Fraud_Detection.ipynb   # full pipeline: EDA → preprocessing → modeling → tuning → evaluation
app.py                                # Streamlit app (single-transaction form + batch CSV scoring)
fraud_detection_pipeline.joblib       # the final tuned, trained scikit-learn pipeline
threshold_cost_table.csv              # cost-based threshold analysis, used by the app's slider
requirements.txt                      # exact dependencies
README.md
```

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`). Use the sidebar slider to move the
fraud-decision threshold and see the live cost trade-off, or upload a CSV of transactions to score in batch.

To reproduce the full analysis: open `Mobile_Wallet_Fraud_Detection.ipynb` in Jupyter and run top-to-bottom
(`Kernel → Restart & Run All`). The raw `paysim_sample.csv` must be in the same folder.

## Key insights (full detail in the notebook)

- Fraud **only** occurs in `TRANSFER` and `CASH_OUT` transactions — the other three types (`CASH_IN`, `DEBIT`,
  `PAYMENT`) have zero fraud cases.
- **97.6%** of fraud cases drain the sender's balance to exactly zero, vs. ~24% of genuine transactions.
- Fraud never targets merchant accounts — only customer-to-customer transfers.
- A cost-based threshold analysis (illustrative: 5,000 EGP per missed fraud, 50 EGP per false alarm) shows the
  lowest-cost operating point sits *below* the default 0.5 cutoff, because missed fraud is far more expensive
  than a false alarm.

## Limitations

- Trained on a simulated dataset — real fraud is more adversarial and would need richer signals (device,
  geolocation, velocity) for production use.
- Trained on a sample, not the full 6.36M rows, for tractability.
- This is a learning/portfolio project, not a production-ready fraud system.

---

Built by **Ahmed Ezz Elaraby** for the NextGen Institute Machine Learning internship — final project.
