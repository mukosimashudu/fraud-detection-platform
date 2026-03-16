import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from src.database import read_transactions
from src.preprocessing import create_features
from src.config import MODEL_PATH, THRESHOLD_PATH


st.title("Fraud Investigation Console")

st.write(
    "Investigate suspicious transactions, explore patterns, "
    "and analyze fraud alerts."
)

# --------------------------------------------------
# Load model and threshold
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)

# --------------------------------------------------
# Load data
# --------------------------------------------------

df = read_transactions().copy()

df_features = create_features(df.copy())

if "Class" in df_features.columns:
    X = df_features.drop(columns=["Class"])
else:
    X = df_features

# --------------------------------------------------
# Score transactions
# --------------------------------------------------

probs = model.predict_proba(X)[:, 1]

df["fraud_probability"] = probs
df["predicted_fraud"] = (df["fraud_probability"] >= threshold).astype(int)

# --------------------------------------------------
# Filters
# --------------------------------------------------

st.subheader("Filter Transactions")

col1, col2, col3 = st.columns(3)

min_prob = col1.slider(
    "Minimum Fraud Probability",
    min_value=0.0,
    max_value=1.0,
    value=0.5
)

show_only_alerts = col2.checkbox("Show Only Fraud Alerts")

sample_size = col3.slider(
    "Max Rows to Display",
    min_value=50,
    max_value=1000,
    value=200
)

filtered = df[df["fraud_probability"] >= min_prob]

if show_only_alerts:
    filtered = filtered[filtered["predicted_fraud"] == 1]

filtered = filtered.sort_values(
    "fraud_probability",
    ascending=False
).head(sample_size)

# --------------------------------------------------
# Display table
# --------------------------------------------------

st.subheader("Suspicious Transactions")

st.dataframe(filtered)

# --------------------------------------------------
# Fraud probability distribution
# --------------------------------------------------

st.subheader("Fraud Probability Distribution")

fig = plt.figure(figsize=(8,4))

plt.hist(df["fraud_probability"], bins=40)

plt.xlabel("Fraud Probability")
plt.ylabel("Transactions")

st.pyplot(fig)

# --------------------------------------------------
# Top suspicious transactions
# --------------------------------------------------

st.subheader("Top Fraud Alerts")

top_alerts = df.sort_values(
    "fraud_probability",
    ascending=False
).head(10)

st.dataframe(top_alerts)

# --------------------------------------------------
# Transaction deep dive
# --------------------------------------------------

st.subheader("Transaction Deep Dive")

index = st.number_input(
    "Select Transaction Index",
    min_value=0,
    max_value=len(df)-1,
    value=0
)

transaction = df.iloc[[index]]

prob = transaction["fraud_probability"].values[0]

st.write("Fraud Probability:", round(prob,4))

if prob >= threshold:
    st.error("High Fraud Risk")
else:
    st.success("Low Fraud Risk")

st.dataframe(transaction)