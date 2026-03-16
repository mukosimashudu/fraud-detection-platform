import streamlit as st
import pandas as pd
import requests

from src.database import read_transactions


# =====================================
# CONFIG
# =====================================

API_URL = "http://127.0.0.1:8000/score_transaction"


# =====================================
# PAGE SETUP
# =====================================

st.set_page_config(
    page_title="AI Fraud Control Center",
    layout="wide"
)

st.title("🏦 AI Fraud Detection Platform")


# =====================================
# LOAD DATA
# =====================================

df = read_transactions()


# =====================================
# DASHBOARD STATS
# =====================================

st.subheader("Fraud Statistics")

total_transactions = len(df)
fraud_transactions = int(df["Class"].sum())
fraud_rate = fraud_transactions / total_transactions

col1, col2, col3 = st.columns(3)

col1.metric("Total Transactions", total_transactions)
col2.metric("Fraud Cases", fraud_transactions)
col3.metric("Fraud Rate", f"{fraud_rate:.4%}")


# =====================================
# DATA PREVIEW
# =====================================

st.subheader("Transaction Dataset")
st.dataframe(df.head(10))


# =====================================
# FRAUD DISTRIBUTION
# =====================================

st.subheader("Fraud Distribution")
st.bar_chart(df["Class"].value_counts())


# =====================================
# TRANSACTION SCORING VIA API
# =====================================

st.subheader("Transaction Scoring (FastAPI)")

transaction_index = st.number_input(
    "Select Transaction Index",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)

selected_transaction = df.iloc[[transaction_index]].copy()

st.write("Selected Transaction")
st.dataframe(selected_transaction)

if st.button("Score Transaction via API"):

    # Convert selected row to JSON payload
    payload = selected_transaction.drop(columns=["Class"]).iloc[0].to_dict()

    try:
        response = requests.post(API_URL, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()

            fraud_probability = result["fraud_probability"]
            threshold = result["threshold"]
            predicted_class = result["predicted_class"]
            risk_level = result["risk_level"]

            st.write("Fraud Probability:", fraud_probability)
            st.write("Fraud Threshold:", threshold)
            st.write("Predicted Class:", predicted_class)
            st.write("Risk Level:", risk_level)

            if predicted_class == 1:
                st.error(f"🚨 High Fraud Risk ({risk_level})")
            else:
                st.success(f"✅ Transaction appears normal ({risk_level})")

        else:
            st.error(f"API error: {response.status_code}")
            st.json(response.json())

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the FastAPI service. "
            "Make sure the API is running on http://127.0.0.1:8000"
        )

    except requests.exceptions.Timeout:
        st.error("The API request timed out.")

    except Exception as e:
        st.error(f"Unexpected error: {e}")


# =====================================
# API HEALTH CHECK
# =====================================

st.subheader("API Status")

if st.button("Check API Health"):
    try:
        health_response = requests.get("http://127.0.0.1:8000/health", timeout=10)

        if health_response.status_code == 200:
            st.success("✅ FastAPI service is running")
            st.json(health_response.json())
        else:
            st.warning(f"API returned status code {health_response.status_code}")

    except Exception as e:
        st.error(f"API is not reachable: {e}")