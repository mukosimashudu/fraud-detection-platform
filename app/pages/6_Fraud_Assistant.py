import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
from src.llm_assistant import explain_transaction

st.title("AI Fraud Analyst Assistant")

transaction_text = st.text_area(
    "Describe transaction:",
    height=180,
    placeholder="Card country: South Africa\nIP location: Eastern Europe\nTime: 03:14 AM\nDevice: New device"
)

if st.button("Analyze Transaction"):
    if not transaction_text.strip():
        st.warning("Please enter transaction details first.")
    else:
        explanation = explain_transaction(transaction_text)
        st.write(explanation)