import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd

from src.database import read_transactions

st.title("Fraud Monitoring")

df = read_transactions()

df["hour"] = (df["Time"] // 3600) % 24

fraud_by_hour = df.groupby("hour")["Class"].sum()

st.line_chart(fraud_by_hour)