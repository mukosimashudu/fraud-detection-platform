import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
import joblib

from src.config import MODEL_PATH
from src.preprocessing import create_features
from src.database import read_transactions

st.title("Transaction Scoring")

model = joblib.load(MODEL_PATH)

df = read_transactions()

sample = df.iloc[[0]].copy()

sample = create_features(sample)

sample = sample.drop(columns=["Class"])

# enforce correct column order
sample = sample[model.feature_names_in_]

if st.button("Score Transaction"):

    prob = model.predict_proba(sample)[0][1]

    st.write("Fraud Probability:", prob)

    if prob > 0.5:
        st.error("High Fraud Risk")
    else:
        st.success("Low Fraud Risk")

