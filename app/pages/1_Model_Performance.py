import streamlit as st
import pandas as pd

st.title("Model Performance")

data = {
    "Model": ["LogisticRegression", "RandomForest", "XGBoost"],
    "Precision": [0.13, 0.61, 0.68],
    "Recall": [0.89, 0.86, 0.88],
    "F1": [0.23, 0.71, 0.77],
    "PR-AUC": [0.71, 0.82, 0.87]
}

df = pd.DataFrame(data)

st.dataframe(df)

st.bar_chart(df.set_index("Model"))