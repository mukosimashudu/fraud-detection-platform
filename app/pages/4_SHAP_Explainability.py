import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import streamlit as st
import shap
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from src.database import read_transactions
from src.preprocessing import create_features
from src.config import MODEL_PATH


st.title("SHAP Explainability")
st.write("SHAP explains how each feature influences the fraud prediction.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def get_fitted_base_model(model):
    """
    Extract the fitted underlying estimator for SHAP.
    Handles CalibratedClassifierCV, pipelines, and normal estimators.
    """
    base_model = model

    # CalibratedClassifierCV fitted model
    if hasattr(base_model, "calibrated_classifiers_") and len(base_model.calibrated_classifiers_) > 0:
        calibrated = base_model.calibrated_classifiers_[0]

        if hasattr(calibrated, "estimator"):
            base_model = calibrated.estimator
        elif hasattr(calibrated, "base_estimator"):
            base_model = calibrated.base_estimator

    # Pipeline -> final model step
    if hasattr(base_model, "named_steps"):
        step_names = list(base_model.named_steps.keys())
        if len(step_names) > 0:
            base_model = base_model.named_steps[step_names[-1]]

    return base_model


@st.cache_data
def load_sample_data(n=200):
    df = read_transactions().copy()
    sample = df.sample(min(n, len(df)), random_state=42).copy()
    sample = create_features(sample)

    if "Class" in sample.columns:
        sample = sample.drop(columns=["Class"])

    return sample


def align_features(X, model):
    """
    Align dataframe columns to the exact order expected by the fitted model.
    """
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)

        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0

        X = X[expected_cols]

    return X


def get_explainer(model, background_data):
    """
    Choose a SHAP explainer that works for the fitted model.
    """
    model_name = model.__class__.__name__.lower()

    tree_keywords = ["xgb", "randomforest", "extratrees", "decisiontree", "gradientboosting", "lgbm"]

    if any(keyword in model_name for keyword in tree_keywords):
        return shap.TreeExplainer(model)

    # fallback for non-tree models
    background = background_data.sample(min(100, len(background_data)), random_state=42)
    return shap.Explainer(model.predict, background)


try:
    # -------------------------
    # Load model
    # -------------------------
    model = load_model()
    base_model = get_fitted_base_model(model)

    # -------------------------
    # Load and prepare data
    # -------------------------
    sample = load_sample_data(200)
    sample = align_features(sample, base_model)

    # -------------------------
    # Build SHAP explainer
    # -------------------------
    explainer = get_explainer(base_model, sample)

    # For tree models
    if isinstance(explainer, shap.TreeExplainer):
        shap_values = explainer.shap_values(sample)

        # Binary classification can return list
        if isinstance(shap_values, list):
            shap_values_for_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_values_for_plot = shap_values
    else:
        explanation = explainer(sample)
        shap_values_for_plot = explanation.values

    # -------------------------
    # Global feature importance
    # -------------------------
    st.subheader("Global Feature Importance")

    st.markdown(
        """
        <style>
        .stPlotlyChart, .stImage, canvas {
            max-width: 800px !important;
            margin: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    fig1 = plt.figure(figsize=(8, 5))

    shap.summary_plot(
        shap_values_for_plot,
        sample,
        max_display=10,
        show=False,
        plot_size=(8, 5)
    )

    st.pyplot(fig1, clear_figure=True)

    # -------------------------
    # Individual transaction explanation
    # -------------------------
    st.subheader("Explain Individual Transaction")

    row_index = st.slider(
        "Select transaction row",
        min_value=0,
        max_value=len(sample) - 1,
        value=0
    )

    transaction = sample.iloc[[row_index]].copy()

    prediction = base_model.predict(transaction)[0]

    fraud_probability = None
    if hasattr(base_model, "predict_proba"):
        fraud_probability = float(base_model.predict_proba(transaction)[0][1])

    col1, col2 = st.columns(2)
    col1.metric("Prediction", "Fraud" if prediction == 1 else "Legitimate")
    if fraud_probability is not None:
        col2.metric("Fraud Probability", f"{fraud_probability:.4f}")

    # -------------------------
    # Local SHAP values
    # -------------------------
    if isinstance(explainer, shap.TreeExplainer):
        single_shap = explainer.shap_values(transaction)

        if isinstance(single_shap, list):
            single_shap_values = single_shap[1][0] if len(single_shap) > 1 else single_shap[0][0]
            expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
        else:
            single_shap_values = single_shap[0]
            expected_value = explainer.expected_value
    else:
        single_exp = explainer(transaction)
        single_shap_values = single_exp.values[0]
        try:
            expected_value = single_exp.base_values[0]
        except Exception:
            expected_value = 0.0

    # Waterfall plot
    st.subheader("Waterfall Explanation")

    waterfall_explanation = shap.Explanation(
        values=single_shap_values,
        base_values=expected_value,
        data=transaction.iloc[0].values,
        feature_names=transaction.columns.tolist()
    )

    fig2 = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(waterfall_explanation, max_display=12, show=False)
    st.pyplot(fig2, clear_figure=True)

    # -------------------------
    # Feature contribution table
    # -------------------------
    st.subheader("Feature Contributions")

    contribution_df = pd.DataFrame({
        "Feature": transaction.columns,
        "Feature Value": transaction.iloc[0].values,
        "SHAP Impact": single_shap_values
    })

    contribution_df["Absolute Impact"] = contribution_df["SHAP Impact"].abs()
    contribution_df = contribution_df.sort_values("Absolute Impact", ascending=False)

    st.dataframe(
        contribution_df[["Feature", "Feature Value", "SHAP Impact"]].reset_index(drop=True),
        use_container_width=True
    )

except Exception as e:
    st.error(f"Unable to generate SHAP plot: {e}")