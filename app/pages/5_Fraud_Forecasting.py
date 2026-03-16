import streamlit as st
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA

st.title("Fraud Forecasting")

# Dummy example
data = [10,12,9,15,14,18,20]

model = ARIMA(data, order=(2,1,2))
model_fit = model.fit()

forecast = model_fit.forecast(5)

st.write("Next Fraud Forecast:", forecast)