import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

from src.database import read_transactions


def fraud_time_series():

    df = read_transactions()

    # Create hour feature
    df["hour"] = (df["Time"] // 3600) % 24

    # Count fraud per hour
    fraud_by_hour = df.groupby("hour")["Class"].sum()

    return fraud_by_hour


def forecast_fraud():

    fraud_series = fraud_time_series()

    model = ARIMA(fraud_series, order=(2,1,2))

    model_fit = model.fit()

    forecast = model_fit.forecast(steps=5)

    print("Fraud forecast:", forecast)

    return fraud_series, forecast


def plot_forecast():

    fraud_series, forecast = forecast_fraud()

    plt.figure(figsize=(10,5))

    plt.plot(fraud_series, label="Historical Fraud")

    future_index = range(len(fraud_series), len(fraud_series)+len(forecast))

    plt.plot(future_index, forecast, label="Forecast")

    plt.legend()

    plt.title("Fraud Trend Forecast")

    plt.show()


if __name__ == "__main__":
    plot_forecast()