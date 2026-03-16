import time
import random
import requests
import pandas as pd

from src.database import read_transactions

API_URL = "http://127.0.0.1:8000/score_transaction"


def simulate_transactions():

    df = read_transactions()

    print("Starting transaction stream...\n")

    while True:

        # randomly select a transaction
        sample = df.sample(1).iloc[0]

        transaction = sample.to_dict()

        try:

            response = requests.post(API_URL, json=transaction)

            result = response.json()

            print("Transaction scored")

            print("Fraud Probability:", result["fraud_probability"])
            print("Risk Level:", result["risk_level"])
            print("-" * 40)

        except Exception as e:

            print("API connection error:", e)

        # simulate delay between transactions
        time.sleep(random.uniform(1, 3))


if __name__ == "__main__":
    simulate_transactions()