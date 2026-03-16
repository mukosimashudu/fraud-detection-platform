import pandas as pd
import math
from sklearn.model_selection import train_test_split

from .config import TARGET_COL, TEST_SIZE, RANDOM_STATE


def create_features(df):
    """
    Feature engineering used during both training and inference.
    """

    df = df.copy()

    # Extract hour from transaction time
    df["hour"] = (df["Time"] // 3600) % 24

    # Log transform transaction amount
    df["log_amount"] = df["Amount"].apply(lambda x: math.log(x + 1))

    return df


def split_data(df):
    """
    Prepare data for model training.
    """

    # Apply feature engineering
    df = create_features(df)

    # Separate features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test