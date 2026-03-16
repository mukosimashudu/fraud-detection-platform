import pandas as pd
from sqlalchemy import create_engine

from src.config import DB_PATH, RAW_DATA_PATH


# create sqlite engine using config path
engine = create_engine(f"sqlite:///{DB_PATH}")


def load_dataset():
    """Load the credit card dataset"""
    df = pd.read_csv(RAW_DATA_PATH)
    return df


def create_database():
    """Create database and populate transactions table"""

    df = load_dataset()

    df.to_sql(
        "transactions",
        engine,
        if_exists="replace",
        index=False
    )

    print(f"Database created at: {DB_PATH}")
    print(f"Inserted rows: {len(df)}")


def read_transactions():
    """Read transactions from database"""

    df = pd.read_sql("SELECT * FROM transactions", engine)

    return df