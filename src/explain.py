import shap
import joblib
import matplotlib.pyplot as plt

from src.config import MODEL_PATH
from src.database import read_transactions
from src.preprocessing import create_features


def explain_model():

    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)

    print("Loading transactions...")
    df = read_transactions()

    # Apply the SAME feature engineering used during training
    df = create_features(df)

    # Remove target column
    X = df.drop(columns=["Class"])

    # Sample data (SHAP is expensive)
    sample = X.sample(200, random_state=42)

    print("Creating SHAP explainer...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer(sample)

    print("Generating SHAP beeswarm plot...")

    shap.summary_plot(
        shap_values.values,
        sample
    )

    print("Generating SHAP feature importance...")

    shap.summary_plot(
        shap_values.values,
        sample,
        plot_type="bar"
    )

    plt.show()


if __name__ == "__main__":
    explain_model()