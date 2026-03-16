
# AI Fraud Detection Platform

An end-to-end **AI-powered fraud detection platform** that detects suspicious financial transactions, explains model decisions, and identifies coordinated fraud rings.

The system integrates **machine learning, explainable AI, graph analytics, real-time APIs, and interactive dashboards** to simulate how modern financial institutions detect and investigate fraud.

---

# Problem Statement

Financial institutions process millions of transactions daily, making manual fraud detection impractical. Traditional rule-based fraud systems struggle to detect sophisticated attacks, particularly when fraudsters operate through **coordinated networks of cards, devices, and merchants**.

This project addresses the problem by building an **AI-driven fraud detection system** capable of:

• Detecting fraudulent transactions using machine learning
• Explaining predictions using Explainable AI (SHAP)
• Identifying fraud rings using graph analytics
• Providing real-time scoring through an API
• Enabling fraud investigation through an analyst dashboard

The goal is to replicate the architecture used in **modern fintech fraud detection platforms**.

---

# System Architecture

The platform is built as a modular AI system combining **data engineering, machine learning, APIs, and visualization tools**.

```
Fraud Detection Platform
│
├── Data Pipeline
│   ├── Raw Dataset
│   ├── Feature Engineering
│   └── Processed Data
│
├── Machine Learning Pipeline
│   ├── SMOTE imbalance handling
│   ├── Model comparison
│   ├── Threshold optimization
│   └── Best model selection
│
├── MLflow Experiment Tracking
│
├── FastAPI Fraud Scoring Service
│
├── Streamlit Fraud Dashboard
│   ├── Model Performance
│   ├── Transaction Scoring
│   ├── Fraud Monitoring
│   ├── SHAP Explainability
│   ├── Fraud Forecasting
│   ├── Fraud Assistant (LLM)
│   ├── Fraud Ring Detection
│   └── Investigation Console
│
└── Graph Analytics
    └── Fraud community detection
```

---

# Key Achievements

This project demonstrates a **complete fraud detection ecosystem** including:

• Machine Learning Fraud Detection
• Explainable AI using SHAP
• Fraud Ring Detection using Graph Analytics
• Real-Time Fraud Scoring API
• Interactive Fraud Investigation Dashboard
• MLflow Experiment Tracking
• Dockerized Deployment

The platform simulates how **banks and fintech companies detect and investigate fraud in real time**.

---

# Model Performance

Multiple machine learning models were trained and compared.

Models evaluated:

• Logistic Regression
• Random Forest
• XGBoost (Calibrated)

Evaluation metrics:

• Precision
• Recall
• F1 Score
• ROC-AUC
• PR-AUC

The **best model is automatically selected using PR-AUC**, which is more appropriate for imbalanced fraud datasets.

Example results:

| Model               | Precision | Recall    | F1 Score | ROC-AUC   | PR-AUC    |
| ------------------- | --------- | --------- | -------- | --------- | --------- |
| Logistic Regression | High      | Moderate  | Balanced | Strong    | Good      |
| Random Forest       | Very High | Good      | Strong   | Excellent | Very Good |
| XGBoost             | Excellent | Excellent | Best     | Highest   | Highest   |

---

# Dashboard Features

The Streamlit dashboard functions as a **Fraud Monitoring Center**.

### Model Performance

Displays model evaluation metrics and comparison results.

### Transaction Scoring

Allows analysts to score individual transactions via the FastAPI service.

### Fraud Monitoring

Displays fraud statistics and dataset exploration.

### SHAP Explainability

Explains which features influenced fraud predictions.

### Fraud Forecasting

Identifies fraud trends and future risk patterns.

### Fraud Ring Detection

Uses graph analytics to identify suspicious networks of:

• credit cards
• merchants
• devices
• IP addresses

### Investigation Console

Allows fraud analysts to explore suspicious transactions and investigate alerts.

---

# Dashboard Preview

*(Add screenshots here after pushing the project)*

Example:

```
screenshots/
    dashboard_overview.png
    shap_explainability.png
    fraud_ring_detection.png
```

Example in README:

```markdown
![Dashboard Overview](screenshots/dashboard_overview.png)

![SHAP Explainability](screenshots/shap_explainability.png)

![Fraud Ring Detection](screenshots/fraud_ring_detection.png)```
```


# Real-Time Fraud Scoring API

The platform exposes a **FastAPI scoring service**.

Endpoint:

```
POST /score_transaction
```

Example response:

```json
{
  "fraud_probability": 0.91,
  "predicted_class": 1,
  "risk_level": "High"
}
```

Interactive API documentation:

```
http://localhost:8000/docs
```

---

# Dataset

The project uses the **Credit Card Fraud Detection Dataset**.

Download from Kaggle:

[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)


The dataset is **not included in this repository due to size restrictions**.

---

# Installation

Clone the repository:

```
git clone https://github.com/mukosimashudu/fraud-detection-platform.git
cd fraud-detection-platform
```

Install dependencies:

```
pip install -r requirements.txt
pip install -r requirements_api.txt
```

---

# Train the Model

```
python -m src.train
```

This will:

• prepare the dataset
• train multiple models
• select the best model
• save the model and threshold

---

# Run the API

```
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API docs:

```
http://localhost:8000/docs
```

---

# Run the Dashboard

```
streamlit run app/streamlit_app.py
```

Dashboard:

```
http://localhost:8501
```

---

# Docker Deployment

Build containers:

```
docker compose build
```

Start services:

```
docker compose up
```

Services:

Dashboard

```
http://localhost:8501
```

API

```
http://localhost:8000/docs
```

---

# Technologies Used

Python
Scikit-learn
XGBoost
Streamlit
FastAPI
MLflow
SHAP
NetworkX
PyVis
Docker

---

# Future Improvements

Planned upgrades include:

• Data drift monitoring
• Model registry
• Real-time fraud alerts
• Cloud deployment
• Automated retraining pipeline

---

# Author

Mashudu Mukosi

