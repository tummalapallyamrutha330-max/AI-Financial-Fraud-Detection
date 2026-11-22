# 🛡️ FraudShield AI: Production-Grade Financial Fraud Detection & Risk Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-EB4034?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SQLite](https://img.shields.io/badge/SQLite-SQLAlchemy-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![Tests](https://img.shields.io/badge/Tests-23%20Passed-10B981?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org)

An end-to-end, full-stack Machine Learning application and financial analytics dashboard engineered to analyze credit card transactions, detect fraudulent activity in real time, calculate calibrated risk probabilities, and deliver mathematically grounded explainability through SHAP.

Designed specifically as a portfolio centerpiece, demonstrating production engineering standards, data leakage prevention, class imbalance management, and clean architecture.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [Project Objectives](#-project-objectives)
4. [Key Features](#-key-features)
5. [Technology Stack](#-technology-stack)
6. [System Architecture](#-system-architecture)
7. [Dataset Information](#-dataset-information)
8. [Machine Learning Workflow & Leakage Prevention](#-machine-learning-workflow--leakage-prevention)
9. [Empirical Model Evaluation Results](#-empirical-model-evaluation-results)
10. [Dashboard Screenshots & Interface](#-dashboard-screenshots--interface)
11. [Installation & Setup](#-installation--setup)
12. [How to Run the Application](#-how-to-run-the-application)
13. [REST API Documentation](#-rest-api-documentation)
14. [Automated Testing Suite](#-automated-testing-suite)
15. [Deployment Guide](#-deployment-guide)
16. [Future Enhancements](#-future-enhancements)
17. [License & Disclaimer](#-license--disclaimer)

---

## 📌 Project Overview
Fraudulent financial transactions cost the global economy over $30 billion annually. Traditional rule-based engines struggle against evolving adversarial patterns and produce unmanageable false positive rates that overwhelm risk analysts.

**FraudShield AI** addresses this challenge by combining modern gradient boosting algorithms (`XGBoost`), rigorous class imbalance handling (`SMOTE`), statistical threshold optimization, and cooperative game-theoretic Explainable AI (`SHAP`). The system is packaged within a decoupled **FastAPI** backend microservice and a responsive, dark-navy themed **Streamlit** executive dashboard with persistent **SQLite** audit logging.

---

## 🎯 Problem Statement
1. **Extreme Class Imbalance:** Fraud represents only ~0.17% of total credit card transaction volume (1 fraud per ~580 legitimate transactions).
2. **Deceptive Accuracy Paradox:** A naive model predicting all transactions as legitimate achieves **99.83% accuracy** while detecting exactly **0%** of frauds. High accuracy is completely meaningless without high Precision, Recall, and PR-AUC.
3. **Data Leakage in Academic Work:** Many implementations fit scalers across the whole dataset or apply SMOTE before train-test splitting, artificially inflating test scores that fail in real-world deployment.
4. **Black-Box Opacity:** Regulatory mandates (such as GDPR Article 22 and the US Fair Credit Reporting Act) require explainable decisions for adverse financial actions.

---

## 🚀 Key Features

### 1. Executive Analytics Dashboard
- Comprehensive KPI telemetry: Total Screened, Verified Frauds, Audit Volumes, and Model Performance.
- Interactive Plotly visualizations: Class distribution donut chart, log-scaled monetary amount boxplots, and operational status.

### 2. Real-Time Single Transaction Scoring
- Manual parameter input form across all 30 features (`Time`, `Amount`, `V1` to `V28`).
- One-click presets for sample Legitimate and Confirmed Fraud transactions.
- Instant risk classification into **Low**, **Medium**, and **High Risk** tiers.
- Instant local SHAP explanation showing exact positive and negative feature drivers.

### 3. High-Throughput Batch File Analysis
- Upload CSV transaction files for automated risk scoring.
- Downloadable ready-to-test sample dataset (5,492 rows).
- Interactive filterable results table with risk tier categorization.
- Export scored predictions as an enriched CSV report.

### 4. Persistent Transaction Audit Log
- SQLite database backing with SQLAlchemy ORM.
- Filter records by risk level (`HIGH`, `MEDIUM`, `LOW`, `ALL`).
- Full-text search by custom Transaction ID.
- One-click audit history CSV export.

### 5. Multi-Model Benchmark & Deep Evaluation
- Comparative benchmarking across **Logistic Regression**, **Random Forest**, and **XGBoost**.
- Interactive Precision-Recall (PR) and Receiver Operating Characteristic (ROC) curve overlays.
- Confusion matrix heatmap with False Positive Rate (FPR) and False Negative Rate (FNR) metrics.
- Global feature importance ranking.

### 6. Explainable AI (SHAP)
- Global feature importance hierarchy across all PCA components.
- Local waterfall and diverging bar charts for individual transactions.
- Automated, non-technical natural language compliance narratives for investigators.

### 7. RESTful Backend Microservice
- Fully validated FastAPI endpoints with Pydantic request/response schemas.
- Interactive OpenAPI / Swagger UI documentation at `/docs`.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | Streamlit, Plotly Graph Objects & Express, Custom CSS (Dark Navy Fintech Theme) |
| **Backend API** | FastAPI, Uvicorn, Pydantic v2, Python Dotenv |
| **Machine Learning** | Scikit-learn, XGBoost, Imbalanced-learn (SMOTE), SHAP, Joblib |
| **Data Processing** | Pandas, NumPy |
| **Persistence** | SQLite, SQLAlchemy 2.0 ORM |
| **Testing** | Pytest, FastAPI TestClient |
| **Development** | Git, VS Code, Jupyter Notebook (`nbformat`) |

---

## 🏛️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                      Client Presentation Layer                         │
│   Streamlit Web Application (Dark Navy Fintech Dashboard - Port 8501)   │
│   ├── Executive Telemetry   ├── Real-Time Inference  ├── Batch Audits  │
│   ├── Model Benchmarks      ├── SHAP Interpretability├── Audit Log     │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     │ REST HTTP / In-Memory
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FastAPI REST API Service                        │
│                        (Uvicorn ASGI - Port 8000)                      │
│   ├── GET  /health          ├── POST /predict        ├── POST /batch   │
│   ├── GET  /transactions    ├── GET  /metrics        ├── GET  /info    │
│   └── POST /explain         └── Pydantic v2 Schema Validation         │
└──────────────────┬─────────────────────────────────┬───────────────────┘
                   │                                 │
                   ▼                                 ▼
┌───────────────────────────────────┐ ┌───────────────────────────────────┐
│     Machine Learning Pipeline     │ │        Persistence Layer          │
│  - Pre-fitted RobustScaler        │ │  - SQLite Database Engine         │
│  - XGBoost Classifier (Tuned)     │ │  - SQLAlchemy TransactionRecord  │
│  - Validation Decision Threshold  │ │  - Audit Logs & KPI Aggregates    │
│  - SHAP TreeExplainer Engine      │ └───────────────────────────────────┘
└───────────────────────────────────┘
```

---

## 📊 Dataset Information

This project utilizes the **Credit Card Fraud Detection Dataset** released by the Machine Learning Group (MLG) at Université Libre de Bruxelles (ULB).

- **Total Transactions:** 284,807
- **Fraudulent Transactions:** 492 (0.172% extreme class imbalance)
- **Legitimate Transactions:** 284,315 (99.828%)
- **Input Features (30 continuous variables):**
  - `Time`: Seconds elapsed between this transaction and the first transaction in the dataset.
  - `V1` to `V28`: Principal components obtained via PCA (anonymized for client confidentiality).
  - `Amount`: Transaction monetary value (in Euros).
- **Target Variable:**
  - `Class`: `0` for legitimate transactions, `1` for fraudulent transactions.

---

## 🔬 Machine Learning Workflow & Leakage Prevention

To guarantee genuine production-level performance, the pipeline enforces strict data hygiene:

```
[Raw Dataset: 284,807 Transactions]
                 │
                 ▼
       [Deduplication: -1,081 rows]
                 │
                 ▼
   [Stratified 70/15/15 3-Way Split]
                 │
   ┌─────────────┼─────────────┐
   ▼             ▼             ▼
Train Set     Val Set      Test Set
(70%: 198.6k) (15%: 42.5k) (15%: 42.5k)
   │             │             │
   ▼             │             │
[Fit RobustScaler on Train]    │
   │──Apply Scaler to Val──────┤
   │──Apply Scaler to Test─────┘
   │
   ▼
[Apply SMOTE exclusively to Train]
(minority resampled to 10% ratio)
   │
   ▼
[Train Candidate Models: LogReg, RF, XGBoost]
   │
   ▼
[Benchmark on Validation Set & Tune Decision Threshold]
(Maximize harmonic F1-score / PR-AUC trade-off)
   │
   ▼
[Evaluate Selected Model on Pristine Unseen Test Set]
(Honest reporting of PR-AUC, F1, Recall, Precision)
```

1. **Stratified Partitioning:** Stratification ensures the exact ~0.17% fraud prevalence is mirrored across train, validation, and test splits.
2. **RobustScaler Fitting:** Outliers in financial amounts and elapsed time are scaled using interquartile ranges, fitted **exclusively on the training partition**.
3. **Training-Only SMOTE:** Synthetic minority oversampling is applied strictly after splitting. The validation and test sets remain completely unaltered representations of real-world traffic.
4. **Validation-Guided Threshold Optimization:** The decision boundary is tuned on the validation set to balance false alarms and fraud capture, rather than blindly defaulting to 0.5.

---

## 📈 Empirical Model Evaluation Results

### Validation Set Model Comparison Matrix

| Model | PR-AUC | ROC-AUC | F1-Score | Precision | Recall | Accuracy | Optimal Threshold | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Selected)** | **0.8499** | **0.9782** | **0.8855** | **0.9667** | **0.8169** | **0.9997** | **0.9207** | **2.08s** |
| Random Forest | 0.8565 | 0.9865 | 0.8636 | 0.9344 | 0.8028 | 0.9996 | 0.8019 | 14.02s |
| Logistic Regression | 0.7521 | 0.9754 | 0.8000 | 0.7838 | 0.8169 | 0.9993 | 0.9207 | 0.52s |

### Final Unseen Test Set Performance (XGBoost @ Threshold 0.9207)

| Metric | Score | Practical Significance |
| :--- | :---: | :--- |
| **PR-AUC** | **0.7869** | High area under the Precision-Recall curve under extreme 0.17% imbalance |
| **ROC-AUC** | **0.9693** | Outstanding global discriminative separation power |
| **F1-Score** | **0.8226** | Strong harmonic balance between detecting fraud and minimizing false alarms |
| **Precision** | **96.23%** | When the system flags a transaction as fraud, it is correct **96.2%** of the time |
| **Recall** | **71.83%** | Captures over **71.8%** of all fraudulent transactions autonomously |
| **Accuracy** | **99.95%** | Overall classification accuracy across 42,559 unseen transactions |

### Unseen Test Confusion Matrix (42,559 Transactions)

```
                       Predicted Legitimate    Predicted Fraud
Actual Legitimate             42,486                  2   (Only 2 False Alarms!)
Actual Fraud                      20                 51   (51 Frauds Caught)
```
- **False Positive Rate (FPR):** $2 / 42,488 \approx \mathbf{0.0047\%}$ (minimal analyst workload).
- **False Negative Rate (FNR):** $20 / 71 \approx \mathbf{28.1\%}$.

---

## 📸 Dashboard Screenshots & Interface

The application features a modern dark-navy aesthetic tailored for fintech security operations centers (SOC):

- **Executive Dashboard:** Live telemetry cards, donut class split, transaction amount boxplots.
- **Real-Time Detection:** Interactive sliders and number inputs with instant risk badges (`LOW`, `MEDIUM`, `HIGH`).
- **Batch Processing:** File drag-and-drop, progress bar scoring, and enriched CSV export.
- **Audit History:** Filterable table with search, pagination, and persistent SQLite storage.
- **Model Benchmark:** Dynamic Plotly ROC & PR curves, confusion matrix heatmaps.
- **Explainable AI:** SHAP waterfall charts illustrating exact feature-level risk contributions.

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.11 or higher
- Git

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/AI_Fraud_Detection.git
cd AI_Fraud_Detection
```

### Step 2: Create & Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset
```bash
python dataset/download_dataset.py
```
*Note: This automatically downloads the authentic 143 MB ULB dataset mirror and generates a stratified sample for quick testing.*

---

## 🏃 How to Run the Application

### Option A: Run Streamlit Analytics Dashboard
```bash
streamlit run app/dashboard.py
```
Open your browser at `http://localhost:8501`.

### Option B: Run FastAPI Backend Service
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive OpenAPI documentation will be accessible at `http://127.0.0.1:8000/docs`.

### Option C: Retrain Machine Learning Models
```bash
python src/train_model.py
```

---

## 🔌 REST API Documentation

| Method | Endpoint | Description | Request Body | Response |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service health & model readiness | None | `HealthResponse` |
| `POST` | `/predict` | Score single transaction | `TransactionInput` (30 features) | `PredictionResponse` |
| `POST` | `/predict/batch` | Bulk transaction scoring | `BatchPredictionRequest` | `BatchPredictionResponse` |
| `GET` | `/transactions` | Query SQLite audit history | Query params (`risk_level`, `search`) | `TransactionHistoryResponse` |
| `GET` | `/metrics` | Retrieve model evaluation metrics | None | Benchmark JSON |
| `GET` | `/model/info` | Inspect active model architecture | None | `ModelInfoResponse` |
| `POST` | `/explain` | Compute SHAP feature attributions | `TransactionInput` | SHAP breakdown JSON |

### Example Request (`POST /predict`)
```json
{
  "Time": 406.0,
  "V1": -2.312,
  "V2": 1.951,
  "V3": -1.609,
  "V4": 3.997,
  "V14": -4.289,
  "Amount": 142.50,
  "transaction_id": "TXN-2026-001"
}
```

### Example Response
```json
{
  "transaction_id": "TXN-2026-001",
  "amount": 142.50,
  "fraud_probability": 0.9412,
  "prediction": 1,
  "prediction_label": "Fraudulent",
  "risk_level": "HIGH",
  "threshold_used": 0.9207,
  "model_used": "XGBoost",
  "disclaimer": "Risk estimate generated by AI model. Not a definitive verified determination."
}
```

---

## 🧪 Automated Testing Suite

The project includes 23 unit and integration tests covering preprocessing, threshold tuning, inference edge cases, SHAP attribution, API endpoints, and database operations.

Run the test suite with:
```bash
pytest -v tests/
```

```text
tests/test_api.py::test_health_check_endpoint PASSED
tests/test_api.py::test_model_info_endpoint PASSED
tests/test_api.py::test_metrics_endpoint PASSED
tests/test_api.py::test_predict_single_success PASSED
tests/test_api.py::test_predict_single_negative_amount PASSED
tests/test_api.py::test_predict_single_missing_field PASSED
tests/test_api.py::test_predict_batch_success PASSED
tests/test_api.py::test_predict_batch_empty PASSED
tests/test_api.py::test_get_transactions_history PASSED
tests/test_api.py::test_get_transactions_filter_risk PASSED
tests/test_api.py::test_explain_endpoint PASSED
tests/test_model.py::test_split_data_proportions PASSED
tests/test_model.py::test_scaler_fitting_and_transformation PASSED
tests/test_model.py::test_missing_columns_validation PASSED
tests/test_model.py::test_missing_values_imputation PASSED
tests/test_model.py::test_duplicate_records_removal PASSED
tests/test_model.py::test_calculate_metrics PASSED
tests/test_model.py::test_optimal_threshold_tuning PASSED
tests/test_model.py::test_predictor_single_prediction PASSED
tests/test_model.py::test_predictor_fraud_vs_normal PASSED
tests/test_model.py::test_predictor_empty_batch PASSED
tests/test_model.py::test_predictor_batch_prediction PASSED
tests/test_model.py::test_shap_explanation PASSED

======================= 23 passed in 5.09s =======================
```

---

## 🚀 Deployment Guide

### Deploying Dashboard to Streamlit Community Cloud
1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Click **New App**, select your repository, branch `main`, and main file path `app/dashboard.py`.
4. Add environment variables in **Advanced Settings** if needed.
5. Click **Deploy**.

### Deploying Backend to Render / Railway
1. Create a new **Web Service** on [Render](https://render.com).
2. Connect your GitHub repository.
3. Set the Environment to **Python 3**.
4. Set Build Command:
   ```bash
   pip install -r requirements.txt
   ```
5. Set Start Command:
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port $PORT
   ```

---

## 🔮 Future Enhancements
- **Graph Neural Networks (GNN):** Incorporate cardholder-merchant transaction graph topologies to identify organized fraud rings.
- **Continuous Learning & Drift Monitoring:** Real-time drift detection (Evidently AI / KS-test) to trigger model retraining upon concept drift.
- **Kafka / Flink Streaming Pipeline:** Process sub-second streaming transactions from event buses at 10,000+ transactions per second.
- **Federated Learning:** Train collaborative fraud models across financial institutions without sharing proprietary customer PII.

---

## 📄 License & Disclaimer

### Open-Source Notice
This project is open-source under standard permissive terms for educational, research, and portfolio demonstration purposes.

### Regulatory & Financial Disclaimer
Predictions generated by this system represent probabilistic risk scores and statistical estimations. They do not constitute definitive legal or financial determinations of fraud. Always employ a human-in-the-loop compliance review process prior to taking adverse customer actions.
