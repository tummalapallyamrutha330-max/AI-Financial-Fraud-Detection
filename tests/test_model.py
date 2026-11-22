"""
Unit tests for data preprocessing, ML models, inference, and explainability.
Tests:
- Normal transaction prediction
- Fraud transaction prediction
- Empty DataFrame / CSV handling
- Missing columns validation
- Missing values (NaN) handling
- Duplicate records handling
- Invalid numerical inputs
- Optimal threshold calculation
- Model metrics evaluation
- SHAP explainability attribution
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import (
    load_raw_dataset,
    split_data,
    fit_scaler,
    transform_features,
    preprocess_input_features,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)
from src.evaluation import calculate_metrics, find_optimal_threshold
from src.predict import FraudPredictor
from src.explainability import FraudExplainer


@pytest.fixture
def sample_features():
    """Valid standard transaction features dictionary."""
    features = {f"V{i}": 0.05 * (i % 3 - 1) for i in range(1, 29)}
    features["Time"] = 1000.0
    features["Amount"] = 45.50
    return features


@pytest.fixture
def fraud_like_features():
    """Features known to be heavily shifted towards fraud (extreme V14, V10, V4)."""
    features = {f"V{i}": 0.0 for i in range(1, 29)}
    features["V14"] = -7.5
    features["V10"] = -5.0
    features["V4"] = 4.5
    features["V12"] = -4.0
    features["V17"] = -5.5
    features["Time"] = 500.0
    features["Amount"] = 250.0
    return features


# -------------------------------------------------------------
# 1. Preprocessing & Data Validation Tests
# -------------------------------------------------------------
def test_split_data_proportions():
    """Test 3-way stratified split preserves required ratios."""
    # Create synthetic balanced test frame
    n_samples = 1000
    data = {col: np.random.randn(n_samples) for col in FEATURE_COLUMNS}
    # 5% minority class
    data[TARGET_COLUMN] = [1 if i < 50 else 0 for i in range(n_samples)]
    df = pd.DataFrame(data)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, test_size=0.15, val_size=0.15)

    assert len(X_train) == 700
    assert len(X_val) == 150
    assert len(X_test) == 150
    # Stratification check: target must be present in all splits
    assert y_train.sum() > 0
    assert y_val.sum() > 0
    assert y_test.sum() > 0


def test_scaler_fitting_and_transformation():
    """Verify RobustScaler fits on training data and transforms Time and Amount."""
    X_train = pd.DataFrame({
        "Time": [10.0, 50.0, 100.0, 500.0],
        "Amount": [5.0, 25.0, 50.0, 1000.0]
    })
    scaler = fit_scaler(X_train)
    scaled_df = transform_features(X_train, scaler)

    assert "Time" in scaled_df.columns
    assert "Amount" in scaled_df.columns
    # Scaled values must differ from raw
    assert not np.array_equal(scaled_df["Amount"].values, X_train["Amount"].values)


def test_missing_columns_validation():
    """Verify that omitting required feature columns raises ValueError."""
    incomplete_data = pd.DataFrame({"Time": [10.0], "Amount": [50.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        preprocess_input_features(incomplete_data)


def test_missing_values_imputation():
    """Verify that NaN values in inputs are filled without failing."""
    data = {col: [np.nan] for col in FEATURE_COLUMNS}
    df = pd.DataFrame(data)
    processed = preprocess_input_features(df)

    assert processed.isna().sum().sum() == 0


def test_duplicate_records_removal(tmp_path):
    """Verify load_raw_dataset detects and removes duplicate rows."""
    csv_file = tmp_path / "dup_test.csv"
    data = {col: [1.0, 2.0, 1.0] for col in FEATURE_COLUMNS}
    data[TARGET_COLUMN] = [0, 1, 0]  # Rows 0 and 2 are duplicates
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    loaded_df = load_raw_dataset(str(csv_file))
    assert len(loaded_df) == 2


# -------------------------------------------------------------
# 2. Evaluation Metrics & Threshold Tests
# -------------------------------------------------------------
def test_calculate_metrics():
    """Verify accuracy, precision, recall, F1, PR-AUC, and confusion matrix calculation."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.6, 0.8, 0.9])

    metrics = calculate_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 0.8
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == round(2 / 3, 4)
    assert metrics["confusion_matrix"]["true_positives"] == 2
    assert metrics["confusion_matrix"]["false_positives"] == 1
    assert metrics["confusion_matrix"]["true_negatives"] == 2
    assert metrics["confusion_matrix"]["false_negatives"] == 0
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_optimal_threshold_tuning():
    """Verify optimal threshold tuning finds threshold maximizing F1."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.15, 0.25, 0.35, 0.40, 0.65, 0.85, 0.95])

    thresh, best_f1 = find_optimal_threshold(y_true, y_prob)
    assert 0.0 < thresh < 1.0
    assert best_f1 > 0.8


# -------------------------------------------------------------
# 3. Predictor Engine Tests
# -------------------------------------------------------------
def test_predictor_single_prediction(sample_features):
    """Verify single transaction prediction returns expected schema and keys."""
    predictor = FraudPredictor()
    if not predictor.is_ready():
        pytest.skip("Model artifacts not yet trained. Skipping live inference test.")

    res = predictor.predict_single(sample_features)

    assert "transaction_id" in res
    assert "fraud_probability" in res
    assert "prediction" in res
    assert "risk_level" in res
    assert res["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert 0.0 <= res["fraud_probability"] <= 1.0


def test_predictor_fraud_vs_normal(sample_features, fraud_like_features):
    """Verify high-risk features produce a higher fraud probability than normal features."""
    predictor = FraudPredictor()
    if not predictor.is_ready():
        pytest.skip("Model artifacts not yet trained.")

    normal_res = predictor.predict_single(sample_features)
    fraud_res = predictor.predict_single(fraud_like_features)

    assert fraud_res["fraud_probability"] > normal_res["fraud_probability"]


def test_predictor_empty_batch():
    """Verify batch inference rejects empty DataFrame with clear error."""
    predictor = FraudPredictor()
    if not predictor.is_ready():
        pytest.skip("Model artifacts not yet trained.")

    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="empty"):
        predictor.predict_batch(empty_df)


def test_predictor_batch_prediction(sample_features):
    """Verify batch prediction appends prediction columns and returns summary."""
    predictor = FraudPredictor()
    if not predictor.is_ready():
        pytest.skip("Model artifacts not yet trained.")

    # Create 5 rows
    rows = [sample_features.copy() for _ in range(5)]
    df_batch = pd.DataFrame(rows)

    res_df, summary = predictor.predict_batch(df_batch)

    assert len(res_df) == 5
    assert "fraud_probability" in res_df.columns
    assert "prediction" in res_df.columns
    assert "risk_level" in res_df.columns
    assert summary["total_transactions"] == 5


# -------------------------------------------------------------
# 4. Explainable AI (SHAP) Tests
# -------------------------------------------------------------
def test_shap_explanation(sample_features):
    """Verify SHAP explainer produces positive and negative factors and plain narrative."""
    explainer = FraudExplainer()
    if explainer.explainer is None:
        pytest.skip("Explainer not ready (requires trained model).")

    res = explainer.explain_transaction(sample_features)

    assert "top_positive_factors" in res
    assert "top_negative_factors" in res
    assert "narrative_explanation" in res
    assert len(res["narrative_explanation"]) > 20
