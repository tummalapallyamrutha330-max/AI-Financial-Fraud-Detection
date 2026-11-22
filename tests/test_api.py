"""
Automated tests for FastAPI endpoints, request validation, error handling,
and SQLite database integration.
"""
import os
import sys
import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.main import app
from database.db import init_db, SessionLocal, TransactionRecord, get_transaction_stats

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_database():
    """Ensure database schema is created before tests run."""
    init_db()


@pytest.fixture
def valid_transaction_payload():
    """Generate a valid transaction payload."""
    payload = {f"V{i}": 0.0 for i in range(1, 29)}
    payload["Time"] = 120.0
    payload["Amount"] = 85.50
    payload["transaction_id"] = "TXN-TEST-001"
    return payload


# -------------------------------------------------------------
# 1. Health & Metadata Endpoints
# -------------------------------------------------------------
def test_health_check_endpoint():
    """Test GET /health returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["model_loaded"] is True
    assert data["database_connected"] is True


def test_model_info_endpoint():
    """Test GET /model/info returns active model configuration."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] in ["XGBoost", "Random Forest", "Logistic Regression"]
    assert data["feature_count"] == 30
    assert 0.0 < data["optimal_threshold"] < 1.0


def test_metrics_endpoint():
    """Test GET /metrics returns benchmark curves and test performance."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "test_metrics" in data
    assert "validation_comparison" in data
    assert "optimal_threshold" in data


# -------------------------------------------------------------
# 2. Prediction Endpoints & Input Validation
# -------------------------------------------------------------
def test_predict_single_success(valid_transaction_payload):
    """Test POST /predict successfully scores a valid transaction."""
    response = client.post("/predict", json=valid_transaction_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "TXN-TEST-001"
    assert data["amount"] == 85.50
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert "disclaimer" in data


def test_predict_single_negative_amount(valid_transaction_payload):
    """Test POST /predict rejects negative transaction amount with 422."""
    bad_payload = valid_transaction_payload.copy()
    bad_payload["Amount"] = -50.0
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422  # Unprocessable Entity (Pydantic validation)


def test_predict_single_missing_field():
    """Test POST /predict rejects request with missing required PCA features."""
    incomplete_payload = {"Time": 50.0, "Amount": 100.0}
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


def test_predict_batch_success(valid_transaction_payload):
    """Test POST /predict/batch scores multiple items and returns summary."""
    item1 = valid_transaction_payload.copy()
    item1["transaction_id"] = "TXN-BATCH-001"
    item2 = valid_transaction_payload.copy()
    item2["transaction_id"] = "TXN-BATCH-002"
    item2["Amount"] = 450.0

    batch_payload = {"transactions": [item1, item2]}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert data["summary"]["total_transactions"] == 2
    assert len(data["predictions"]) == 2


def test_predict_batch_empty():
    """Test POST /predict/batch rejects empty transactions list."""
    response = client.post("/predict/batch", json={"transactions": []})
    assert response.status_code == 422


# -------------------------------------------------------------
# 3. Transaction History & Database Integration
# -------------------------------------------------------------
def test_get_transactions_history():
    """Test GET /transactions returns database records and aggregate stats."""
    response = client.get("/transactions?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "stats" in data
    assert "records" in data


def test_get_transactions_filter_risk():
    """Test GET /transactions filtering by risk level."""
    response = client.get("/transactions?risk_level=HIGH")
    assert response.status_code == 200
    data = response.json()
    for rec in data["records"]:
        assert rec["risk_level"] == "HIGH"


# -------------------------------------------------------------
# 4. Explainable AI (SHAP) Endpoint
# -------------------------------------------------------------
def test_explain_endpoint(valid_transaction_payload):
    """Test POST /explain computes feature attribution and plain-language summary."""
    response = client.post("/explain", json=valid_transaction_payload)
    assert response.status_code == 200
    data = response.json()
    assert "top_positive_factors" in data
    assert "top_negative_factors" in data
    assert "narrative_explanation" in data
    assert "all_contributions" in data
