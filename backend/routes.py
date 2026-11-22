"""
FastAPI route definitions for Financial Fraud Detection API.
"""
import os
import json
from datetime import datetime
from typing import Optional, List
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from database.db import (
    get_db,
    save_transaction,
    save_transactions_batch,
    get_transactions,
    get_transaction_stats,
    TransactionRecord
)
from src.predict import FraudPredictor
from src.explainability import FraudExplainer
from backend.schemas import (
    TransactionInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchSummary,
    TransactionHistoryResponse,
    TransactionRecordOut,
    ModelInfoResponse,
    HealthResponse
)

router = APIRouter()

# Global predictor and explainer instances
predictor = FraudPredictor()
explainer = None


def get_explainer() -> FraudExplainer:
    """Lazy initialize SHAP explainer when needed."""
    global explainer
    if explainer is None and predictor.is_ready():
        explainer = FraudExplainer()
    return explainer


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(db: Session = Depends(get_db)):
    """Check API service health, ML model readiness, and database connectivity."""
    db_ok = False
    try:
        db.execute(TransactionRecord.__table__.select().limit(1))
        db_ok = True
    except Exception:
        db_ok = False

    return HealthResponse(
        status="healthy" if predictor.is_ready() and db_ok else "degraded",
        model_loaded=predictor.is_ready(),
        database_connected=db_ok,
        timestamp=datetime.utcnow().isoformat()
    )


@router.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single_transaction(
    payload: TransactionInput,
    db: Session = Depends(get_db)
):
    """
    Score a single financial transaction for fraud risk.
    Persists prediction and risk classification to database.
    """
    if not predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model is not loaded. Please train models first."
        )

    try:
        feature_dict = payload.model_dump(exclude={"transaction_id"})
        result = predictor.predict_single(
            features=feature_dict,
            transaction_id=payload.transaction_id
        )

        # Persist to database
        save_transaction(
            db=db,
            transaction_id=result["transaction_id"],
            amount=result["amount"],
            fraud_probability=result["fraud_probability"],
            prediction=result["prediction"],
            risk_level=result["risk_level"],
            threshold_used=result["threshold_used"],
            source="API",
            model_version=result["model_used"]
        )

        result["timestamp"] = datetime.utcnow().isoformat()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference error: {str(e)}"
        )


@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_transactions(
    payload: BatchPredictionRequest,
    db: Session = Depends(get_db)
):
    """
    Score multiple transactions in batch mode.
    Computes aggregated risk summaries and bulk-persists records to SQLite.
    """
    if not predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model is not loaded. Please train models first."
        )

    try:
        raw_list = [t.model_dump() for t in payload.transactions]
        df = pd.DataFrame(raw_list)

        result_df, summary = predictor.predict_batch(df)

        # Bulk save to DB
        records_to_save = []
        prediction_responses = []

        for _, row in result_df.iterrows():
            rec_data = {
                "transaction_id": row["transaction_id"],
                "amount": float(row["Amount"]),
                "fraud_probability": float(row["fraud_probability"]),
                "prediction": int(row["prediction"]),
                "risk_level": str(row["risk_level"]),
                "threshold_used": float(summary["threshold_used"]),
                "source": "BATCH_API",
                "model_version": summary["model_name"]
            }
            records_to_save.append(rec_data)

            prediction_responses.append(PredictionResponse(
                transaction_id=row["transaction_id"],
                amount=float(row["Amount"]),
                fraud_probability=float(row["fraud_probability"]),
                prediction=int(row["prediction"]),
                prediction_label=row["prediction_label"],
                risk_level=row["risk_level"],
                threshold_used=summary["threshold_used"],
                model_used=summary["model_name"],
                disclaimer="Risk estimate generated by AI model. Not a definitive verified determination.",
                timestamp=datetime.utcnow().isoformat()
            ))

        save_transactions_batch(db, records_to_save)

        return BatchPredictionResponse(
            summary=BatchSummary(**summary),
            predictions=prediction_responses
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch inference failed: {str(e)}"
        )


@router.get("/transactions", response_model=TransactionHistoryResponse, tags=["History"])
def get_transaction_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    risk_level: Optional[str] = Query(None, description="Filter by LOW, MEDIUM, or HIGH"),
    search: Optional[str] = Query(None, description="Search by transaction ID"),
    db: Session = Depends(get_db)
):
    """Retrieve transaction prediction history with optional risk filtering and ID search."""
    records = get_transactions(db, skip=skip, limit=limit, risk_filter=risk_level, search=search)
    stats = get_transaction_stats(db)

    out_records = [
        TransactionRecordOut(
            id=r.id,
            transaction_id=r.transaction_id,
            timestamp=r.timestamp.isoformat() if r.timestamp else "",
            amount=r.amount,
            fraud_probability=r.fraud_probability,
            prediction=r.prediction,
            risk_level=r.risk_level,
            threshold_used=r.threshold_used,
            source=r.source,
            model_version=r.model_version
        )
        for r in records
    ]

    return TransactionHistoryResponse(
        total=len(records),
        stats=stats,
        records=out_records
    )


@router.get("/metrics", tags=["Model Analytics"])
def get_model_metrics():
    """Retrieve comprehensive model evaluation metrics, comparison benchmarks, and curves."""
    metrics_path = predictor.metrics_path
    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metrics report not found. Train models first."
        )

    try:
        with open(metrics_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read metrics: {str(e)}"
        )


@router.get("/model/info", response_model=ModelInfoResponse, tags=["Model Analytics"])
def get_model_info():
    """Retrieve active model architecture metadata and feature configuration."""
    metrics_path = predictor.metrics_path
    if not os.path.exists(metrics_path):
        return ModelInfoResponse(
            model_name="Not Loaded",
            optimal_threshold=0.5,
            feature_count=30,
            features=['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        )

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    return ModelInfoResponse(
        model_name=metrics.get("best_model_name", "XGBoost"),
        optimal_threshold=metrics.get("optimal_threshold", 0.5),
        feature_count=len(metrics.get("feature_columns", [])),
        features=metrics.get("feature_columns", []),
        trained_timestamp=metrics.get("trained_timestamp"),
        training_summary=metrics.get("data_summary")
    )


@router.post("/explain", tags=["Explainable AI"])
def explain_transaction_risk(payload: TransactionInput):
    """
    Compute SHAP-based feature contributions and plain-language narrative
    explaining risk factors for a specific transaction.
    """
    exp = get_explainer()
    if exp is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SHAP explainer engine is not ready. Train model first."
        )

    try:
        features = payload.model_dump(exclude={"transaction_id"})
        explanation = exp.explain_transaction(features)
        return explanation
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Explanation calculation failed: {str(e)}"
        )
