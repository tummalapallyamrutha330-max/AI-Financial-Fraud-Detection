"""
Pydantic schemas for request validation and response formatting.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class TransactionInput(BaseModel):
    """Features for an individual financial transaction."""
    Time: float = Field(..., description="Seconds elapsed since the reference start transaction", ge=0.0)
    V1: float = Field(..., description="PCA component V1")
    V2: float = Field(..., description="PCA component V2")
    V3: float = Field(..., description="PCA component V3")
    V4: float = Field(..., description="PCA component V4")
    V5: float = Field(..., description="PCA component V5")
    V6: float = Field(..., description="PCA component V6")
    V7: float = Field(..., description="PCA component V7")
    V8: float = Field(..., description="PCA component V8")
    V9: float = Field(..., description="PCA component V9")
    V10: float = Field(..., description="PCA component V10")
    V11: float = Field(..., description="PCA component V11")
    V12: float = Field(..., description="PCA component V12")
    V13: float = Field(..., description="PCA component V13")
    V14: float = Field(..., description="PCA component V14")
    V15: float = Field(..., description="PCA component V15")
    V16: float = Field(..., description="PCA component V16")
    V17: float = Field(..., description="PCA component V17")
    V18: float = Field(..., description="PCA component V18")
    V19: float = Field(..., description="PCA component V19")
    V20: float = Field(..., description="PCA component V20")
    V21: float = Field(..., description="PCA component V21")
    V22: float = Field(..., description="PCA component V22")
    V23: float = Field(..., description="PCA component V23")
    V24: float = Field(..., description="PCA component V24")
    V25: float = Field(..., description="PCA component V25")
    V26: float = Field(..., description="PCA component V26")
    V27: float = Field(..., description="PCA component V27")
    V28: float = Field(..., description="PCA component V28")
    Amount: float = Field(..., description="Transaction monetary amount", ge=0.0)
    transaction_id: Optional[str] = Field(default=None, description="Optional custom transaction identifier")

    @field_validator("Amount")
    @classmethod
    def validate_amount_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Transaction amount cannot be negative")
        return v


class PredictionResponse(BaseModel):
    """Output risk prediction for a single transaction."""
    transaction_id: str
    amount: float
    fraud_probability: float
    prediction: int
    prediction_label: str
    risk_level: str
    threshold_used: float
    model_used: str
    disclaimer: str
    timestamp: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    """Batch prediction payload with list of transactions."""
    transactions: List[TransactionInput] = Field(..., min_length=1, description="List of transactions to score")


class BatchSummary(BaseModel):
    total_transactions: int
    flagged_fraud_count: int
    legitimate_count: int
    fraud_rate_pct: float
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    total_amount_analyzed: float
    flagged_fraud_amount: float
    threshold_used: float
    model_name: str


class BatchPredictionResponse(BaseModel):
    summary: BatchSummary
    predictions: List[PredictionResponse]


class TransactionRecordOut(BaseModel):
    id: int
    transaction_id: str
    timestamp: str
    amount: float
    fraud_probability: float
    prediction: int
    risk_level: str
    threshold_used: float
    source: str
    model_version: str


class TransactionHistoryResponse(BaseModel):
    total: int
    stats: Dict[str, Any]
    records: List[TransactionRecordOut]


class ModelInfoResponse(BaseModel):
    model_name: str
    optimal_threshold: float
    feature_count: int
    features: List[str]
    trained_timestamp: Optional[str] = None
    training_summary: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool
    timestamp: str
