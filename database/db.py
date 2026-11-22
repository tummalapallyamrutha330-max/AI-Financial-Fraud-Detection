"""
Database models and session management for Financial Fraud Detection System.
Uses SQLite and SQLAlchemy for persistence.
"""
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Index
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Database path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "fraud_detection.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class TransactionRecord(Base):
    """
    Persisted transaction record with ML prediction output and risk categorization.
    """
    __tablename__ = "transaction_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    transaction_id = Column(String(64), unique=True, index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    amount = Column(Float, nullable=False)
    
    # ML Prediction Outputs
    fraud_probability = Column(Float, nullable=False)
    prediction = Column(Integer, nullable=False)  # 0: Legitimate, 1: Fraud
    risk_level = Column(String(16), nullable=False)  # LOW, MEDIUM, HIGH
    threshold_used = Column(Float, nullable=False, default=0.5)
    
    # Metadata
    source = Column(String(32), default="MANUAL")  # MANUAL, BATCH, API
    model_version = Column(String(32), default="XGBoost-v1.0")

    __table_args__ = (
        Index("idx_txn_timestamp", "timestamp"),
        Index("idx_txn_risk", "risk_level"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "transaction_id": self.transaction_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "amount": self.amount,
            "fraud_probability": round(self.fraud_probability, 4),
            "prediction": self.prediction,
            "risk_level": self.risk_level,
            "threshold_used": self.threshold_used,
            "source": self.source,
            "model_version": self.model_version
        }


def init_db():
    """Create all database tables if they do not exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for obtaining database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_transaction(
    db: Session,
    transaction_id: str,
    amount: float,
    fraud_probability: float,
    prediction: int,
    risk_level: str,
    threshold_used: float = 0.5,
    source: str = "MANUAL",
    model_version: str = "v1.0"
) -> TransactionRecord:
    """Save or update an individual transaction prediction record in the database."""
    record = db.query(TransactionRecord).filter(TransactionRecord.transaction_id == transaction_id).first()
    if record:
        record.amount = amount
        record.fraud_probability = fraud_probability
        record.prediction = prediction
        record.risk_level = risk_level
        record.threshold_used = threshold_used
        record.source = source
        record.model_version = model_version
        record.timestamp = datetime.utcnow()
    else:
        record = TransactionRecord(
            transaction_id=transaction_id,
            amount=amount,
            fraud_probability=fraud_probability,
            prediction=prediction,
            risk_level=risk_level,
            threshold_used=threshold_used,
            source=source,
            model_version=model_version,
            timestamp=datetime.utcnow()
        )
        db.add(record)
    db.commit()
    db.refresh(record)
    return record


def save_transactions_batch(
    db: Session,
    records_data: List[Dict[str, Any]]
) -> int:
    """Insert or update batch transaction prediction records."""
    count = 0
    for r in records_data:
        tid = r["transaction_id"]
        rec = db.query(TransactionRecord).filter(TransactionRecord.transaction_id == tid).first()
        if rec:
            rec.amount = r["amount"]
            rec.fraud_probability = r["fraud_probability"]
            rec.prediction = r["prediction"]
            rec.risk_level = r["risk_level"]
            rec.threshold_used = r.get("threshold_used", 0.5)
            rec.source = r.get("source", "BATCH")
            rec.model_version = r.get("model_version", "v1.0")
            rec.timestamp = r.get("timestamp", datetime.utcnow())
        else:
            rec = TransactionRecord(
                transaction_id=tid,
                amount=r["amount"],
                fraud_probability=r["fraud_probability"],
                prediction=r["prediction"],
                risk_level=r["risk_level"],
                threshold_used=r.get("threshold_used", 0.5),
                source=r.get("source", "BATCH"),
                model_version=r.get("model_version", "v1.0"),
                timestamp=r.get("timestamp", datetime.utcnow())
            )
            db.add(rec)
        count += 1
    db.commit()
    return count


def get_transactions(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    risk_filter: Optional[str] = None,
    search: Optional[str] = None
) -> List[TransactionRecord]:
    """Retrieve transactions with optional filtering and search."""
    query = db.query(TransactionRecord)
    if risk_filter and risk_filter.upper() != "ALL":
        query = query.filter(TransactionRecord.risk_level == risk_filter.upper())
    if search:
        query = query.filter(TransactionRecord.transaction_id.ilike(f"%{search}%"))
    return query.order_by(TransactionRecord.timestamp.desc()).offset(skip).limit(limit).all()


def get_transaction_stats(db: Session) -> Dict[str, Any]:
    """Retrieve aggregated stats for dashboard KPI cards."""
    total = db.query(TransactionRecord).count()
    if total == 0:
        return {
            "total_transactions": 0,
            "fraud_count": 0,
            "legit_count": 0,
            "fraud_rate": 0.0,
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "low_risk_count": 0,
            "total_amount_screened": 0.0,
            "fraud_amount_flagged": 0.0
        }
    fraud_count = db.query(TransactionRecord).filter(TransactionRecord.prediction == 1).count()
    legit_count = total - fraud_count
    high_risk = db.query(TransactionRecord).filter(TransactionRecord.risk_level == "HIGH").count()
    med_risk = db.query(TransactionRecord).filter(TransactionRecord.risk_level == "MEDIUM").count()
    low_risk = db.query(TransactionRecord).filter(TransactionRecord.risk_level == "LOW").count()

    all_records = db.query(TransactionRecord.amount, TransactionRecord.prediction).all()
    total_screened = sum(r[0] for r in all_records)
    fraud_flagged = sum(r[0] for r in all_records if r[1] == 1)

    return {
        "total_transactions": total,
        "fraud_count": fraud_count,
        "legit_count": legit_count,
        "fraud_rate": round((fraud_count / total) * 100, 2),
        "high_risk_count": high_risk,
        "medium_risk_count": med_risk,
        "low_risk_count": low_risk,
        "total_amount_screened": round(total_screened, 2),
        "fraud_amount_flagged": round(fraud_flagged, 2)
    }
