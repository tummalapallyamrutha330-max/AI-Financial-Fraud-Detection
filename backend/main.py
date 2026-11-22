"""
Main FastAPI application entrypoint for Financial Fraud Detection System.
"""
import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database.db import init_db
from backend.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite database schema on startup
    init_db()
    print("Database schema verified and initialized.")
    yield


app = FastAPI(
    title="AI-Powered Financial Fraud Detection API",
    description=(
        "Production-grade REST API for real-time and batch credit card transaction fraud detection. "
        "Leverages XGBoost / Random Forest machine learning models trained on imbalanced financial datasets "
        "with validation-tuned decision thresholds and Explainable AI (SHAP)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for Streamlit frontend and local integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(router, prefix="")


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
