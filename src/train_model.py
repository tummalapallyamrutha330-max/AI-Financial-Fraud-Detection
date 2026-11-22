"""
Model training and benchmarking pipeline for Credit Card Fraud Detection.
Trains and compares:
1. Logistic Regression
2. Random Forest
3. XGBoost

Ensures zero data leakage:
- Stratified 70/15/15 split.
- Scaler fitted exclusively on train set.
- SMOTE applied exclusively on train set.
- Threshold tuned exclusively on validation set.
- Final performance reported honestly on held-out test set.
"""
import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import sys
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import (
    load_and_preprocess_data,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)
from src.evaluation import (
    calculate_metrics,
    find_optimal_threshold,
    generate_roc_curve_data,
    generate_pr_curve_data
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "creditcard.csv")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_fraud_model.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")


def train_and_evaluate_models(
    dataset_path: str = DATASET_PATH,
    sample_size: int = None
) -> Dict[str, Any]:
    """
    Executes end-to-end model training, validation benchmarking,
    threshold tuning, best model selection, and test evaluation.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=" * 70)
    print("AI FINANCIAL FRAUD DETECTION - ML TRAINING PIPELINE")
    print("=" * 70)

    # 1. Preprocessing & Splitting
    print("\n[Step 1/5] Loading dataset & preprocessing...")
    data = load_and_preprocess_data(
        csv_path=dataset_path,
        scaler_save_path=SCALER_PATH,
        apply_smote_train=True,
        sample_size=sample_size
    )

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print(f"Data Splits:")
    print(f"  - Training (with SMOTE):   {len(X_train):,} samples (Fraud: {int(y_train.sum()):,})")
    print(f"  - Validation (original):   {len(X_val):,} samples (Fraud: {int(y_val.sum()):,})")
    print(f"  - Test (original unseen):  {len(X_test):,} samples (Fraud: {int(y_test.sum()):,})")

    # 2. Define Model Configurations
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=0.1,
            random_state=42,
            solver="lbfgs"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            n_jobs=-1,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,  # SMOTE already balanced train set
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        )
    }

    # 3. Model Training & Validation Evaluation
    print("\n[Step 2/5] Training candidate models...")
    validation_results = {}
    trained_models = {}

    for name, model in models.items():
        t0 = time.time()
        print(f"  -> Training {name}...")
        model.fit(X_train, y_train)
        train_time = round(time.time() - t0, 2)
        trained_models[name] = model

        # Predict probabilities on Validation split
        val_probs = model.predict_proba(X_val)[:, 1]

        # Tune threshold on Validation split
        opt_thresh, opt_f1 = find_optimal_threshold(y_val.values, val_probs)
        val_preds_opt = (val_probs >= opt_thresh).astype(int)

        # Calculate validation metrics
        val_metrics = calculate_metrics(y_val.values, val_preds_opt, val_probs)
        val_metrics["optimal_threshold"] = opt_thresh
        val_metrics["training_time_seconds"] = train_time
        val_metrics["roc_curve"] = generate_roc_curve_data(y_val.values, val_probs)
        val_metrics["pr_curve"] = generate_pr_curve_data(y_val.values, val_probs)

        validation_results[name] = val_metrics
        print(f"     Val PR-AUC: {val_metrics['pr_auc']:.4f} | "
              f"ROC-AUC: {val_metrics['roc_auc']:.4f} | "
              f"F1: {val_metrics['f1_score']:.4f} | "
              f"Recall: {val_metrics['recall']:.4f} | "
              f"Prec: {val_metrics['precision']:.4f} | "
              f"Opt Thresh: {opt_thresh} ({train_time}s)")

    # 4. Model Selection (Prioritize PR-AUC and F1-score)
    print("\n[Step 3/5] Selecting best model based on Validation PR-AUC & F1...")
    # Composite score prioritizing PR-AUC and F1
    best_name = max(
        validation_results.keys(),
        key=lambda k: (validation_results[k]["pr_auc"] * 0.6 + validation_results[k]["f1_score"] * 0.4)
    )
    best_model = trained_models[best_name]
    best_val_metrics = validation_results[best_name]
    best_threshold = best_val_metrics["optimal_threshold"]
    print(f"  ==> Selected Best Model: {best_name} (Threshold: {best_threshold})")

    # 5. Final Evaluation on Held-out Test Set
    print("\n[Step 4/5] Evaluating selected best model on UNSEEN TEST SET...")
    test_probs = best_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)
    test_metrics = calculate_metrics(y_test.values, test_preds, test_probs)
    test_metrics["threshold_applied"] = best_threshold
    test_metrics["roc_curve"] = generate_roc_curve_data(y_test.values, test_probs)
    test_metrics["pr_curve"] = generate_pr_curve_data(y_test.values, test_probs)

    print(f"  Test Performance on Unseen Data:")
    print(f"    - PR-AUC:    {test_metrics['pr_auc']:.4f}")
    print(f"    - ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"    - F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"    - Precision: {test_metrics['precision']:.4f}")
    print(f"    - Recall:    {test_metrics['recall']:.4f}")
    print(f"    - Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"    - Confusion: {test_metrics['confusion_matrix']['matrix']}")

    # 6. Save Model Artifacts
    print("\n[Step 5/5] Saving model and metadata artifacts...")
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"  Saved best model to: {BEST_MODEL_PATH}")

    # Feature Importance for tree-based models
    feature_importance = {}
    if hasattr(best_model, "feature_importances_"):
        fi = best_model.feature_importances_
        sorted_indices = np.argsort(fi)[::-1]
        feature_importance = {
            FEATURE_COLUMNS[i]: round(float(fi[i]), 4)
            for i in sorted_indices
        }

    full_report = {
        "best_model_name": best_name,
        "optimal_threshold": best_threshold,
        "feature_columns": FEATURE_COLUMNS,
        "feature_importance": feature_importance,
        "validation_comparison": {
            name: {k: v for k, v in res.items() if k not in ["roc_curve", "pr_curve"]}
            for name, res in validation_results.items()
        },
        "validation_curves": {
            name: {
                "roc_curve": res["roc_curve"],
                "pr_curve": res["pr_curve"]
            }
            for name, res in validation_results.items()
        },
        "test_metrics": test_metrics,
        "data_summary": data["raw_counts"],
        "trained_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"  Saved metrics report to: {METRICS_PATH}")
    print("\nModel training and benchmark completed successfully!")
    print("=" * 70)

    return full_report


if __name__ == "__main__":
    train_and_evaluate_models()
