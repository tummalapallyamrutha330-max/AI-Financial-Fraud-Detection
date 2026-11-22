"""
Model evaluation utilities for imbalanced fraud detection.
Prioritizes PR-AUC, F1-score, Recall, and Precision over raw accuracy.
Includes threshold tuning on validation set.
"""
from typing import Dict, Any, Tuple, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics for binary fraud classification.
    """
    cm = confusion_matrix(y_true, y_pred)
    # cm layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "matrix": [[tn, fp], [fn, tp]]
        },
        "rates": {
            "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0,
            "false_negative_rate": round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0.0,
        }
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    steps: int = 100
) -> Tuple[float, float]:
    """
    Find optimal classification threshold on the validation split
    by maximizing the F1-score (harmonic mean of Precision & Recall).
    Returns (best_threshold, best_f1).
    """
    thresholds = np.linspace(0.01, 0.99, steps)
    best_f1 = -1.0
    best_threshold = 0.5

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(t)

    return round(best_threshold, 4), round(float(best_f1), 4)


def generate_roc_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_points: int = 100
) -> Dict[str, Any]:
    """
    Generate subsampled FPR and TPR coordinates for interactive plotting.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    # Downsample points for efficient JSON serialization and UI rendering
    if len(fpr) > max_points:
        idx = np.round(np.linspace(0, len(fpr) - 1, max_points)).astype(int)
        fpr = fpr[idx]
        tpr = tpr[idx]

    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    return {
        "fpr": [round(float(x), 4) for x in fpr],
        "tpr": [round(float(y), 4) for y in tpr],
        "auc": round(roc_auc, 4)
    }


def generate_pr_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_points: int = 100
) -> Dict[str, Any]:
    """
    Generate subsampled Precision and Recall coordinates for interactive plotting.
    """
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    if len(prec) > max_points:
        idx = np.round(np.linspace(0, len(prec) - 1, max_points)).astype(int)
        prec = prec[idx]
        rec = rec[idx]

    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    return {
        "precision": [round(float(p), 4) for p in prec],
        "recall": [round(float(r), 4) for r in rec],
        "pr_auc": round(pr_auc, 4)
    }
