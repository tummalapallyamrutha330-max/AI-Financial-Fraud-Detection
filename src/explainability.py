"""
Explainable AI (XAI) engine using SHAP (SHapley Additive exPlanations).
Provides global feature importance, individual transaction breakdown,
positive/negative risk factors, and non-technical business interpretations.
"""
import os
import joblib
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List, Optional

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import FEATURE_COLUMNS, preprocess_input_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_fraud_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")


class FraudExplainer:
    """
    SHAP-based explainability engine for financial fraud risk decisions.
    """
    def __init__(
        self,
        model_path: str = BEST_MODEL_PATH,
        scaler_path: str = SCALER_PATH,
        background_data: Optional[pd.DataFrame] = None
    ):
        self.model = joblib.load(model_path) if os.path.exists(model_path) else None
        self.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        self.explainer = None
        self.expected_value = 0.0

        if self.model is not None:
            self._init_explainer(background_data)

    def _init_explainer(self, background_data: Optional[pd.DataFrame] = None):
        """Initialize SHAP TreeExplainer or fallback."""
        try:
            # TreeExplainer is fast and exact for XGBoost / RandomForest
            self.explainer = shap.TreeExplainer(self.model)
            ev = self.explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                self.expected_value = float(ev[1]) if len(ev) > 1 else float(ev[0])
            else:
                self.expected_value = float(ev)
        except Exception as e:
            print(f"Standard TreeExplainer init note: {e}. Using background sample...")
            if background_data is not None:
                sample_bg = preprocess_input_features(background_data.head(100), self.scaler)
                self.explainer = shap.Explainer(self.model, sample_bg)
            else:
                self.explainer = shap.Explainer(self.model)

    def explain_transaction(
        self,
        features: Dict[str, float],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Explain an individual transaction decision.
        Extracts positive (risk-increasing) and negative (risk-reducing) feature impacts.
        """
        if self.explainer is None or self.scaler is None:
            raise RuntimeError("Explainer or Scaler not initialized. Model artifacts missing.")

        df_input = pd.DataFrame([features])
        df_scaled = preprocess_input_features(df_input, self.scaler)

        shap_values = self.explainer.shap_values(df_scaled)

        # Handle various SHAP return shapes for binary classifiers
        if isinstance(shap_values, list):
            # Binary classifier: index 1 corresponds to fraud class
            values = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif len(shap_values.shape) == 3:
            values = shap_values[0, :, 1]
        elif len(shap_values.shape) == 2:
            values = shap_values[0]
        else:
            values = shap_values

        contributions = []
        for col, val, raw_val in zip(FEATURE_COLUMNS, values, df_input.iloc[0]):
            contributions.append({
                "feature": col,
                "raw_value": round(float(raw_val), 4),
                "shap_value": round(float(val), 4),
                "impact": "Increases Fraud Risk" if val > 0 else "Decreases Fraud Risk",
                "abs_impact": abs(float(val))
            })

        # Sort by absolute SHAP magnitude
        contributions.sort(key=lambda x: x["abs_impact"], reverse=True)

        positive_factors = [c for c in contributions if c["shap_value"] > 0][:top_k]
        negative_factors = [c for c in contributions if c["shap_value"] < 0][:top_k]

        # Plain language non-technical narrative
        narrative = self._generate_narrative(positive_factors, negative_factors, features.get("Amount", 0.0))

        return {
            "expected_value": round(float(self.expected_value), 4),
            "top_positive_factors": positive_factors,
            "top_negative_factors": negative_factors,
            "all_contributions": contributions[:15],
            "narrative_explanation": narrative
        }

    def _generate_narrative(
        self,
        positive_factors: List[Dict[str, Any]],
        negative_factors: List[Dict[str, Any]],
        amount: float
    ) -> str:
        """Constructs an intuitive, non-technical financial analyst summary."""
        parts = []

        if positive_factors:
            top_risks = ", ".join([f"{f['feature']} (+{f['shap_value']:.3f})" for f in positive_factors[:3]])
            parts.append(f"Primary risk drivers elevating fraud probability: {top_risks}.")
        else:
            parts.append("No significant abnormal risk-elevating patterns detected.")

        if negative_factors:
            top_safe = ", ".join([f"{f['feature']} ({f['shap_value']:.3f})" for f in negative_factors[:3]])
            parts.append(f"Mitigating factors aligning with typical legitimate behavior: {top_safe}.")

        parts.append(f"Transaction amount screened: ${amount:,.2f}.")
        return " ".join(parts)

    def get_global_importance(
        self,
        sample_df: pd.DataFrame,
        top_k: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Compute mean absolute SHAP values across a representative data sample
        to determine global feature hierarchy.
        """
        if self.explainer is None:
            return []

        df_scaled = preprocess_input_features(sample_df.head(200), self.scaler)
        shap_values = self.explainer.shap_values(df_scaled)

        if isinstance(shap_values, list):
            values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif len(shap_values.shape) == 3:
            values = shap_values[:, :, 1]
        else:
            values = shap_values

        mean_abs = np.mean(np.abs(values), axis=0)
        sorted_indices = np.argsort(mean_abs)[::-1]

        results = []
        for idx in sorted_indices[:top_k]:
            results.append({
                "feature": FEATURE_COLUMNS[idx],
                "mean_absolute_shap": round(float(mean_abs[idx]), 4),
                "rank": len(results) + 1
            })

        return results
