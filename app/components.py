"""
Reusable visual components, Plotly chart builders, and UI helpers
for the AI Fraud Detection Streamlit Dashboard.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any, List

# Plotly dark fintech theme template
DARK_PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(21, 29, 48, 0.5)",
    font=dict(color="#F8FAFC", family="Segoe UI, sans-serif"),
    margin=dict(l=40, r=30, t=50, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)"),
)


def load_css(css_path: str):
    """Inject custom CSS into Streamlit."""
    try:
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"CSS styling failed to load: {e}")


def render_header(title: str, subtitle: str, icon: str = "🛡️"):
    """Render standardized top header banner."""
    st.markdown(
        f"""
        <div class="fintech-header">
            <h1 class="fintech-title">{icon} {title}</h1>
            <div class="fintech-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_kpi(label: str, value: str, subtext: str = "", delta_color: str = "#3B82F6"):
    """Render a single modern KPI metric block."""
    return f"""
    <div class="kpi-card" style="border-top: 3px solid {delta_color};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{subtext}</div>
    </div>
    """


def get_risk_badge_html(risk_level: str) -> str:
    """Return colored risk badge HTML."""
    risk = risk_level.upper()
    if risk == "HIGH":
        return '<span class="badge-high">⚠️ HIGH RISK</span>'
    elif risk == "MEDIUM":
        return '<span class="badge-medium">⚡ MEDIUM RISK</span>'
    else:
        return '<span class="badge-low">✅ LOW RISK</span>'


def plot_fraud_distribution_pie(legit_count: int, fraud_count: int) -> go.Figure:
    """Interactive Donut Chart of Legitimate vs Fraudulent transactions."""
    labels = ["Legitimate", "Fraudulent"]
    values = [legit_count, fraud_count]
    colors = ["#10B981", "#EF4444"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color="#0A0E1A", width=2)),
        textinfo="label+percent",
        hoverinfo="label+value+percent",
        textfont=dict(color="#FFFFFF", size=13)
    )])

    fig.update_layout(
        **DARK_PLOT_LAYOUT,
        title=dict(text="Transaction Class Distribution", font=dict(size=16, color="#F8FAFC")),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    return fig


def plot_amount_comparison(legit_amounts: List[float], fraud_amounts: List[float]) -> go.Figure:
    """Box plot comparing transaction amounts between legitimate and fraud cases."""
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=legit_amounts,
        name="Legitimate",
        marker_color="#10B981",
        boxpoints="outliers"
    ))
    fig.add_trace(go.Box(
        y=fraud_amounts,
        name="Fraudulent",
        marker_color="#EF4444",
        boxpoints="outliers"
    ))

    fig.update_layout(
        **DARK_PLOT_LAYOUT,
        title=dict(text="Transaction Amount Distribution ($)", font=dict(size=16, color="#F8FAFC")),
        yaxis=dict(type="log", title="Amount in EUR / USD (Log Scale)"),
        showlegend=True
    )
    return fig


def plot_confusion_matrix_heatmap(matrix: List[List[int]]) -> go.Figure:
    """Interactive Confusion Matrix Heatmap."""
    # matrix is [[TN, FP], [FN, TP]]
    z = matrix
    x = ["Predicted Legit", "Predicted Fraud"]
    y = ["Actual Legit", "Actual Fraud"]

    # Annotations with counts and labels
    annotations = []
    labels_map = [
        [f"TN: {z[0][0]:,}", f"FP: {z[0][1]:,}"],
        [f"FN: {z[1][0]:,}", f"TP: {z[1][1]:,}"]
    ]

    for i in range(2):
        for j in range(2):
            annotations.append(dict(
                x=x[j], y=y[i],
                text=labels_map[i][j],
                showarrow=False,
                font=dict(color="#FFFFFF", size=14, weight="bold")
            ))

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=[[0, "#151D30"], [0.5, "#1E3A8A"], [1, "#3B82F6"]],
        showscale=False
    ))

    fig.update_layout(
        **DARK_PLOT_LAYOUT,
        title=dict(text="Validation Confusion Matrix", font=dict(size=16, color="#F8FAFC")),
        annotations=annotations,
        yaxis=dict(autorange="reversed")
    )
    return fig


def plot_roc_curves_comparison(curves_dict: Dict[str, Any]) -> go.Figure:
    """Plot multiple ROC curves for model comparison."""
    fig = go.Figure()

    # Diagonal baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="#64748B", dash="dash"),
        name="Random Chance (AUC = 0.50)"
    ))

    palette = {"XGBoost": "#3B82F6", "Random Forest": "#10B981", "Logistic Regression": "#F59E0B"}

    for name, data in curves_dict.items():
        roc = data.get("roc_curve", {})
        auc_val = roc.get("auc", 0.5)
        color = palette.get(name, "#8B5CF6")
        fig.add_trace(go.Scatter(
            x=roc.get("fpr", []),
            y=roc.get("tpr", []),
            mode="lines",
            line=dict(color=color, width=2.5),
            name=f"{name} (AUC = {auc_val:.3f})"
        ))

    fig.update_layout(
        **DARK_PLOT_LAYOUT,
        title=dict(text="ROC Curves Comparison (Receiver Operating Characteristic)", font=dict(size=16, color="#F8FAFC")),
        xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[0, 1]),
        yaxis=dict(title="True Positive Rate (Sensitivity / Recall)", range=[0, 1.05]),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center")
    )
    return fig


def plot_pr_curves_comparison(curves_dict: Dict[str, Any]) -> go.Figure:
    """Plot multiple Precision-Recall curves for imbalanced evaluation."""
    fig = go.Figure()
    palette = {"XGBoost": "#3B82F6", "Random Forest": "#10B981", "Logistic Regression": "#F59E0B"}

    for name, data in curves_dict.items():
        pr = data.get("pr_curve", {})
        auc_val = pr.get("pr_auc", 0.0)
        color = palette.get(name, "#8B5CF6")
        fig.add_trace(go.Scatter(
            x=pr.get("recall", []),
            y=pr.get("precision", []),
            mode="lines",
            line=dict(color=color, width=2.5),
            name=f"{name} (PR-AUC = {auc_val:.3f})"
        ))

    fig.update_layout(
        **DARK_PLOT_LAYOUT,
        title=dict(text="Precision-Recall Curves (Critical for Imbalanced Fraud)", font=dict(size=16, color="#F8FAFC")),
        xaxis=dict(title="Recall (Detection Rate)", range=[0, 1]),
        yaxis=dict(title="Precision (Positive Predictive Value)", range=[0, 1.05]),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center")
    )
    return fig


def plot_feature_importance_bar(importance_dict: Dict[str, float], top_k: int = 12) -> go.Figure:
    """Horizontal bar chart for model feature importances."""
    items = list(importance_dict.items())[:top_k]
    items.reverse()  # Top feature at the top
    features = [x[0] for x in items]
    scores = [x[1] for x in items]

    fig = go.Figure(go.Bar(
        x=scores,
        y=features,
        orientation="h",
        marker=dict(
            color=scores,
            colorscale=[[0, "#1E3A8A"], [1, "#06B6D4"]],
            line=dict(color="#0A0E1A", width=1)
        )
    ))

    fig.update_layout(
        **DARK_PLOT_LAYOUT,
        title=dict(text=f"Top {top_k} Most Important Fraud Features", font=dict(size=16, color="#F8FAFC")),
        xaxis=dict(title="Relative Feature Importance Weight"),
        yaxis=dict(title="Feature"),
        height=450
    )
    return fig


def plot_shap_waterfall_bar(contributions: List[Dict[str, Any]], top_k: int = 10) -> go.Figure:
    """Diverging horizontal bar chart displaying SHAP contributions for a transaction."""
    subset = contributions[:top_k]
    subset.reverse()

    features = [f"{c['feature']} = {c['raw_value']}" for c in subset]
    shap_vals = [c["shap_value"] for c in subset]
    colors = ["#EF4444" if v > 0 else "#10B981" for v in shap_vals]

    fig = go.Figure(go.Bar(
        x=shap_vals,
        y=features,
        orientation="h",
        marker=dict(color=colors)
    ))

    fig.update_layout(
        **DARK_PLOT_LAYOUT,
        title=dict(text="SHAP Feature Impact Breakdown (Red: Raises Risk, Green: Lowers Risk)", font=dict(size=15, color="#F8FAFC")),
        xaxis=dict(title="SHAP Value (Contribution to Log-Odds of Fraud)"),
        yaxis=dict(title="Transaction Feature & Observed Value"),
        height=420
    )
    return fig
