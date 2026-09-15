"""
Evaluation metrics suite for fraud detection:
- Precision, Recall, F1-Score
- AUPRC (Area Under Precision-Recall Curve) - Primary metric for fraud
- ROC-AUC
- Confusion Matrix & Latency
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = 0.0

    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        auprc = 0.0

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "auprc": round(auprc, 4),
        "confusion_matrix": cm,
        "threshold": threshold,
    }


def print_metrics_table(model_name: str, metrics: Dict[str, Any]):
    """Pretty prints metrics for presentation / logging."""
    print("=" * 55)
    print(f"📊 Results: {model_name.upper()}")
    print("=" * 55)
    print(f"  Precision:         {metrics['precision']*100:.2f}%")
    print(f"  Recall:            {metrics['recall']*100:.2f}%")
    print(f"  F1-Score:          {metrics['f1_score']*100:.2f}%")
    print(f"  AUPRC (Key):       {metrics['auprc']:.4f}")
    print(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix:  {metrics['confusion_matrix']}")
    print("=" * 55)
