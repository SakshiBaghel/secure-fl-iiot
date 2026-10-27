"""Comprehensive evaluation metrics calculation for intrusion detection systems.

Calculates Accuracy, Precision, Recall (Detection Rate), Specificity (1 - FPR),
F1-score, Confusion Matrix, and ROC AUC score.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_intrusion_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calculates security-oriented evaluation metrics for intrusion detection.

    Why Recall and F1-score are critical in Industrial IoT (IIoT):
    - False Negatives (FN) represent undetected cyber-attacks penetrating industrial infrastructure.
      In an operational SCADA or manufacturing setting, a missed attack can cause physical machinery
      damage, safety system override, or complete production halt. Recall = TP / (TP + FN) directly
      measures the proportion of attacks detected.
    - F1-score provides the harmonic mean of Precision and Recall, ensuring the IDS neither misses
      intrusions nor floods industrial operators with false alarms (alert fatigue).

    Args:
        y_true: Ground truth binary labels (0 = Normal, 1 = Attack).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Predicted probability of positive class (attack), optional for ROC AUC.

    Returns:
        Dictionary of formatted metrics and raw confusion matrix counts.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))  # Also known as Detection Rate (DR)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # False Positive Rate: FP / (FP + TN)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    auc = None
    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = None

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "detection_rate": round(rec, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "specificity": round(specificity, 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "matrix_2x2": cm.tolist(),
        },
        "sample_counts": {
            "total_tested": int(len(y_true)),
            "actual_normal": int(tn + fp),
            "actual_attacks": int(tp + fn),
        },
    }

    return metrics


def save_metrics_to_json(metrics: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Saves metrics dictionary to a JSON file."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def print_metrics_summary(title: str, metrics: Dict[str, Any]) -> None:
    """Pretty prints evaluation metrics to the console."""
    print("=" * 65)
    print(f" {title} ")
    print("=" * 65)
    print(f"  Accuracy                : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision               : {metrics['precision'] * 100:.2f}%")
    print(f"  Recall (Detection Rate) : {metrics['recall'] * 100:.2f}%  <-- Critical for safety")
    print(f"  F1-Score                : {metrics['f1_score']:.4f}")
    print(f"  False Positive Rate     : {metrics['false_positive_rate'] * 100:.2f}%")
    if metrics["roc_auc"] is not None:
        print(f"  ROC AUC Score           : {metrics['roc_auc']:.4f}")
    cm = metrics["confusion_matrix"]
    print("  Confusion Matrix:")
    print(f"    TN: {cm['true_negatives']:<5} | FP: {cm['false_positives']:<5} (Normal traffic)")
    print(f"    FN: {cm['false_negatives']:<5} | TP: {cm['true_positives']:<5} (Attack traffic)")
    print("=" * 65)
