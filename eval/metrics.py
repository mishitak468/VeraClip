"""
veraclip/eval/metrics.py
All evaluation metrics in one place.
Called by evaluate.py and batch_score.py.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score,
)


def compute_all_metrics(
    labels: list[int],
    scores: list[float],
    threshold: float = 0.5,
) -> dict:
    """
    Full metric suite for the binary inconsistency task.

    Args:
        labels:    ground-truth binary labels (0=consistent, 1=inconsistent)
        scores:    model output in [0, 1]
        threshold: decision boundary

    Returns:
        dict with auc, accuracy, f1, precision, recall, ap, tp, tn, fp, fn
    """
    preds = [1 if s >= threshold else 0 for s in scores]

    auc  = roc_auc_score(labels, scores)
    ap   = average_precision_score(labels, scores)   # area under PR curve
    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    cm   = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    return {
        "auc":            round(float(auc),  4),
        "ap":             round(float(ap),   4),
        "accuracy":       round(float(acc),  4),
        "f1":             round(float(f1),   4),
        "precision":      round(float(prec), 4),
        "recall":         round(float(rec),  4),
        "true_positives":  int(tp),
        "true_negatives":  int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "threshold":       threshold,
        "n_samples":       len(labels),
    }


