"""Shared binary ROC/AUC calculations for reports and plots."""

import numpy as np


def roc_curve(y_true, y_score):
    """Return aligned FPR, TPR and thresholds; no curve for a single class.

    Equal scores are evaluated together. Unknown labels and invalid scores
    must be removed by the caller, rather than counted as benign clients.
    """
    labels = np.asarray(y_true).reshape(-1)
    scores = np.asarray(y_score, dtype=float).reshape(-1)
    if labels.size != scores.size:
        raise ValueError("Labels and scores must have the same length.")
    if not np.all(np.isin(labels, [0, 1])) or not np.all(np.isfinite(scores)):
        raise ValueError("ROC requires binary labels and finite scores.")
    positives = np.count_nonzero(labels == 1)
    negatives = labels.size - positives
    if not positives or not negatives:
        return np.array([]), np.array([]), np.array([])

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), labels.size - 1]
    true_positives = np.cumsum(sorted_labels == 1)[ends]
    false_positives = ends + 1 - true_positives
    return (
        np.r_[0.0, false_positives / negatives],
        np.r_[0.0, true_positives / positives],
        np.r_[np.inf, sorted_scores[ends]],
    )


def auc(fpr, tpr):
    """Integrate a ROC curve without reordering its vertical segments."""
    x = np.asarray(fpr, dtype=float).reshape(-1)
    y = np.asarray(tpr, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("FPR and TPR must have the same length.")
    if x.size < 2:
        return float("nan")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("ROC coordinates must be finite.")
    widths = np.diff(x)
    if np.any(widths < 0):
        raise ValueError("FPR must be non-decreasing.")
    return float(np.sum(widths * (y[:-1] + y[1:]) / 2))
