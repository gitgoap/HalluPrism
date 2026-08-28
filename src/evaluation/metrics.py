"""
Evaluation metrics for the MLLM uncertainty project.

Covers:
  - Standard accuracy and classification metrics
  - Per-failure-family breakdown
  - Selective accuracy (for abstention evaluation)
  - Risk-coverage curves (for RQ4)
  - Cohen's kappa (for human audit agreement)
"""

import math
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ======================================================================
# Standard metrics
# ======================================================================

def accuracy(predictions: List[str], golds: List[str]) -> float:
    """Simple accuracy. Case-insensitive, stripped."""
    if not predictions:
        return 0.0
    correct = sum(
        1 for p, g in zip(predictions, golds)
        if p.strip().lower() == g.strip().lower()
    )
    return correct / len(predictions)


def binary_metrics(predictions: List[str], golds: List[str], positive: str = "yes") -> Dict:
    """
    Compute precision, recall, F1 for binary tasks (yes/no, true/false).

    Args:
        predictions: list of predicted labels
        golds: list of gold labels
        positive: the positive class string (e.g. "yes" or "true")

    Returns:
        Dict with accuracy, precision, recall, f1, tp, fp, fn, tn.
    """
    tp = fp = fn = tn = 0
    for pred, gold in zip(predictions, golds):
        p = pred.strip().lower() == positive.lower()
        g = gold.strip().lower() == positive.lower()
        if p and g:
            tp += 1
        elif p and not g:
            fp += 1
        elif not p and g:
            fn += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total": total,
    }


def per_family_accuracy(
    predictions: List[str],
    golds: List[str],
    families: List[str],
) -> Dict[str, Dict]:
    """
    Compute accuracy broken down by failure family.

    Returns:
        Dict mapping family -> {"accuracy": float, "count": int, "correct": int}
    """
    buckets = defaultdict(lambda: {"correct": 0, "count": 0})
    for pred, gold, family in zip(predictions, golds, families):
        buckets[family]["count"] += 1
        if pred.strip().lower() == gold.strip().lower():
            buckets[family]["correct"] += 1

    results = {}
    for family, vals in buckets.items():
        acc = vals["correct"] / vals["count"] if vals["count"] > 0 else 0.0
        results[family] = {"accuracy": acc, **vals}
    return results


# ======================================================================
# Selective accuracy (abstention evaluation)
# ======================================================================

def selective_accuracy(
    predictions: List[str],
    golds: List[str],
    confidences: List[float],
    coverage: float = 1.0,
) -> Dict:
    """
    Compute accuracy on the top-k% most confident predictions.

    Args:
        predictions: model predictions
        golds: gold answers
        confidences: confidence scores (higher = more confident)
        coverage: fraction of examples to keep (e.g. 0.8 = keep top 80%)

    Returns:
        Dict with selective_accuracy, coverage, n_answered, n_abstained.
    """
    n = len(predictions)
    n_keep = max(1, int(n * coverage))

    # Sort by confidence descending
    indexed = sorted(
        zip(confidences, predictions, golds),
        key=lambda x: x[0],
        reverse=True,
    )

    kept = indexed[:n_keep]
    correct = sum(1 for _, p, g in kept if p.strip().lower() == g.strip().lower())
    sel_acc = correct / n_keep if n_keep > 0 else 0.0

    return {
        "selective_accuracy": sel_acc,
        "coverage": coverage,
        "n_answered": n_keep,
        "n_abstained": n - n_keep,
        "n_correct": correct,
    }


def risk_coverage_curve(
    predictions: List[str],
    golds: List[str],
    confidences: List[float],
    n_points: int = 20,
) -> List[Dict]:
    """
    Compute risk-coverage curve data points.

    Risk = 1 - selective_accuracy at each coverage level.
    This is the main figure for RQ4.

    Returns:
        List of dicts with coverage, risk, selective_accuracy.
    """
    points = []
    for i in range(1, n_points + 1):
        cov = i / n_points
        result = selective_accuracy(predictions, golds, confidences, coverage=cov)
        result["risk"] = 1.0 - result["selective_accuracy"]
        points.append(result)
    return points


# ======================================================================
# AUROC for abstention quality
# ======================================================================

def auroc(
    confidences: List[float],
    correct_flags: List[bool],
) -> float:
    """
    Area Under the ROC Curve: how well does confidence separate
    correct from incorrect predictions?

    Higher AUROC = confidence is a better predictor of correctness.
    """
    n = len(confidences)
    if n == 0:
        return 0.5

    pairs = sorted(zip(confidences, correct_flags), reverse=True)

    n_pos = sum(1 for _, c in pairs if c)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Trapezoidal approximation
    tp = fp = 0
    prev_tp = prev_fp = 0
    auc = 0.0
    prev_conf = None

    for conf, correct in pairs:
        if conf != prev_conf and prev_conf is not None:
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
            prev_tp = tp
            prev_fp = fp
        if correct:
            tp += 1
        else:
            fp += 1
        prev_conf = conf

    auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
    return auc / (n_pos * n_neg)


# ======================================================================
# Cohen's kappa (inter-annotator agreement)
# ======================================================================

def cohens_kappa(labels_a: List[str], labels_b: List[str]) -> float:
    """
    Compute Cohen's kappa for inter-annotator agreement.
    Used for the human audit (Week 6).

    Args:
        labels_a: list of labels from annotator A
        labels_b: list of labels from annotator B

    Returns:
        Kappa value in [-1, 1]. Above 0.6 = substantial agreement.
    """
    assert len(labels_a) == len(labels_b), "Label lists must be same length"
    n = len(labels_a)
    if n == 0:
        return 0.0

    all_labels = sorted(set(labels_a) | set(labels_b))

    # Observed agreement
    agreed = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_o = agreed / n

    # Expected agreement by chance
    p_e = 0.0
    for label in all_labels:
        count_a = sum(1 for a in labels_a if a == label)
        count_b = sum(1 for b in labels_b if b == label)
        p_e += (count_a / n) * (count_b / n)

    if p_e >= 1.0:
        return 1.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    return kappa
