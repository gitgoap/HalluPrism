"""
Failure family classification.

Given model predictions and gold answers, classify each error into
one of the pre-defined failure families:
  - object_hallucination: model claims an object exists that doesn't (POPE)
  - spatial: model gets spatial/relational reasoning wrong (VSR)
  - attribute: model gets object attributes wrong (HallusionBench)
  - answerability: model answers when it should abstain (VizWiz)
  - false_certainty: model is confident but wrong (cross-dataset)

This module is used for RQ3: mapping uncertainty components to
failure families.
"""

import logging
from typing import Dict, List, Optional

from src.data.schema import UnifiedSample

logger = logging.getLogger(__name__)


# ======================================================================
# Failure family assignment
# ======================================================================

def classify_failure(sample: UnifiedSample) -> Optional[str]:
    """
    Classify a single incorrect prediction into a failure family.

    Logic:
    1. If prediction matches gold → not a failure → return None
    2. If dataset is POPE and model said "yes" but gold is "no" → object_hallucination
    3. If dataset is VSR → spatial
    4. If dataset is VizWiz and question is unanswerable but model answered → answerability
    5. If dataset is HallusionBench → use subcategory to decide
    6. Fallback → "other"
    """
    pred = (sample.model_prediction or "").strip().lower()
    gold = sample.gold_answer.strip().lower()

    # Correct prediction = no failure
    if pred == gold:
        return None

    dataset = sample.dataset_name

    # POPE: false positive = object hallucination
    if dataset == "POPE":
        if gold == "no" and pred == "yes":
            return "object_hallucination"
        elif gold == "yes" and pred == "no":
            return "missed_object"
        return "object_hallucination"

    # VSR: any error is a spatial/alignment error
    if dataset == "VSR":
        return "spatial"

    # VizWiz: unanswerable but model answered
    if dataset == "VizWiz":
        if sample.metadata.answerable is False and pred != "unanswerable":
            return "answerability"
        if sample.metadata.answerable is True:
            return "attribute"  # wrong answer on answerable question
        return "answerability"

    # HallusionBench: use subcategory
    if dataset == "HallusionBench":
        subcat = sample.metadata.subcategory or ""
        if "relation" in subcat:
            return "spatial"
        if "quantity" in subcat:
            return "attribute"
        return "object_hallucination"

    return "other"


def classify_failures_batch(samples: List[UnifiedSample]) -> Dict[str, List[int]]:
    """
    Classify all failures and return family -> list of sample indices.

    Returns:
        Dict mapping failure_family -> [indices of samples with that failure]
    """
    families: Dict[str, List[int]] = {}
    for idx, sample in enumerate(samples):
        family = classify_failure(sample)
        if family is not None:
            families.setdefault(family, []).append(idx)
    return families


def failure_family_summary(samples: List[UnifiedSample]) -> Dict[str, Dict]:
    """
    Produce a summary table of failure families.

    Returns:
        Dict mapping family -> {count, fraction, example_ids}
    """
    families = classify_failures_batch(samples)
    total_errors = sum(len(v) for v in families.values())

    summary = {}
    for family, indices in families.items():
        summary[family] = {
            "count": len(indices),
            "fraction": len(indices) / total_errors if total_errors > 0 else 0.0,
            "example_ids": [samples[i].sample_id for i in indices[:5]],
        }
    return summary
