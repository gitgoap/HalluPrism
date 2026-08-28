"""
Answer parser — normalize and compare model responses to gold answers.

Handles the quirks of each dataset's answer format:
  - HallusionBench: "yes"/"no" (from 1/0)
  - POPE: "yes"/"no"
  - VSR: "true"/"false"
  - VizWiz: free text or "unanswerable"
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def normalize_answer(text: str) -> str:
    """
    Normalize a model response to a canonical form.
    Lowercase, strip whitespace and punctuation.
    """
    if not text:
        return ""

    text = text.strip().lower()

    # Remove leading explanations — take just the first word/line
    # Models sometimes say "Yes, because..." or "True. The teddy bear..."
    lines = text.split("\n")
    first_line = lines[0].strip()

    # Remove trailing punctuation
    first_line = first_line.rstrip(".,!;:")

    # Yes/No normalization
    if first_line in ("yes", "yeah", "yep", "correct", "right"):
        return "yes"
    if first_line in ("no", "nope", "incorrect", "wrong"):
        return "no"

    # True/False normalization
    if first_line in ("true", "that is true", "the statement is true"):
        return "true"
    if first_line in ("false", "that is false", "the statement is false"):
        return "false"

    # Unanswerable
    if any(phrase in first_line for phrase in [
        "unanswerable", "not answerable", "cannot answer", "can't answer",
        "unable to answer", "cannot be answered", "not possible to answer",
        "i don't know", "i can't tell", "unclear", "cannot determine",
    ]):
        return "unanswerable"

    # For free-text answers (VizWiz), return cleaned first line
    # Remove articles and extra whitespace
    cleaned = re.sub(r"\b(a|an|the)\b", "", first_line)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def is_correct(
    prediction: str,
    gold: str,
    task_type: str = "VQA",
) -> bool:
    """
    Check if a prediction matches the gold answer.

    For yes/no and true/false tasks: exact match after normalization.
    For VQA: check if normalized prediction contains gold or vice versa.
    For answerability: check if both are "unanswerable" or both are valid answers.
    """
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)

    if task_type in ("yes_no", "spatial"):
        return pred_norm == gold_norm

    if task_type == "answerability":
        if gold_norm == "unanswerable":
            return pred_norm == "unanswerable"
        if pred_norm == "unanswerable":
            return False
        # Both are real answers — check overlap
        return pred_norm == gold_norm or pred_norm in gold_norm or gold_norm in pred_norm

    # Default VQA: substring match (lenient)
    return pred_norm == gold_norm or pred_norm in gold_norm or gold_norm in pred_norm
