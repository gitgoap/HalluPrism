"""
Scalar confidence baseline.

Computes confidence as the geometric mean of token probabilities
from the model's generation. This is the most common baseline in
the MLLM confidence/calibration literature.

This is what our source-aware decomposition is compared against.
"""

import math
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def compute_scalar_confidence(metadata: Dict) -> float:
    """
    Geometric mean of token probabilities.

    Args:
        metadata: dict from model.generate() containing "token_probs"
                  as a list of (token_str, probability) tuples.

    Returns:
        Scalar confidence in [0, 1].
    """
    token_probs = metadata.get("token_probs", [])
    if not token_probs:
        return 0.0

    probs = [p for _, p in token_probs if p > 0]
    if not probs:
        return 0.0

    log_mean = sum(math.log(p) for p in probs) / len(probs)
    return math.exp(log_mean)


def compute_first_token_confidence(metadata: Dict) -> float:
    """
    Alternative: confidence = probability of just the first generated token.
    Often used for yes/no tasks where the first token IS the answer.
    """
    token_probs = metadata.get("token_probs", [])
    if not token_probs:
        return 0.0
    _, prob = token_probs[0]
    return prob


def compute_answer_token_confidence(metadata: Dict, answer_start: int = 0, answer_end: int = 1) -> float:
    """
    Confidence over a specific span of tokens (e.g., just the answer tokens,
    excluding any explanation tokens).

    Args:
        answer_start: index of first answer token
        answer_end: index after last answer token

    Returns:
        Geometric mean of probabilities for tokens [answer_start:answer_end].
    """
    token_probs = metadata.get("token_probs", [])
    answer_probs = token_probs[answer_start:answer_end]
    if not answer_probs:
        return 0.0

    probs = [p for _, p in answer_probs if p > 0]
    if not probs:
        return 0.0

    log_mean = sum(math.log(p) for p in probs) / len(probs)
    return math.exp(log_mean)
