"""
Alignment Uncertainty (A)

Core idea: Probe whether the model correctly aligns visual content
with textual semantics. We do this by modifying the text prompt to
force explicit spatial/relational reasoning, then checking if the
answer changes.

Two intervention strategies:
  1. Chain-of-Thought Probe: Force the model to first describe what
     it sees, THEN answer. If adding reasoning changes the answer,
     the original answer had weak alignment.

  2. Negation Probe: Flip the question's polarity ("Is there a dog?"
     → "Is there NO dog?"). A well-aligned model should flip its
     answer. If it doesn't, alignment is broken.

Scoring:
  - cot_flipped: did chain-of-thought change the answer?
  - cot_confidence_delta: how much did confidence change?
  - negation_flipped: did negation properly flip the answer?
  
  A_score = weighted combination of alignment failures
  - A_score close to 1.0 → poor alignment (model doesn't truly ground)
  - A_score close to 0.0 → strong alignment (reasoning is grounded)

This requires 2 extra forward passes per sample.
"""

import logging
import re
from typing import Dict

from PIL import Image

from src.baselines.scalar_confidence import compute_scalar_confidence
from src.evaluation.answer_parser import normalize_answer

logger = logging.getLogger(__name__)


# ======================================================================
# Prompt Interventions
# ======================================================================

def make_cot_prompt(original_prompt: str) -> str:
    """
    Wrap the original prompt with a chain-of-thought instruction.
    Forces the model to reason about visual content before answering.
    """
    return (
        "Look at the image carefully. First, describe what you see in "
        "the image in one sentence. Then answer the following question.\n\n"
        f"{original_prompt}"
    )


def make_negation_prompt(original_prompt: str) -> str:
    """
    Negate the question to test alignment consistency.
    
    For Yes/No questions:
      "Is there a dog?" → "Is there NO dog in the image?"
      "Is the circle round?" → "Is the circle NOT round?"
    
    For True/False:
      "Is the following statement true?" → keeps same but we flip expected answer
    """
    prompt = original_prompt

    # Try to negate "Is there a ..." patterns
    prompt = re.sub(
        r"Is there (a |an )?",
        r"Is there NO ",
        prompt,
        count=1,
        flags=re.IGNORECASE,
    )

    # If that didn't match, try "Is the ..." patterns
    if prompt == original_prompt:
        prompt = re.sub(
            r"Is the ",
            "Is the following NOT the case: is the ",
            prompt,
            count=1,
            flags=re.IGNORECASE,
        )

    # If still no match, just prepend negation instruction
    if prompt == original_prompt:
        prompt = (
            "Answer the OPPOSITE of what you would normally answer "
            "to the following question.\n\n" + prompt
        )

    return prompt


def _answers_are_opposite(pred1: str, pred2: str) -> bool:
    """Check if two binary answers are logical opposites."""
    opposites = {
        ("yes", "no"), ("no", "yes"),
        ("true", "false"), ("false", "true"),
    }
    return (pred1.lower(), pred2.lower()) in opposites


# ======================================================================
# Main Computation
# ======================================================================

def compute_alignment_uncertainty(
    model,
    image: Image.Image,
    prompt: str,
    original_prediction: str,
    original_confidence: float,
    task_type: str = "yes_no",
) -> Dict:
    """
    Compute alignment uncertainty for a single sample.

    Args:
        model: loaded BaseModel instance
        image: the original PIL image
        prompt: the exact same prompt used in the baseline
        original_prediction: the model's original (greedy) answer
        original_confidence: scalar confidence from the original run
        task_type: "yes_no", "true_false", or "open_ended"

    Returns:
        Dict with:
            - a_score: overall alignment uncertainty (0-1)
            - cot_prediction: answer after chain-of-thought probe
            - cot_confidence: confidence after CoT
            - cot_flipped: did CoT change the answer?
            - negation_prediction: answer after negation probe
            - negation_correct_flip: did negation properly flip the answer?
    """
    results = {}

    # --- Intervention 1: Chain-of-Thought Probe ---
    cot_prompt = make_cot_prompt(prompt)
    cot_response, cot_metadata = model.generate(image, cot_prompt)
    cot_confidence = compute_scalar_confidence(cot_metadata)
    cot_prediction = normalize_answer(cot_response)

    cot_flipped = (cot_prediction != original_prediction)
    cot_conf_delta = abs(original_confidence - cot_confidence)

    results["cot_prediction"] = cot_prediction
    results["cot_raw_response"] = cot_response
    results["cot_confidence"] = cot_confidence
    results["cot_flipped"] = cot_flipped
    results["cot_confidence_delta"] = cot_conf_delta

    # --- Intervention 2: Negation Probe ---
    # Only meaningful for binary tasks (yes/no, true/false)
    negation_correct_flip = None
    if task_type in ("yes_no", "true_false"):
        neg_prompt = make_negation_prompt(prompt)
        neg_response, neg_metadata = model.generate(image, neg_prompt)
        neg_confidence = compute_scalar_confidence(neg_metadata)
        neg_prediction = normalize_answer(neg_response)

        # A well-aligned model should flip its answer
        negation_correct_flip = _answers_are_opposite(
            original_prediction, neg_prediction
        )

        results["negation_prediction"] = neg_prediction
        results["negation_raw_response"] = neg_response
        results["negation_confidence"] = neg_confidence
        results["negation_correct_flip"] = negation_correct_flip

    # --- Compute A_score ---
    # CoT component: if adding reasoning changes the answer, alignment is weak
    cot_penalty = 0.5 if cot_flipped else 0.0
    # Add confidence instability as a softer signal
    cot_penalty += min(cot_conf_delta, 0.5)

    # Negation component: if negation did NOT flip the answer, alignment is broken
    if negation_correct_flip is not None:
        neg_penalty = 0.0 if negation_correct_flip else 0.6
    else:
        neg_penalty = 0.0  # skip for open-ended

    # Weighted combination
    if task_type in ("yes_no", "true_false"):
        a_score = min(0.4 * cot_penalty + 0.6 * neg_penalty, 1.0)
    else:
        a_score = min(cot_penalty, 1.0)

    results["a_score"] = a_score

    return results
