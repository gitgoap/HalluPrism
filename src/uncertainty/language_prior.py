"""
Language-Prior Uncertainty (L)

Core idea: Replace the real image with a blank gray image and re-run
the model. If the model still gives the same answer with high confidence,
its knowledge comes from language priors (training data memorization),
NOT from the actual image.

Scoring:
  L_confidence = scalar_confidence(blank_image, same_prompt)
  L_score = L_confidence / original_confidence  (clamped to [0, 1])

  - L_score close to 1.0 → language prior dominates (model ignores image)
  - L_score close to 0.0 → model genuinely uses the image

This is the simplest and fastest of the three components.
One extra forward pass per sample.
"""

import logging
import math
from typing import Dict, Tuple

from PIL import Image

from src.baselines.scalar_confidence import compute_scalar_confidence
from src.evaluation.answer_parser import normalize_answer

logger = logging.getLogger(__name__)

# Default blank image: 384x384 gray (works for all model input sizes)
BLANK_SIZE = (384, 384)
BLANK_COLOR = (128, 128, 128)


def create_blank_image(size: Tuple[int, int] = BLANK_SIZE,
                       color: Tuple[int, int, int] = BLANK_COLOR) -> Image.Image:
    """Create a uniform gray image with no visual information."""
    return Image.new("RGB", size, color)


def compute_language_prior(
    model,
    prompt: str,
    original_prediction: str,
    original_confidence: float,
    blank_image: Image.Image = None,
) -> Dict:
    """
    Compute language-prior uncertainty for a single sample.

    Args:
        model: loaded BaseModel instance
        prompt: the exact same prompt used in the baseline
        original_prediction: the model's original (greedy) answer
        original_confidence: scalar confidence from the real image
        blank_image: optional pre-created blank image (for efficiency)

    Returns:
        Dict with:
            - lp_prediction: model's answer with blank image
            - lp_confidence: scalar confidence with blank image
            - lp_score: L_confidence / original_confidence (0-1)
            - lp_answer_matches: bool, did the answer stay the same?
    """
    if blank_image is None:
        blank_image = create_blank_image()

    # Run model with blank image
    lp_response, lp_metadata = model.generate(blank_image, prompt)
    lp_confidence = compute_scalar_confidence(lp_metadata)
    lp_prediction = normalize_answer(lp_response)

    # Does the model give the same answer without the image?
    answer_matches = (lp_prediction == original_prediction)

    # L score: how much of the original confidence is preserved with no image
    if original_confidence > 0:
        lp_score = min(lp_confidence / original_confidence, 1.0)
    else:
        lp_score = 0.0

    return {
        "lp_prediction": lp_prediction,
        "lp_raw_response": lp_response,
        "lp_confidence": lp_confidence,
        "lp_score": lp_score,
        "lp_answer_matches": answer_matches,
    }
