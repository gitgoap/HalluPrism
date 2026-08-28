"""
Visual Uncertainty (V)

Core idea: Apply multiple visual perturbations (blur, crop, noise, 
brightness) to the image and re-run the model. If the model's answer
or confidence changes significantly, it has high visual uncertainty —
its answer depends on fragile visual details.

Perturbations applied:
  1. Gaussian blur (heavy)    — destroys fine-grained detail
  2. Center crop (aggressive) — removes peripheral context
  3. Brightness reduction     — simulates low-light conditions
  4. Gaussian noise           — simulates camera/sensor noise

Scoring:
  For each perturbation:
    - prediction_flipped? (binary)
    - confidence_delta = |original_conf - perturbed_conf|

  V_score = 1 - mean(agreement_rate across perturbations)
  V_confidence_shift = mean(confidence_delta across perturbations)

  - V_score close to 1.0 → highly visually uncertain (answer keeps changing)
  - V_score close to 0.0 → visually robust (answer is stable)

This requires 4 extra forward passes per sample (one per perturbation).
"""

import logging
import math
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

from src.baselines.scalar_confidence import compute_scalar_confidence
from src.evaluation.answer_parser import normalize_answer

logger = logging.getLogger(__name__)


# ======================================================================
# Image Perturbation Functions
# ======================================================================

def apply_blur_heavy(image: Image.Image) -> Image.Image:
    """Heavy Gaussian blur — destroys fine-grained detail."""
    return image.filter(ImageFilter.GaussianBlur(radius=15))


def apply_crop_aggressive(image: Image.Image) -> Image.Image:
    """Center crop to 50% — removes peripheral context."""
    w, h = image.size
    left = w // 4
    top = h // 4
    right = w - left
    bottom = h - top
    cropped = image.crop((left, top, right, bottom))
    # Resize back to original dimensions so model input shape is unchanged
    return cropped.resize((w, h), Image.BILINEAR)


def apply_brightness_low(image: Image.Image) -> Image.Image:
    """Reduce brightness to 30% — simulates low-light."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(0.3)


def apply_noise(image: Image.Image) -> Image.Image:
    """Add Gaussian noise — simulates sensor noise."""
    arr = np.array(image).astype(np.float32)
    noise = np.random.normal(0, 25, arr.shape)  # stddev=25
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


# Registry of all perturbations
PERTURBATIONS = {
    "blur_heavy": apply_blur_heavy,
    "crop_aggressive": apply_crop_aggressive,
    "brightness_low": apply_brightness_low,
    "noise": apply_noise,
}


# ======================================================================
# Main Computation
# ======================================================================

def compute_visual_uncertainty(
    model,
    image: Image.Image,
    prompt: str,
    original_prediction: str,
    original_confidence: float,
    perturbation_names: List[str] = None,
) -> Dict:
    """
    Compute visual uncertainty for a single sample.

    Args:
        model: loaded BaseModel instance
        image: the original PIL image
        prompt: the exact same prompt used in the baseline
        original_prediction: the model's original (greedy) answer
        original_confidence: scalar confidence from the original run
        perturbation_names: which perturbations to apply (default: all 4)

    Returns:
        Dict with:
            - v_score: overall visual uncertainty (0-1, higher = more uncertain)
            - v_confidence_shift: mean absolute confidence change
            - v_flip_rate: fraction of perturbations that flipped the answer
            - v_per_perturbation: detailed per-perturbation results
    """
    if perturbation_names is None:
        perturbation_names = list(PERTURBATIONS.keys())

    per_perturbation = {}
    flips = 0
    confidence_deltas = []

    for name in perturbation_names:
        perturb_fn = PERTURBATIONS.get(name)
        if perturb_fn is None:
            logger.warning(f"Unknown perturbation: {name}, skipping")
            continue

        # Apply perturbation
        perturbed_image = perturb_fn(image)

        # Run model on perturbed image
        response, metadata = model.generate(perturbed_image, prompt)
        conf = compute_scalar_confidence(metadata)
        pred = normalize_answer(response)

        # Compare with original
        flipped = (pred != original_prediction)
        conf_delta = abs(original_confidence - conf)

        if flipped:
            flips += 1
        confidence_deltas.append(conf_delta)

        per_perturbation[name] = {
            "prediction": pred,
            "confidence": conf,
            "flipped": flipped,
            "confidence_delta": conf_delta,
        }

    n_perturbations = len(per_perturbation)
    if n_perturbations == 0:
        return {"v_score": 0.0, "v_confidence_shift": 0.0,
                "v_flip_rate": 0.0, "v_per_perturbation": {}}

    flip_rate = flips / n_perturbations
    mean_conf_shift = sum(confidence_deltas) / n_perturbations

    # V_score: higher = more visually uncertain
    # Combine flip rate (binary instability) and confidence shift (continuous)
    v_score = 0.6 * flip_rate + 0.4 * min(mean_conf_shift * 2, 1.0)

    return {
        "v_score": v_score,
        "v_confidence_shift": mean_conf_shift,
        "v_flip_rate": flip_rate,
        "v_n_perturbations": n_perturbations,
        "v_per_perturbation": per_perturbation,
    }
