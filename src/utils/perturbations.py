"""
Image perturbation utilities for visual uncertainty estimation.

These functions are used to generate the set P of controlled visual
perturbations for computing V(x, q) — visual uncertainty.

All functions take a PIL Image and return a perturbed PIL Image.
"""

import random
from typing import Callable, List, Tuple

try:
    from PIL import Image, ImageFilter, ImageEnhance
    import PIL
except ImportError:
    raise ImportError("Pillow is required: pip install Pillow")


# --------------------------------------------------------------------------
# Image perturbation functions
# --------------------------------------------------------------------------

def gaussian_blur(image: "Image.Image", radius: float = 5.0) -> "Image.Image":
    """Apply Gaussian blur to the full image."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def center_crop(image: "Image.Image", crop_fraction: float = 0.5) -> "Image.Image":
    """
    Crop to a central square patch and resize back to original dimensions.
    A high crop_fraction (e.g. 0.9) removes only a small border.
    A low crop_fraction (e.g. 0.3) keeps only the center third.
    """
    w, h = image.size
    left = int(w * (1 - crop_fraction) / 2)
    top = int(h * (1 - crop_fraction) / 2)
    right = w - left
    bottom = h - top
    cropped = image.crop((left, top, right, bottom))
    return cropped.resize((w, h), PIL.Image.BICUBIC)


def patch_occlusion(
    image: "Image.Image",
    patch_fraction: float = 0.4,
    fill_color: Tuple[int, int, int] = (128, 128, 128),
    seed: int = 42,
) -> "Image.Image":
    """
    Occlude a random patch of the image with a solid color.
    patch_fraction controls the fraction of width and height occluded.
    """
    rng = random.Random(seed)
    img = image.copy()
    w, h = img.size
    ph = int(h * patch_fraction)
    pw = int(w * patch_fraction)
    top = rng.randint(0, h - ph)
    left = rng.randint(0, w - pw)

    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([left, top, left + pw, top + ph], fill=fill_color)
    return img


def brightness_degradation(image: "Image.Image", factor: float = 0.3) -> "Image.Image":
    """
    Reduce image brightness by a multiplicative factor.
    factor=1.0 is unchanged; factor=0.0 is black.
    """
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def add_noise(image: "Image.Image", noise_std: float = 40.0, seed: int = 42) -> "Image.Image":
    """Add Gaussian pixel noise to the image."""
    import struct
    rng = random.Random(seed)
    pixels = list(image.getdata())
    noisy = []
    for p in pixels:
        if isinstance(p, int):
            noisy_val = max(0, min(255, round(p + rng.gauss(0, noise_std))))
            noisy.append(noisy_val)
        else:
            noisy_channel = tuple(
                max(0, min(255, round(c + rng.gauss(0, noise_std)))) for c in p
            )
            noisy.append(noisy_channel)
    result = Image.new(image.mode, image.size)
    result.putdata(noisy)
    return result


def blank_image(image: "Image.Image", fill_color: Tuple[int, int, int] = (128, 128, 128)) -> "Image.Image":
    """
    Replace image with a uniform gray/blank image of the same size.
    Used for language-prior uncertainty: L(x,q) = KL(model(x,q) || model(blank,q)).
    """
    return Image.new(image.mode, image.size, fill_color)


# --------------------------------------------------------------------------
# Perturbation set P used for visual uncertainty estimation
# --------------------------------------------------------------------------

VISUAL_PERTURBATIONS: List[Tuple[str, Callable]] = [
    ("blur_light",       lambda img: gaussian_blur(img, radius=3.0)),
    ("blur_heavy",       lambda img: gaussian_blur(img, radius=8.0)),
    ("crop_moderate",    lambda img: center_crop(img, crop_fraction=0.7)),
    ("crop_aggressive",  lambda img: center_crop(img, crop_fraction=0.4)),
    ("occlusion",        lambda img: patch_occlusion(img, patch_fraction=0.4)),
    ("brightness_low",   lambda img: brightness_degradation(img, factor=0.3)),
    ("noise",            lambda img: add_noise(img, noise_std=40.0)),
]


def apply_all_perturbations(
    image: "Image.Image",
    perturbations: List[Tuple[str, Callable]] = VISUAL_PERTURBATIONS,
) -> List[Tuple[str, "Image.Image"]]:
    """
    Apply all perturbations in the set P and return (name, perturbed_image) pairs.

    Returns:
        List of (perturbation_name, perturbed_image) tuples.
    """
    results = []
    for name, fn in perturbations:
        try:
            perturbed = fn(image)
            results.append((name, perturbed))
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Perturbation {name!r} failed: {e}"
            )
    return results


# --------------------------------------------------------------------------
# Text perturbation utilities (for alignment uncertainty)
# --------------------------------------------------------------------------

# Standard spatial relation swap pairs for VSR
RELATION_SWAPS: List[Tuple[str, str]] = [
    ("in front of", "behind"),
    ("behind", "in front of"),
    ("left of", "right of"),
    ("right of", "left of"),
    ("above", "below"),
    ("below", "above"),
    ("on top of", "underneath"),
    ("underneath", "on top of"),
    ("inside", "outside"),
    ("outside", "inside"),
    ("near", "far from"),
    ("next to", "away from"),
]


def apply_relation_swap(text: str, swaps: List[Tuple[str, str]] = RELATION_SWAPS) -> List[Tuple[str, str]]:
    """
    Apply all applicable relation swaps to the text.
    Returns a list of (swap_name, modified_text) pairs.
    Only swaps that actually appear in the text are applied.
    """
    results = []
    for original, replacement in swaps:
        if original in text.lower():
            swapped = text.lower().replace(original, replacement, 1)
            # Restore original casing on first character
            if text and swapped:
                swapped = swapped[0].upper() + swapped[1:]
            results.append((f"swap_{original}_to_{replacement}", swapped))
    return results
