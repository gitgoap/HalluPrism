"""
Uncertainty Decomposition — Package Init

This package implements the three source-aware uncertainty components:
  V — Visual Uncertainty (image perturbation sensitivity)
  L — Language-Prior Uncertainty (blank-image baseline)
  A — Alignment Uncertainty (text-visual reasoning probe)
"""

from src.uncertainty.language_prior import compute_language_prior
from src.uncertainty.visual import compute_visual_uncertainty
from src.uncertainty.alignment import compute_alignment_uncertainty

__all__ = [
    "compute_language_prior",
    "compute_visual_uncertainty",
    "compute_alignment_uncertainty",
]
