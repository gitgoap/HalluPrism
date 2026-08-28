"""
Seed control utilities for reproducible experiments.

Usage:
    from src.utils.seed_control import set_global_seed
    set_global_seed(42)
"""

import random
import logging

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """
    Set random seed for Python's random module.
    Extend this function as needed for NumPy / PyTorch / HuggingFace.

    Args:
        seed: Integer seed value. Use the same seed for all runs.
    """
    random.seed(seed)
    logger.info(f"Global seed set to {seed}")

    try:
        import numpy as np
        np.random.seed(seed)
        logger.info("NumPy seed set.")
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info("PyTorch seed set.")
    except ImportError:
        pass

    try:
        import transformers
        transformers.set_seed(seed)
        logger.info("HuggingFace transformers seed set.")
    except ImportError:
        pass


# --------------------------------------------------------------------------
# Canonical split seeds to use throughout the project
# --------------------------------------------------------------------------

GLOBAL_SEED = 42

# Use these seeds for dataset splitting / sampling to ensure reproducibility
DATASET_SEEDS = {
    "HallusionBench": 42,
    "POPE": 42,
    "VSR": 42,
    "VizWiz": 42,
    "MMHal": 42,
    "DocVQA": 42,
}

# Human audit sampling seed
AUDIT_SAMPLE_SEED = 123
