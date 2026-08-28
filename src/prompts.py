"""
Prompt templates for each dataset and task type.

Each template is designed to produce short, parseable answers.
The model wrappers accept these prompts directly.

Usage:
    from src.prompts import get_prompt

    prompt = get_prompt("hallusionbench", question="Is the circle round?")
"""

from typing import Optional


# ======================================================================
# Dataset-specific prompt templates
# ======================================================================

def hallusionbench_prompt(question: str) -> str:
    """
    HallusionBench: Yes/No visual reasoning.
    Forces short answer for reliable parsing.
    """
    return (
        f"{question}\n"
        "Answer with only 'Yes' or 'No'."
    )


def pope_prompt(question: str) -> str:
    """
    POPE: Yes/No object presence probing.
    Question is already in the form 'Is there a X in the image?'
    """
    return (
        f"{question}\n"
        "Answer with only 'Yes' or 'No'."
    )


def vsr_prompt(caption: str) -> str:
    """
    VSR: True/False spatial caption verification.
    The input is a caption, not a question.
    """
    return (
        f"Look at the image carefully. Is the following statement true or false?\n"
        f"Statement: \"{caption}\"\n"
        "Answer with only 'True' or 'False'."
    )


def vizwiz_prompt(question: str) -> str:
    """
    VizWiz: Open-ended VQA with answerability.
    Must handle unanswerable questions.
    """
    return (
        f"{question}\n"
        "If the image is too unclear to answer, or the question cannot be "
        "answered from the image, say 'unanswerable'. "
        "Otherwise, give a short answer (one or two words)."
    )


# ======================================================================
# Self-reported confidence prompt (baseline)
# ======================================================================

def self_reported_confidence_prompt(question: str, dataset: str) -> str:
    """
    Prompt the model to answer AND report its confidence.
    Used for the self-reported confidence baseline.
    """
    base = get_prompt(dataset, question)
    return (
        f"{base}\n\n"
        "After your answer, on a new line, state your confidence as a "
        "percentage (0-100). Format: 'Confidence: XX%'"
    )


# ======================================================================
# Anti-prior prompt (language-prior intervention, RQ5)
# ======================================================================

def anti_prior_prompt(question: str, dataset: str) -> str:
    """
    Intervention prompt that pushes the model to rely on image evidence
    rather than language priors.
    """
    base = get_prompt(dataset, question)
    return (
        "Look at the image very carefully before answering. "
        "Do NOT guess based on common sense or what is usually true. "
        "Base your answer ONLY on what you can actually see in this specific image.\n\n"
        f"{base}"
    )


# ======================================================================
# Relation-check prompt (alignment intervention, RQ5)
# ======================================================================

def relation_check_prompt(caption: str) -> str:
    """
    Intervention prompt for spatial/relation tasks (VSR).
    Asks the model to first identify objects, then check the relation.
    """
    return (
        "Follow these steps:\n"
        "1. First, list the objects you can see in the image.\n"
        "2. Then, describe the spatial relationship between them.\n"
        f"3. Finally, is this statement true or false: \"{caption}\"\n"
        "Answer with only 'True' or 'False' on the last line."
    )


# ======================================================================
# Dispatcher
# ======================================================================

_PROMPT_MAP = {
    "hallusionbench": hallusionbench_prompt,
    "HallusionBench": hallusionbench_prompt,
    "pope": pope_prompt,
    "POPE": pope_prompt,
    "vsr": vsr_prompt,
    "VSR": vsr_prompt,
    "vizwiz": vizwiz_prompt,
    "VizWiz": vizwiz_prompt,
}


def get_prompt(dataset: str, question: str) -> str:
    """Get the appropriate prompt for a dataset."""
    fn = _PROMPT_MAP.get(dataset)
    if fn is None:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose from {list(_PROMPT_MAP)}")
    return fn(question)
