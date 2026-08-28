"""
Self-reported (verbalized) confidence baseline.

Asks the model to state its confidence as a percentage,
then parses the number from the response.

This baseline tests whether models can accurately self-assess.
Most literature shows self-reported confidence is poorly calibrated.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def parse_self_reported_confidence(response: str) -> Optional[float]:
    """
    Parse a confidence percentage from model output.

    Looks for patterns like:
      - "Confidence: 85%"
      - "I am 90% confident"
      - "confidence = 75%"
      - "My confidence is 60 percent"

    Returns:
        Float in [0, 1] or None if no confidence found.
    """
    # Pattern 1: "Confidence: XX%"
    match = re.search(r"[Cc]onfidence[:\s=]+(\d{1,3})%", response)
    if match:
        return _clamp(int(match.group(1)) / 100.0)

    # Pattern 2: "XX% confident"
    match = re.search(r"(\d{1,3})%\s*confident", response)
    if match:
        return _clamp(int(match.group(1)) / 100.0)

    # Pattern 3: "confidence is XX"
    match = re.search(r"confidence\s+(?:is|=|:)\s*(\d{1,3})", response, re.IGNORECASE)
    if match:
        val = int(match.group(1))
        return _clamp(val / 100.0 if val > 1 else val)

    # Pattern 4: "XX percent"
    match = re.search(r"(\d{1,3})\s*percent", response, re.IGNORECASE)
    if match:
        return _clamp(int(match.group(1)) / 100.0)

    logger.debug(f"Could not parse confidence from: {response[:100]}...")
    return None


def extract_answer_and_confidence(response: str) -> tuple:
    """
    Split a response into answer text and confidence value.

    Assumes format:
        Answer text
        Confidence: XX%

    Returns:
        (answer_text: str, confidence: Optional[float])
    """
    confidence = parse_self_reported_confidence(response)

    # Remove the confidence line from the answer
    answer = re.sub(
        r"\n\s*[Cc]onfidence[:\s=]+\d{1,3}%?\s*$",
        "",
        response
    ).strip()

    return answer, confidence


def _clamp(value: float) -> float:
    """Clamp to [0, 1]."""
    return max(0.0, min(1.0, value))
