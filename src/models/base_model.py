"""
Abstract base class for MLLM wrappers.

All model wrappers (LLaVA, InstructBLIP, etc.) inherit from this.
Downstream code only interacts through the BaseModel interface,
so swapping models requires zero code changes.
"""

import abc
import logging
from typing import Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


class BaseModel(abc.ABC):
    """
    Shared interface for all MLLMs used in this project.

    Every model must implement:
        - generate()      : produce a text response for (image, prompt)
        - get_confidence() : extract scalar confidence from the generation

    Optional to override:
        - generate_with_samples() : MC sampling for disagreement baseline
        - generate_multiple()     : batch generation
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "float16",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = None
        self.processor = None

    @abc.abstractmethod
    def load(self) -> None:
        """Load model and processor into memory."""
        ...

    @abc.abstractmethod
    def generate(
        self,
        image: Image.Image,
        prompt: str,
    ) -> Tuple[str, Dict]:
        """
        Generate a response for a single (image, prompt) pair.

        Args:
            image: PIL Image.
            prompt: Text prompt string.

        Returns:
            (response_text, metadata_dict)
            metadata_dict should include at minimum:
                - "token_probs": list of (token_str, probability) for generated tokens
                - "input_token_count": int
                - "output_token_count": int
        """
        ...

    @abc.abstractmethod
    def get_confidence(self, metadata: Dict) -> float:
        """
        Extract scalar confidence from generation metadata.
        Default: geometric mean of token probabilities.
        """
        ...

    def generate_with_samples(
        self,
        image: Image.Image,
        prompt: str,
        n_samples: int = 5,
        temperature: float = 0.7,
    ) -> List[Tuple[str, Dict]]:
        """
        Generate n_samples responses with temperature sampling.
        Used for sampling disagreement baseline.

        Default implementation: call generate() n times with temperature.
        Override for batch-efficient implementations.
        """
        results = []
        original_temp = self.temperature
        self.temperature = temperature
        for _ in range(n_samples):
            result = self.generate(image, prompt)
            results.append(result)
        self.temperature = original_temp
        return results

    def generate_with_blank_image(
        self,
        image: Image.Image,
        prompt: str,
        blank_color: Tuple[int, int, int] = (128, 128, 128),
    ) -> Tuple[str, Dict]:
        """
        Generate using a blank image of the same size.
        Used for language-prior uncertainty estimation.
        """
        blank = Image.new(image.mode, image.size, blank_color)
        return self.generate(blank, prompt)

    def is_loaded(self) -> bool:
        return self.model is not None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r}, device={self.device!r})"
