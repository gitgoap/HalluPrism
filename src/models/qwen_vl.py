"""
Qwen3-VL-8B-Instruct model wrapper using HuggingFace transformers.

Model: Qwen/Qwen3-VL-8B-Instruct
Paper: Qwen-VL series (Alibaba)

Usage:
    from src.models.qwen_vl import QwenVLModel

    model = QwenVLModel()
    model.load()
    response, metadata = model.generate(image, prompt)
    confidence = model.get_confidence(metadata)
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

QWEN3_VL_8B = "Qwen/Qwen3-VL-8B-Instruct"


class QwenVLModel(BaseModel):
    """
    Qwen3-VL wrapper.

    Qwen3-VL uses a ViT vision encoder integrated with the Qwen language model.
    It supports dynamic resolution and multi-image inputs.

    On A100 (80 GB): 8B loads in float16 (~18 GB).
    """

    def __init__(
        self,
        model_name: str = QWEN3_VL_8B,
        model_path: Optional[str] = None,
        device: str = "cuda",
        cuda_device: Optional[str] = None,
        dtype: str = "bfloat16",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        load_in_4bit: bool = False,
    ):
        resolved_name = model_path if model_path else model_name
        super().__init__(
            model_name=resolved_name,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        self.cuda_device = cuda_device
        self.load_in_4bit = load_in_4bit

    def load(self) -> None:
        """Load Qwen3-VL model and processor onto GPU."""
        # qwen_vl_utils is required for processing vision inputs
        try:
            from qwen_vl_utils import process_vision_info
            self._process_vision_info = process_vision_info
        except ImportError:
            logger.warning(
                "qwen_vl_utils not found. Install with: pip install qwen-vl-utils. "
                "Falling back to direct PIL image passing."
            )
            self._process_vision_info = None
        from transformers import AutoProcessor, BitsAndBytesConfig

        logger.info(f"Loading Qwen3-VL model: {self.model_name}")

        torch_dtype = getattr(torch, self.dtype, torch.float16)

        if self.cuda_device:
            device_map = self.cuda_device
            logger.info(f"Pinning model to {self.cuda_device}")
        else:
            device_map = "auto"
            logger.info("Using device_map=auto")

        load_kwargs = {
            "pretrained_model_name_or_path": self.model_name,
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "low_cpu_mem_usage": True,
        }

        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["quantization_config"] = bnb_config

        # Try Qwen3-specific class first, then fall back
        try:
            from transformers import Qwen3VLForConditionalGeneration
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(**load_kwargs)
        except (ImportError, ValueError):
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(**load_kwargs)

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model.eval()

        logger.info(f"Qwen3-VL loaded. model={self.model_name}, device={device_map}, dtype={self.dtype}")

    def generate(
        self,
        image: Image.Image,
        prompt: str,
    ) -> Tuple[str, Dict]:
        """
        Generate response for a single (image, prompt) pair.

        Qwen-VL uses a chat template with image content blocks.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call model.load() first.")

        # Strip any <image> tokens
        clean_prompt = prompt.replace("<image>", "").strip()

        # Qwen-VL expects messages with image content
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": clean_prompt},
                ],
            }
        ]

        # Apply chat template to get text prompt
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision inputs using qwen_vl_utils if available
        if self._process_vision_info is not None:
            image_inputs, video_inputs = self._process_vision_info(messages)
        else:
            image_inputs = [image]
            video_inputs = None

        # Process inputs
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        # Remove token_type_ids if present — Qwen3-VL doesn't expect them
        inputs.pop("token_type_ids", None)

        input_len = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Decode response
        generated_ids = outputs.sequences[0][input_len:]
        response_text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()

        # Token probabilities
        token_probs = self._extract_token_probs(outputs.scores, generated_ids)

        metadata = {
            "token_probs": token_probs,
            "input_token_count": input_len,
            "output_token_count": len(generated_ids),
            "response_text": response_text,
        }

        return response_text, metadata

    def get_confidence(self, metadata: Dict) -> float:
        """Geometric mean of token probabilities."""
        token_probs = metadata.get("token_probs", [])
        if not token_probs:
            return 0.0

        probs = [p for _, p in token_probs if p > 0]
        if not probs:
            return 0.0

        log_mean = sum(math.log(p) for p in probs) / len(probs)
        return math.exp(log_mean)

    def _extract_token_probs(
        self, scores: tuple, generated_ids: torch.Tensor
    ) -> List[Tuple[str, float]]:
        """Extract (token_string, probability) pairs."""
        token_probs = []
        for step_idx, (score, token_id) in enumerate(zip(scores, generated_ids)):
            prob_dist = torch.softmax(score[0], dim=-1)
            token_prob = prob_dist[token_id].item()
            token_str = self.processor.decode([token_id])
            token_probs.append((token_str, token_prob))
        return token_probs
