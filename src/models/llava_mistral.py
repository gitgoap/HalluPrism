"""
LLaVA-1.6-Mistral-7B (LLaVA-NeXT) model wrapper using HuggingFace transformers.

Model: llava-hf/llava-v1.6-mistral-7b-hf

Usage:
    from src.models.llava_mistral import LlavaMistralModel

    model = LlavaMistralModel()
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

LLAVA_1_6_MISTRAL_7B = "llava-hf/llava-v1.6-mistral-7b-hf"


class LlavaMistralModel(BaseModel):
    """
    LLaVA-v1.6 (LLaVA-NeXT) wrapper for Mistral 7B backend.
    """

    def __init__(
        self,
        model_name: str = LLAVA_1_6_MISTRAL_7B,
        model_path: Optional[str] = None,
        device: str = "cuda",
        cuda_device: Optional[str] = None,
        dtype: str = "float16",
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
        """Load LLaVA-1.6-Mistral model and processor onto GPU."""
        from transformers import (
            LlavaNextForConditionalGeneration,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        logger.info(f"Loading LLaVA-1.6-Mistral model: {self.model_name}")

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

        self.model = LlavaNextForConditionalGeneration.from_pretrained(**load_kwargs)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model.eval()

        logger.info(f"LLaVA-1.6-Mistral loaded. model={self.model_name}, device={device_map}, dtype={self.dtype}")

    def generate(
        self,
        image: Image.Image,
        prompt: str,
    ) -> Tuple[str, Dict]:
        """
        Generate response for a single (image, prompt) pair.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call model.load() first.")

        # Ensure image token is present
        if "<image>" not in prompt:
            prompt = f"<image>\n{prompt}"

        # LLaVA-1.6 uses standard chat templates
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt.replace("<image>", "").strip()},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs = self.processor(
            text=text_prompt,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = outputs.sequences[0][input_len:]
        response_text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()

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
        token_probs = []
        for step_idx, (score, token_id) in enumerate(zip(scores, generated_ids)):
            prob_dist = torch.softmax(score[0], dim=-1)
            token_prob = prob_dist[token_id].item()
            token_str = self.processor.decode([token_id])
            token_probs.append((token_str, token_prob))
        return token_probs
