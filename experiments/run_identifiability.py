"""
Identifiability Study (RQ2) — Week 5

This script proves that our 3 uncertainty metrics (V, L, A) are
measuring what they claim to measure. It does this by running
controlled interventions that should selectively spike one component.

The 3 Interventions (all done in-memory, original data untouched):
--------------------------------------------------------------------
1. VISUAL RUINED:
   - Every image is resized to 10x10 px then back to original size.
   - This destroys all visual information but image still exists.
   - Expected: V-score spikes, L and A stay roughly the same.

2. LANGUAGE RUINED:
   - The real question is replaced with a semantically unrelated one.
   - Model cannot use language priors because the question is random.
   - Expected: L-score spikes (model is forced to use the image).

3. ALIGNMENT RUINED:
   - Spatial/relational words are swapped (left<->right, above<->below).
   - The question now contradicts the image content deliberately.
   - Expected: A-score spikes, V and L stay roughly the same.

Output:
   - results/identifiability/<intervention>/<model>/<dataset>.jsonl
   - Compare average V, L, A scores against clean decomposition results
     to build the Identifiability Matrix (diagonal should dominate).

Usage:
    python -m experiments.run_identifiability \
        --dataset pope \
        --model llava_mistral \
        --model_path /home/models/llava-v1.6-mistral-7b-hf \
        --cuda_device cuda:0 \
        --intervention visual_ruined \
        --data_root data/ \
        --output results/identifiability/visual_ruined/llava_mistral/pope.jsonl

    # Smoke test (5 samples):
    python -m experiments.run_identifiability \
        --dataset pope \
        --model llava_mistral \
        --model_path /home/models/llava-v1.6-mistral-7b-hf \
        --cuda_device cuda:0 \
        --intervention visual_ruined \
        --data_root data/ \
        --output results/identifiability/visual_ruined/llava_mistral/test_pope.jsonl \
        --max_samples 5
"""

import argparse
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.schema import UnifiedSample
from src.models.base_model import BaseModel
from src.prompts import get_prompt
from src.baselines.scalar_confidence import compute_scalar_confidence
from src.evaluation.answer_parser import normalize_answer, is_correct
from src.uncertainty.language_prior import compute_language_prior, create_blank_image
from src.uncertainty.visual import compute_visual_uncertainty
from src.uncertainty.alignment import compute_alignment_uncertainty
from src.utils.logging_utils import setup_logging, ExperimentLogger
from src.utils.seed_control import set_global_seed, GLOBAL_SEED

logger = logging.getLogger(__name__)

INTERVENTIONS = ["visual_ruined", "language_ruined", "alignment_ruined"]

# ======================================================================
# Decoy questions for Language-Ruined intervention
# These are semantically unrelated to typical VQA tasks.
# ======================================================================
DECOY_QUESTIONS = [
    "What is the capital city of France?",
    "How many days are in a leap year?",
    "What is the boiling point of water in Celsius?",
    "Who wrote the play Romeo and Juliet?",
    "What is the chemical symbol for gold?",
    "How many continents are on Earth?",
    "What is the speed of light in a vacuum?",
    "In which year did World War II end?",
    "What is the largest planet in our solar system?",
    "What language is spoken in Brazil?",
]

# ======================================================================
# Intervention Functions
# ======================================================================

def apply_visual_ruined(image: Image.Image) -> Image.Image:
    """
    Resize image to 10x10 px then back to original size.
    Completely destroys visual information while preserving image format.
    Original data is never touched — this operates on a RAM copy.
    """
    original_size = image.size
    destroyed = image.resize((10, 10), Image.NEAREST)
    return destroyed.resize(original_size, Image.NEAREST)


def apply_language_ruined(original_prompt: str, seed: int = 42) -> str:
    """
    Replace the real question with a semantically unrelated decoy.
    The decoy is chosen deterministically using a seed so results
    are reproducible. Original prompt is not modified.
    """
    rng = random.Random(seed)
    decoy_question = rng.choice(DECOY_QUESTIONS)

    # Keep only the answer constraint from the original prompt
    # (e.g., "Answer with only 'Yes' or 'No'.")
    constraint = ""
    for line in original_prompt.split("\n"):
        if "answer with" in line.lower() or "say only" in line.lower():
            constraint = line.strip()
            break

    if constraint:
        return f"{decoy_question}\n{constraint}"
    return decoy_question


def apply_alignment_ruined(original_prompt: str) -> str:
    """
    Swap spatial/relational words to create prompt-image misalignment.
    Original prompt is not modified.
    """
    ruined = original_prompt

    # Swap left <-> right
    ruined = re.sub(r"\bleft\b", "__LEFT__", ruined, flags=re.IGNORECASE)
    ruined = re.sub(r"\bright\b", "left", ruined, flags=re.IGNORECASE)
    ruined = re.sub(r"__LEFT__", "right", ruined, flags=re.IGNORECASE)

    # Swap above <-> below
    ruined = re.sub(r"\babove\b", "__ABOVE__", ruined, flags=re.IGNORECASE)
    ruined = re.sub(r"\bbelow\b", "above", ruined, flags=re.IGNORECASE)
    ruined = re.sub(r"__ABOVE__", "below", ruined, flags=re.IGNORECASE)

    # Swap in front of <-> behind
    ruined = re.sub(r"\bin front of\b", "__INFRONT__", ruined, flags=re.IGNORECASE)
    ruined = re.sub(r"\bbehind\b", "in front of", ruined, flags=re.IGNORECASE)
    ruined = re.sub(r"__INFRONT__", "behind", ruined, flags=re.IGNORECASE)

    # Swap on top of <-> under
    ruined = re.sub(r"\bon top of\b", "__ONTOP__", ruined, flags=re.IGNORECASE)
    ruined = re.sub(r"\bunder\b", "on top of", ruined, flags=re.IGNORECASE)
    ruined = re.sub(r"__ONTOP__", "under", ruined, flags=re.IGNORECASE)

    return ruined


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run identifiability study (RQ2) with controlled interventions."
    )
    parser.add_argument("--dataset", required=True,
                        choices=["hallusionbench", "pope", "vsr", "vizwiz"])
    parser.add_argument("--model", required=True,
                        choices=["llava_mistral", "llava_vicuna", "qwen_vl", "gemma_vl"])
    parser.add_argument("--intervention", required=True,
                        choices=INTERVENTIONS,
                        help="Which intervention to apply.")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--cuda_device", default=None)
    parser.add_argument("--data_root", default="data/")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--load_in_4bit", action="store_true")
    return parser.parse_args()


# ======================================================================
# Model + Dataset Loaders (identical to run_decomposition.py)
# ======================================================================

def create_model(model_name, model_path=None, cuda_device=None, load_in_4bit=False):
    if model_name == "llava_mistral":
        from src.models.llava_mistral import LlavaMistralModel
        return LlavaMistralModel(model_path=model_path, cuda_device=cuda_device,
                                 load_in_4bit=load_in_4bit)
    elif model_name == "llava_vicuna":
        from src.models.llava_vicuna import LlavaVicunaModel
        return LlavaVicunaModel(model_path=model_path, cuda_device=cuda_device,
                                load_in_4bit=load_in_4bit)
    elif model_name == "qwen_vl":
        from src.models.qwen_vl import QwenVLModel
        return QwenVLModel(model_path=model_path, cuda_device=cuda_device,
                           load_in_4bit=load_in_4bit)
    elif model_name == "gemma_vl":
        from src.models.gemma_vl import GemmaVLModel
        return GemmaVLModel(model_path=model_path, cuda_device=cuda_device,
                            load_in_4bit=load_in_4bit)
    raise ValueError(f"Unknown model: {model_name}")


def load_dataset(dataset_name, data_root, max_samples=None):
    root = Path(data_root)
    if dataset_name == "hallusionbench":
        from src.data.loaders.hallusionbench import HallusionBenchLoader
        return HallusionBenchLoader(
            data_root=str(root / "HallusionBench"), max_samples=max_samples).load()
    elif dataset_name == "pope":
        from src.data.loaders.pope import POPELoader
        return POPELoader(data_root=str(root / "POPE"), max_samples=max_samples).load()
    elif dataset_name == "vsr":
        from src.data.loaders.vsr import VSRLoader
        return VSRLoader(data_root=str(root / "VSR"), split="dev",
                         max_samples=max_samples).load()
    elif dataset_name == "vizwiz":
        from src.data.loaders.vizwiz import VizWizLoader
        return VizWizLoader(data_root=str(root / "VizWiz"), split="val",
                            max_samples=max_samples).load()
    raise ValueError(f"Unknown dataset: {dataset_name}")


# ======================================================================
# Per-sample processing
# ======================================================================

def process_sample(
    sample: UnifiedSample,
    model: BaseModel,
    intervention: str,
    blank_image: Image.Image,
    seed: int = 42,
) -> dict:
    """
    Run decomposition on one sample with the specified intervention applied.

    The intervention mutates either the image (visual_ruined) or the
    prompt (language_ruined / alignment_ruined) in memory only.
    Original sample data is never written back.
    """
    if sample.image_path is None:
        result = sample.to_dict()
        result["skipped"] = True
        result["model_prediction"] = "[skipped_no_image]"
        result["intervention"] = intervention
        return result

    # Load original image from disk
    try:
        original_image = Image.open(sample.image_path).convert("RGB")
    except Exception as e:
        logger.warning(f"Could not load image {sample.image_path}: {e}")
        result = sample.to_dict()
        result["model_prediction"] = "[image_load_error]"
        result["error"] = str(e)
        result["intervention"] = intervention
        return result

    # Get original prompt
    original_prompt = get_prompt(sample.dataset_name, sample.text_input)

    # === Apply intervention (in memory only) ===
    if intervention == "visual_ruined":
        inference_image = apply_visual_ruined(original_image)
        inference_prompt = original_prompt
    elif intervention == "language_ruined":
        inference_image = original_image
        inference_prompt = apply_language_ruined(original_prompt, seed=seed)
    elif intervention == "alignment_ruined":
        inference_image = original_image
        inference_prompt = apply_alignment_ruined(original_prompt)
    else:
        raise ValueError(f"Unknown intervention: {intervention}")

    # --- Original greedy inference (on intervened input) ---
    response_text, metadata = model.generate(inference_image, inference_prompt)
    original_confidence = compute_scalar_confidence(metadata)
    original_prediction = normalize_answer(response_text)

    # Correctness still measured against gold answer
    correct = is_correct(original_prediction, sample.gold_answer, sample.task_type)

    result = sample.to_dict()
    result["intervention"] = intervention
    result["intervened_prompt"] = inference_prompt
    result["model_prediction"] = original_prediction
    result["raw_response"] = response_text
    result["scalar_confidence"] = original_confidence
    result["is_correct"] = correct

    # --- L score (blank image probe) ---
    lp_result = compute_language_prior(
        model=model,
        prompt=inference_prompt,
        original_prediction=original_prediction,
        original_confidence=original_confidence,
        blank_image=blank_image,
    )
    result.update(lp_result)

    # --- V score (visual perturbations) ---
    v_result = compute_visual_uncertainty(
        model=model,
        image=inference_image,
        prompt=inference_prompt,
        original_prediction=original_prediction,
        original_confidence=original_confidence,
    )
    result["v_score"] = v_result["v_score"]
    result["v_confidence_shift"] = v_result["v_confidence_shift"]
    result["v_flip_rate"] = v_result["v_flip_rate"]

    # --- A score (alignment probes) ---
    a_result = compute_alignment_uncertainty(
        model=model,
        image=inference_image,
        prompt=inference_prompt,
        original_prediction=original_prediction,
        original_confidence=original_confidence,
        task_type=sample.task_type,
    )
    result["a_score"] = a_result["a_score"]
    result["cot_prediction"] = a_result["cot_prediction"]
    result["cot_flipped"] = a_result["cot_flipped"]
    if "negation_correct_flip" in a_result:
        result["negation_correct_flip"] = a_result["negation_correct_flip"]

    return result


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()

    output_path = Path(args.output)
    model_dir = output_path.parent
    log_dir = model_dir / f"log_files_{args.model}"
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / output_path.with_suffix(".log").name
    setup_logging(log_file=str(log_file))
    set_global_seed(args.seed)

    logger.info(
        f"=== Identifiability Run: dataset={args.dataset}, "
        f"model={args.model}, intervention={args.intervention} ==="
    )

    samples = load_dataset(args.dataset, args.data_root, args.max_samples)
    logger.info(f"Loaded {len(samples)} samples.")

    model = create_model(args.model, args.model_path, args.cuda_device, args.load_in_4bit)
    model.load()
    logger.info("Model loaded.")

    blank_image = create_blank_image()

    with ExperimentLogger(
        str(output_path),
        experiment_name=f"identifiability_{args.intervention}_{args.dataset}_{args.model}"
    ) as exp_log:
        for i, sample in enumerate(samples):
            if i % 50 == 0:
                logger.info(f"Processing sample {i}/{len(samples)}...")

            t0 = time.time()
            result = process_sample(
                sample, model,
                intervention=args.intervention,
                blank_image=blank_image,
                seed=args.seed + i,       # vary seed per sample for language_ruined
            )
            result["inference_time_s"] = time.time() - t0
            exp_log.log_sample(result)

        exp_log.log_summary({
            "dataset": args.dataset,
            "model": args.model,
            "intervention": args.intervention,
            "n_samples": len(samples),
        })

    logger.info(f"Done. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
