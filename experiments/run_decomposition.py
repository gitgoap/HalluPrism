"""
Run Uncertainty Decomposition on a dataset.

This is the Phase 2 experiment script. For each sample it:
  1. Runs greedy inference (same as baseline) to get original answer + confidence
  2. Computes L (Language-Prior): blank image probe
  3. Computes V (Visual): 4 image perturbations
  4. Computes A (Alignment): CoT + negation probes
  5. Saves all scores to a JSONL file

Usage:
    python -m experiments.run_decomposition \
        --dataset pope \
        --model llava_mistral \
        --model_path /home/models/llava-v1.6-mistral-7b-hf \
        --cuda_device cuda:0 \
        --data_root data/ \
        --output results/decomposition/llava_mistral/pope.jsonl

    # Smoke test (5 samples):
    python -m experiments.run_decomposition \
        --dataset pope \
        --model llava_mistral \
        --model_path /home/models/llava-v1.6-mistral-7b-hf \
        --cuda_device cuda:0 \
        --data_root data/ \
        --output results/decomposition/llava_mistral/test_pope.jsonl \
        --max_samples 5
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run uncertainty decomposition (V, L, A)."
    )
    parser.add_argument("--dataset", required=True,
                        choices=["hallusionbench", "pope", "vsr", "vizwiz"],
                        help="Dataset to evaluate on.")
    parser.add_argument("--model", required=True,
                        choices=["llava_mistral", "llava_vicuna", "qwen_vl", "gemma_vl"],
                        help="Model to use.")
    parser.add_argument("--model_path", default=None,
                        help="Path to local model directory.")
    parser.add_argument("--cuda_device", default=None,
                        help="GPU to use, e.g. cuda:0")
    parser.add_argument("--data_root", default="data/",
                        help="Root directory containing dataset folders.")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file path.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples for debugging.")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Start index for processing chunks.")
    parser.add_argument("--end_index", type=int, default=None,
                        help="End index (exclusive) for processing chunks.")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Use 4-bit quantization.")
    parser.add_argument("--skip_visual", action="store_true",
                        help="Skip V (visual perturbation) to save time.")
    parser.add_argument("--skip_alignment", action="store_true",
                        help="Skip A (alignment probes) to save time.")
    return parser.parse_args()


def create_model(model_name, model_path=None, cuda_device=None, load_in_4bit=False):
    """Instantiate the specified model (same factory as run_baseline)."""
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
    """Load dataset (same loaders as run_baseline)."""
    root = Path(data_root)

    if dataset_name == "hallusionbench":
        from src.data.loaders.hallusionbench import HallusionBenchLoader
        return HallusionBenchLoader(
            data_root=str(root / "HallusionBench"), max_samples=max_samples
        ).load()
    elif dataset_name == "pope":
        from src.data.loaders.pope import POPELoader
        return POPELoader(
            data_root=str(root / "POPE"), max_samples=max_samples
        ).load()
    elif dataset_name == "vsr":
        from src.data.loaders.vsr import VSRLoader
        return VSRLoader(
            data_root=str(root / "VSR"), split="dev", max_samples=max_samples
        ).load()
    elif dataset_name == "vizwiz":
        from src.data.loaders.vizwiz import VizWizLoader
        return VizWizLoader(
            data_root=str(root / "VizWiz"), split="val", max_samples=max_samples
        ).load()
    raise ValueError(f"Unknown dataset: {dataset_name}")


def process_sample(
    sample: UnifiedSample,
    model: BaseModel,
    blank_image: Image.Image,
    skip_visual: bool = False,
    skip_alignment: bool = False,
) -> dict:
    """
    Run full uncertainty decomposition on one sample.

    Per sample this does:
      - 1 forward pass: original answer (greedy)
      - 1 forward pass: L (blank image)
      - 4 forward passes: V (perturbations) [unless skipped]
      - 2 forward passes: A (CoT + negation) [unless skipped]
      = 7-8 total forward passes per sample
    """
    # Skip samples without images
    if sample.image_path is None:
        result = sample.to_dict()
        result["skipped"] = True
        result["model_prediction"] = "[skipped_no_image]"
        return result

    # Load image
    try:
        image = Image.open(sample.image_path).convert("RGB")
    except Exception as e:
        logger.warning(f"Could not load image {sample.image_path}: {e}")
        result = sample.to_dict()
        result["model_prediction"] = "[image_load_error]"
        result["error"] = str(e)
        return result

    # Get prompt
    prompt = get_prompt(sample.dataset_name, sample.text_input)

    # --- Step 0: Original greedy inference ---
    response_text, metadata = model.generate(image, prompt)
    original_confidence = compute_scalar_confidence(metadata)
    original_prediction = normalize_answer(response_text)
    correct = is_correct(original_prediction, sample.gold_answer, sample.task_type)

    result = sample.to_dict()
    result["model_prediction"] = original_prediction
    result["raw_response"] = response_text
    result["scalar_confidence"] = original_confidence
    result["is_correct"] = correct

    # --- Step 1: Language-Prior (L) ---
    lp_result = compute_language_prior(
        model=model,
        prompt=prompt,
        original_prediction=original_prediction,
        original_confidence=original_confidence,
        blank_image=blank_image,
    )
    result.update(lp_result)

    # --- Step 2: Visual Uncertainty (V) ---
    if not skip_visual:
        v_result = compute_visual_uncertainty(
            model=model,
            image=image,
            prompt=prompt,
            original_prediction=original_prediction,
            original_confidence=original_confidence,
        )
        # Flatten: skip the per-perturbation details in JSONL (too verbose)
        result["v_score"] = v_result["v_score"]
        result["v_confidence_shift"] = v_result["v_confidence_shift"]
        result["v_flip_rate"] = v_result["v_flip_rate"]
        result["v_n_perturbations"] = v_result["v_n_perturbations"]
        # Store per-perturbation as nested dict
        result["v_per_perturbation"] = v_result["v_per_perturbation"]

    # --- Step 3: Alignment Uncertainty (A) ---
    if not skip_alignment:
        a_result = compute_alignment_uncertainty(
            model=model,
            image=image,
            prompt=prompt,
            original_prediction=original_prediction,
            original_confidence=original_confidence,
            task_type=sample.task_type,
        )
        result["a_score"] = a_result["a_score"]
        result["cot_prediction"] = a_result["cot_prediction"]
        result["cot_flipped"] = a_result["cot_flipped"]
        result["cot_confidence"] = a_result["cot_confidence"]
        if "negation_correct_flip" in a_result:
            result["negation_prediction"] = a_result["negation_prediction"]
            result["negation_correct_flip"] = a_result["negation_correct_flip"]

    return result


def main():
    args = parse_args()

    # Setup output paths (same pattern as run_baseline)
    output_path = Path(args.output)
    model_dir = output_path.parent
    log_dir = model_dir / f"log_files_{args.model}"
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / output_path.with_suffix(".log").name
    setup_logging(log_file=str(log_file))
    set_global_seed(args.seed)

    logger.info(f"=== Decomposition Run: dataset={args.dataset}, model={args.model} ===")

    # Load data
    logger.info("Loading dataset...")
    samples = load_dataset(args.dataset, args.data_root, args.max_samples)
    if args.end_index is not None:
        samples = samples[args.start_index:args.end_index]
    else:
        samples = samples[args.start_index:]
    logger.info(f"Loaded {len(samples)} samples from {args.dataset} (indices {args.start_index} to {args.end_index})")

    # Load model
    logger.info(f"Loading model: {args.model}...")
    model = create_model(
        args.model,
        model_path=args.model_path,
        cuda_device=args.cuda_device,
        load_in_4bit=args.load_in_4bit,
    )
    model.load()
    logger.info("Model loaded successfully.")

    # Pre-create blank image (reused for all L computations)
    blank_image = create_blank_image()

    # Compute forward passes needed per sample
    passes_per_sample = 1 + 1  # original + L
    if not args.skip_visual:
        passes_per_sample += 4  # V perturbations
    if not args.skip_alignment:
        passes_per_sample += 2  # A probes
    logger.info(f"Forward passes per sample: {passes_per_sample}")
    logger.info(f"Estimated total forward passes: {passes_per_sample * len(samples)}")

    # Run decomposition
    with ExperimentLogger(str(output_path),
                          experiment_name=f"decomp_{args.dataset}_{args.model}") as exp_log:
        for i, sample in enumerate(samples):
            if i % 50 == 0:
                logger.info(f"Processing sample {i}/{len(samples)}...")

            t0 = time.time()
            result = process_sample(
                sample, model,
                blank_image=blank_image,
                skip_visual=args.skip_visual,
                skip_alignment=args.skip_alignment,
            )
            result["inference_time_s"] = time.time() - t0
            exp_log.log_sample(result)

        # Summary stats
        all_results = exp_log._results if hasattr(exp_log, "_results") else []
        summary = {
            "dataset": args.dataset,
            "model": args.model,
            "n_samples": len(samples),
            "passes_per_sample": passes_per_sample,
        }
        exp_log.log_summary(summary)

    logger.info(f"Decomposition complete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
