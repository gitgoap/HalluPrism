"""
RQ5: Matched Intervention Policy (GPU experiment)

Core idea: If we know WHY a model is uncertain, we can apply the RIGHT fix.
  - High V (visual)    → Enhance image (zoom + contrast boost)
  - High L (language)   → Anti-prior prompt ("base answer ONLY on what you see")
  - High A (alignment)  → Chain-of-thought grounding prompt

We compare:
  - Matched policy:  Apply the fix that matches the dominant source
  - Generic policy:  Always apply the same fix regardless of source (CoT for all)
  - No intervention: Original prediction (from decomposition baseline)

If matched > generic, that proves knowing the source is actionable.

Prerequisites:
  - Decomposition results must exist (we read them to know the dominant source)

Usage:
    python -m experiments.run_matched_intervention \
        --decomp_results results/decomposition/llava_mistral/pope.jsonl \
        --model llava_mistral \
        --model_path /home/models/llava-v1.6-mistral-7b-hf \
        --cuda_device cuda:0 \
        --data_root data/ \
        --output results/intervention/llava_mistral/pope.jsonl

    # Smoke test:
    python -m experiments.run_matched_intervention \
        --decomp_results results/decomposition/llava_mistral/pope.jsonl \
        --model llava_mistral \
        --model_path /home/models/llava-v1.6-mistral-7b-hf \
        --cuda_device cuda:0 \
        --data_root data/ \
        --output results/intervention/llava_mistral/test_pope.jsonl \
        --max_samples 5
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from PIL import Image, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.base_model import BaseModel
from src.baselines.scalar_confidence import compute_scalar_confidence
from src.evaluation.answer_parser import normalize_answer, is_correct
from src.utils.logging_utils import setup_logging, ExperimentLogger
from src.utils.seed_control import set_global_seed, GLOBAL_SEED

logger = logging.getLogger(__name__)


# ======================================================================
# Intervention Functions
# ======================================================================

def apply_visual_intervention(image: Image.Image) -> Image.Image:
    """
    Fix for high visual uncertainty:
    Upscale resolution (1.5x) + Color Saturation boost (1.1x) + Sharpness boost (2.5x).
    Forces model to use more tokens and clarifies object boundaries without deleting edge pixels.
    """
    w, h = image.size

    # 1. Upscale resolution by 1.5x (using LANCZOS for high quality)
    upscaled = image.resize((int(w * 1.5), int(h * 1.5)), Image.LANCZOS)

    # 2. Boost color saturation by 1.1x
    enhancer_color = ImageEnhance.Color(upscaled)
    color_boosted = enhancer_color.enhance(1.1)

    # 3. Boost sharpness by 2.5x
    enhancer_sharp = ImageEnhance.Sharpness(color_boosted)
    final_image = enhancer_sharp.enhance(2.5)

    return final_image


def apply_language_prior_intervention(original_prompt: str) -> str:
    """
    Fix for high language-prior uncertainty:
    Prepend an anti-prior instruction that forces image-based reasoning.
    """
    return (
        "Look at the image very carefully before answering. "
        "Do NOT guess based on common sense or what is usually true. "
        "Base your answer ONLY on what you can actually see in this "
        "specific image.\n\n"
        f"{original_prompt}"
    )


def apply_alignment_intervention(original_prompt: str) -> str:
    """
    Fix for high alignment uncertainty:
    Force step-by-step reasoning about visual content before answering.
    """
    return (
        "Look at the image carefully. First, describe what you see "
        "in the image in one sentence. Then answer the following "
        "question based on your description.\n\n"
        f"{original_prompt}"
    )


def apply_generic_intervention(original_prompt: str) -> str:
    """
    Generic fix (applied to ALL samples regardless of source):
    Simple chain-of-thought prompt. This is the baseline we compare against.
    """
    return (
        "Think step by step about the image and question before "
        "answering.\n\n"
        f"{original_prompt}"
    )


# ======================================================================
# Dominant Source Detection
# ======================================================================

def get_dominant_source(decomp_result: dict) -> str:
    """Determine the dominant uncertainty source from decomposition scores."""
    v = decomp_result.get("v_score", 0.0)
    lp = decomp_result.get("lp_score", 0.0)
    a = decomp_result.get("a_score", 0.0)

    scores = {"visual": v, "language_prior": lp, "alignment": a}
    return max(scores, key=scores.get)


# ======================================================================
# CLI + Model/Data Loaders
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run matched intervention policy (RQ5)."
    )
    parser.add_argument("--decomp_results", required=True,
                        help="Path to decomposition JSONL file (to read V/L/A scores)")
    parser.add_argument("--model", required=True,
                        choices=["llava_mistral", "llava_vicuna", "qwen_vl", "gemma_vl"])
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--cuda_device", default=None)
    parser.add_argument("--data_root", default="data/")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--load_in_4bit", action="store_true")
    return parser.parse_args()


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


def load_decomp_results(decomp_path: str) -> dict:
    """Load decomposition results indexed by sample_id."""
    results = {}
    with open(decomp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = r.get("sample_id")
            if sid and not r.get("skipped"):
                results[sid] = r
    return results


# ======================================================================
# Per-sample Processing
# ======================================================================

def process_sample(
    decomp_result: dict,
    model: BaseModel,
    original_prompt: str,
) -> dict:
    """
    Run matched + generic interventions on one sample.

    Per sample: 2 forward passes (matched intervention + generic intervention).
    """
    image_path = decomp_result.get("image_path")
    if not image_path:
        return {"skipped": True, "reason": "no_image_path"}

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"skipped": True, "reason": f"image_load_error: {e}"}

    gold = decomp_result.get("gold_answer", "")
    task_type = decomp_result.get("task_type", "yes_no")
    original_pred = decomp_result.get("model_prediction", "")
    original_correct = decomp_result.get("is_correct", False)
    original_conf = decomp_result.get("scalar_confidence", 0.0)

    # Determine dominant source
    dominant = get_dominant_source(decomp_result)

    # === Matched Intervention ===
    if dominant == "visual":
        matched_image = apply_visual_intervention(image)
        matched_prompt = original_prompt
        intervention_type = "visual_enhance"
    elif dominant == "language_prior":
        matched_image = image
        matched_prompt = apply_language_prior_intervention(original_prompt)
        intervention_type = "anti_prior_prompt"
    elif dominant == "alignment":
        matched_image = image
        matched_prompt = apply_alignment_intervention(original_prompt)
        intervention_type = "cot_grounding"
    else:
        matched_image = image
        matched_prompt = original_prompt
        intervention_type = "none"

    matched_response, matched_meta = model.generate(matched_image, matched_prompt)
    matched_conf = compute_scalar_confidence(matched_meta)
    matched_pred = normalize_answer(matched_response)
    matched_correct = is_correct(matched_pred, gold, task_type)

    # === Generic Intervention (always CoT) ===
    generic_prompt = apply_generic_intervention(original_prompt)
    generic_response, generic_meta = model.generate(image, generic_prompt)
    generic_conf = compute_scalar_confidence(generic_meta)
    generic_pred = normalize_answer(generic_response)
    generic_correct = is_correct(generic_pred, gold, task_type)

    return {
        "sample_id": decomp_result.get("sample_id"),
        "dataset_name": decomp_result.get("dataset_name"),
        "gold_answer": gold,
        "failure_family": decomp_result.get("failure_family"),
        "dominant_source": dominant,
        "v_score": decomp_result.get("v_score"),
        "lp_score": decomp_result.get("lp_score"),
        "a_score": decomp_result.get("a_score"),
        # Original (no intervention)
        "original_prediction": original_pred,
        "original_correct": original_correct,
        "original_confidence": original_conf,
        # Matched intervention
        "matched_intervention_type": intervention_type,
        "matched_prediction": matched_pred,
        "matched_correct": matched_correct,
        "matched_confidence": matched_conf,
        "matched_raw_response": matched_response,
        # Generic intervention
        "generic_prediction": generic_pred,
        "generic_correct": generic_correct,
        "generic_confidence": generic_conf,
        "generic_raw_response": generic_response,
        # Did intervention fix the error?
        "matched_fixed": (not original_correct) and matched_correct,
        "generic_fixed": (not original_correct) and generic_correct,
        "matched_broke": original_correct and (not matched_correct),
        "generic_broke": original_correct and (not generic_correct),
    }


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

    logger.info(f"=== Matched Intervention Run: model={args.model} ===")

    # Load decomposition results
    logger.info(f"Loading decomposition results from {args.decomp_results}")
    decomp_results = load_decomp_results(args.decomp_results)
    logger.info(f"Loaded {len(decomp_results)} decomposition results")

    if args.max_samples:
        sample_ids = list(decomp_results.keys())[:args.max_samples]
        decomp_results = {sid: decomp_results[sid] for sid in sample_ids}
        logger.info(f"Limited to {len(decomp_results)} samples")

    # Load model
    logger.info(f"Loading model: {args.model}")
    model = create_model(args.model, args.model_path, args.cuda_device, args.load_in_4bit)
    model.load()
    logger.info("Model loaded.")

    # Import prompt generator
    from src.prompts import get_prompt

    # Run interventions
    with ExperimentLogger(str(output_path),
                          experiment_name=f"intervention_{args.model}") as exp_log:
        sample_items = list(decomp_results.items())
        if args.end_index is not None:
            sample_items = sample_items[args.start_index:args.end_index]
        else:
            sample_items = sample_items[args.start_index:]
            
        for i, (sample_id, decomp) in enumerate(sample_items):
            if i % 50 == 0:
                logger.info(f"Processing sample {i}/{len(sample_items)}...")

            # Reconstruct the original prompt
            dataset_name = decomp.get("dataset_name", "")
            text_input = decomp.get("text_input", "")
            try:
                original_prompt = get_prompt(dataset_name, text_input)
            except ValueError:
                logger.warning(f"Unknown dataset {dataset_name} for sample {sample_id}")
                continue

            t0 = time.time()
            result = process_sample(decomp, model, original_prompt)
            result["inference_time_s"] = time.time() - t0
            exp_log.log_sample(result)

        # Summary
        exp_log.log_summary({
            "model": args.model,
            "decomp_source": args.decomp_results,
            "n_samples": len(decomp_results),
        })

    logger.info(f"Done. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
