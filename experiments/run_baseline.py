"""
Run baseline MLLM inference on all four core datasets.

This is the main experiment script for Week 3. It:
  1. Loads a specified dataset via the unified loaders
  2. Loads the specified MLLM (LLaVA or InstructBLIP)
  3. Runs inference on each sample with the appropriate prompt
  4. Collects scalar confidence, self-reported confidence
  5. Optionally runs sampling disagreement (MC sampling)
  6. Saves all results to a JSONL file

Usage:
    python -m experiments.run_baseline \
        --dataset hallusionbench \
        --model llava_mistral \
        --data_root data/ \
        --output results/baselines/llava_mistral/hallusionbench.jsonl

    python -m experiments.run_baseline \
        --dataset pope \
        --model instructblip \
        --data_root data/ \
        --output results/baselines/pope_instructblip.jsonl \
        --mc_samples 5
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
from src.prompts import get_prompt, self_reported_confidence_prompt
from src.baselines.scalar_confidence import compute_scalar_confidence
from src.baselines.self_reported import extract_answer_and_confidence

from src.evaluation.answer_parser import normalize_answer, is_correct
from src.evaluation.metrics import accuracy, binary_metrics, per_family_accuracy
from src.utils.logging_utils import setup_logging, ExperimentLogger
from src.utils.seed_control import set_global_seed, GLOBAL_SEED


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline MLLM inference.")
    parser.add_argument("--dataset", required=True,
                        choices=["hallusionbench", "pope", "vsr", "vizwiz"],
                        help="Dataset to evaluate on.")
    parser.add_argument("--model", default="llava_mistral",
                        choices=["llava_mistral", "llava_vicuna", "qwen_vl", "gemma_vl"],
                        help="Model to use.")
    parser.add_argument("--model_path", default=None,
                        help="Path to local model directory, e.g. /home/models/llava-hf_llava-1.5-13b-hf. "
                             "If not set, the HuggingFace hub ID is used (requires internet).")
    parser.add_argument("--cuda_device", default=None,
                        help="Specific GPU to use, e.g. cuda:1. If not set, device_map=auto is used.")
    parser.add_argument("--data_root", default="data/",
                        help="Root directory containing dataset folders.")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file path.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples for debugging.")

    parser.add_argument("--self_reported", action="store_true",
                        help="Also run self-reported confidence prompt.")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Use 4-bit quantization (saves VRAM).")
    return parser.parse_args()


def create_model(
    model_name: str,
    model_path: str = None,
    cuda_device: str = None,
    load_in_4bit: bool = False,
) -> BaseModel:
    """Instantiate the specified model."""
    if model_name == "llava_mistral":
        from src.models.llava_mistral import LlavaMistralModel
        return LlavaMistralModel(
            model_path=model_path,
            cuda_device=cuda_device,
            load_in_4bit=load_in_4bit,
        )
    elif model_name == "llava_vicuna":
        from src.models.llava_vicuna import LlavaVicunaModel
        return LlavaVicunaModel(
            model_path=model_path,
            cuda_device=cuda_device,
            load_in_4bit=load_in_4bit,
        )
    elif model_name == "qwen_vl":
        from src.models.qwen_vl import QwenVLModel
        return QwenVLModel(
            model_path=model_path,
            cuda_device=cuda_device,
            load_in_4bit=load_in_4bit,
        )
    elif model_name == "gemma_vl":
        from src.models.gemma_vl import GemmaVLModel
        return GemmaVLModel(
            model_path=model_path,
            cuda_device=cuda_device,
            load_in_4bit=load_in_4bit,
        )

    raise ValueError(f"Unknown model: {model_name}")


def load_dataset(dataset_name: str, data_root: str, max_samples: int = None):
    """Load dataset using unified loaders."""
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
    run_self_reported: bool = False,
) -> dict:
    """
    Run inference on one sample, collect all baseline signals.

    Returns a results dict ready for JSONL logging.
    """
    # Skip text-only samples (no image)
    if sample.image_path is None:
        logger.info(f"Skipping text-only sample {sample.sample_id} (no image)")
        result = sample.to_dict()
        result["model_prediction"] = "[skipped_no_image]"
        result["skipped"] = True
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

    # Primary generation (greedy)
    response_text, metadata = model.generate(image, prompt)
    scalar_conf = compute_scalar_confidence(metadata)

    # Normalize prediction
    prediction = normalize_answer(response_text)
    correct = is_correct(prediction, sample.gold_answer, sample.task_type)

    result = sample.to_dict()
    result["model_prediction"] = prediction
    result["raw_response"] = response_text
    result["scalar_confidence"] = scalar_conf
    result["is_correct"] = correct

    # Self-reported confidence (optional)
    if run_self_reported:
        sr_prompt = self_reported_confidence_prompt(sample.text_input, sample.dataset_name)
        sr_response, sr_meta = model.generate(image, sr_prompt)
        _, sr_conf = extract_answer_and_confidence(sr_response)
        result["self_reported_confidence"] = sr_conf
        result["self_reported_raw"] = sr_response


    return result


def main():
    args = parse_args()

    # Setup output paths
    # Expected convention: results/baselines/<model>/<dataset>.jsonl
    # Logs go to:          results/baselines/<model>/log_files_<model>/<dataset>.log
    output_path = Path(args.output)
    model_dir = output_path.parent          # e.g. results/baselines/llava_mistral/
    log_dir = model_dir / f"log_files_{args.model}"  # e.g. .../log_files_llava_mistral/
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / output_path.with_suffix(".log").name
    setup_logging(log_file=str(log_file))
    set_global_seed(args.seed)

    logger.info(f"=== Baseline Run: dataset={args.dataset}, model={args.model} ===")

    # Load data
    logger.info("Loading dataset...")
    samples = load_dataset(args.dataset, args.data_root, args.max_samples)
    logger.info(f"Loaded {len(samples)} samples from {args.dataset}")

    # Load model
    logger.info(f"Loading model: {args.model} (path={args.model_path}, device={args.cuda_device})...")
    model = create_model(
        args.model,
        model_path=args.model_path,
        cuda_device=args.cuda_device,
        load_in_4bit=args.load_in_4bit,
    )
    model.load()
    logger.info("Model loaded successfully.")

    # Run inference
    all_predictions = []
    all_golds = []
    all_confidences = []
    all_families = []

    with ExperimentLogger(str(output_path), experiment_name=f"baseline_{args.dataset}_{args.model}") as exp_log:
        for i, sample in enumerate(samples):
            if i % 100 == 0:
                logger.info(f"Processing sample {i}/{len(samples)}...")

            t0 = time.time()
            result = process_sample(
                sample, model,
                run_self_reported=args.self_reported,
            )
            result["inference_time_s"] = time.time() - t0
            exp_log.log_sample(result)

            # Collect for summary (use "or" to convert None -> "")
            pred = result.get("model_prediction") or ""
            all_predictions.append(pred)
            all_golds.append(sample.gold_answer)
            all_confidences.append(result.get("scalar_confidence") or 0.0)
            all_families.append(sample.failure_family)

        # Compute summary metrics
        acc = accuracy(all_predictions, all_golds)
        family_acc = per_family_accuracy(all_predictions, all_golds, all_families)

        # Binary metrics for yes/no datasets
        summary = {
            "dataset": args.dataset,
            "model": args.model,
            "n_samples": len(samples),
            "accuracy": acc,
            "per_family_accuracy": family_acc,
        }

        if args.dataset in ("hallusionbench", "pope"):
            bm = binary_metrics(all_predictions, all_golds, positive="yes")
            summary["binary_metrics"] = bm
        elif args.dataset == "vsr":
            bm = binary_metrics(all_predictions, all_golds, positive="true")
            summary["binary_metrics"] = bm

        exp_log.log_summary(summary)
        logger.info(f"Overall accuracy: {acc:.4f}")
        logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
