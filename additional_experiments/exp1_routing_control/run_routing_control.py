import argparse
import json
import logging
import sys
import time
import random
from pathlib import Path

from PIL import Image, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.base_model import BaseModel
from src.baselines.scalar_confidence import compute_scalar_confidence
from src.evaluation.answer_parser import normalize_answer, is_correct
from src.utils.logging_utils import setup_logging, ExperimentLogger
from src.utils.seed_control import set_global_seed, GLOBAL_SEED

from experiments.run_matched_intervention import (
    apply_visual_intervention,
    apply_language_prior_intervention,
    apply_alignment_intervention,
    apply_generic_intervention,
    get_dominant_source,
    create_model,
    load_decomp_results
)
from src.prompts import get_prompt

logger = logging.getLogger(__name__)

def apply_specific_intervention(image: Image.Image, original_prompt: str, int_type: str):
    if int_type == "visual":
        return apply_visual_intervention(image), original_prompt
    elif int_type == "language_prior":
        return image, apply_language_prior_intervention(original_prompt)
    elif int_type == "alignment":
        return image, apply_alignment_intervention(original_prompt)
    else:
        return image, original_prompt

def process_sample_control(decomp_result: dict, model: BaseModel, original_prompt: str, rng: random.Random) -> dict:
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
    
    dominant = get_dominant_source(decomp_result)
    
    # 1. Matched
    matched_img, matched_pr = apply_specific_intervention(image, original_prompt, dominant)
    resp, meta = model.generate(matched_img, matched_pr)
    matched_correct = is_correct(normalize_answer(resp), gold, task_type)
    
    # 2. Permuted/Cyclic
    cyclic_map = {"visual": "language_prior", "language_prior": "alignment", "alignment": "visual"}
    permuted = cyclic_map.get(dominant, "visual")
    permuted_img, permuted_pr = apply_specific_intervention(image, original_prompt, permuted)
    resp, meta = model.generate(permuted_img, permuted_pr)
    permuted_correct = is_correct(normalize_answer(resp), gold, task_type)
    
    # 3. Random
    random_int = rng.choice(["visual", "language_prior", "alignment"])
    random_img, random_pr = apply_specific_intervention(image, original_prompt, random_int)
    resp, meta = model.generate(random_img, random_pr)
    random_correct = is_correct(normalize_answer(resp), gold, task_type)
    
    # 4. Generic
    generic_pr = apply_generic_intervention(original_prompt)
    resp, meta = model.generate(image, generic_pr)
    generic_correct = is_correct(normalize_answer(resp), gold, task_type)
    
    return {
        "sample_id": decomp_result.get("sample_id"),
        "dominant_source": dominant,
        "original_correct": original_correct,
        "matched_correct": matched_correct,
        "permuted_correct": permuted_correct,
        "random_correct": random_correct,
        "generic_correct": generic_correct,
        
        "matched_fixed": (not original_correct) and matched_correct,
        "matched_broke": original_correct and (not matched_correct),
        "permuted_fixed": (not original_correct) and permuted_correct,
        "permuted_broke": original_correct and (not permuted_correct),
        "random_fixed": (not original_correct) and random_correct,
        "random_broke": original_correct and (not random_correct),
        "generic_fixed": (not original_correct) and generic_correct,
        "generic_broke": original_correct and (not generic_correct),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomp_results", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--cuda_device", default=None)
    parser.add_argument("--data_root", default="data/")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging()
    set_global_seed(args.seed)
    rng = random.Random(args.seed)

    decomp_results = load_decomp_results(args.decomp_results)
    if args.max_samples:
        decomp_results = {k: v for i, (k, v) in enumerate(decomp_results.items()) if i < args.max_samples}
        
    model = create_model(args.model, args.model_path, args.cuda_device, args.load_in_4bit)
    model.load()

    with ExperimentLogger(str(output_path), experiment_name=f"routing_control_{args.model}") as exp_log:
        sample_items = list(decomp_results.items())
        if args.end_index is not None:
            sample_items = sample_items[args.start_index:args.end_index]
        else:
            sample_items = sample_items[args.start_index:]
            
        for i, (sample_id, decomp) in enumerate(sample_items):
            original_prompt = get_prompt(decomp.get("dataset_name"), decomp.get("text_input"))
            result = process_sample_control(decomp, model, original_prompt, rng)
            exp_log.log_sample(result)

    print(f"Done. Saved to {output_path}")

if __name__ == "__main__":
    main()
