"""
Identifiability Matrix Analyzer (RQ2)

Compares average V, L, A scores from the 3 controlled interventions
against the clean decomposition baseline to build the Identifiability Matrix.

The matrix should show diagonal dominance:
  - visual_ruined    → V spikes, L and A stable
  - language_ruined  → L spikes, V and A stable
  - alignment_ruined → A spikes, V and L stable

Usage:
    python analysis/analyze_identifiability.py \
        --clean_dir results/decomposition/ \
        --identifiability_dir results/identifiability/ \
        --model llava_mistral

    # All models at once:
    python analysis/analyze_identifiability.py \
        --clean_dir results/decomposition/ \
        --identifiability_dir results/identifiability/ \
        --all_models
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

INTERVENTIONS = ["visual_ruined", "language_ruined", "alignment_ruined"]


def load_avg_scores(jsonl_path: str) -> dict:
    """Load a JSONL file and compute average V, L, A scores."""
    v_scores, lp_scores, a_scores = [], [], []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("skipped") or r.get("model_prediction") in (
                "[skipped_no_image]", "[image_load_error]"
            ):
                continue

            v_scores.append(r.get("v_score", 0.0))
            lp_scores.append(r.get("lp_score", 0.0))
            a_scores.append(r.get("a_score", 0.0))

    if not v_scores:
        return {"v": 0.0, "lp": 0.0, "a": 0.0, "n": 0}

    return {
        "v": float(np.mean(v_scores)),
        "lp": float(np.mean(lp_scores)),
        "a": float(np.mean(a_scores)),
        "n": len(v_scores),
    }


def build_identifiability_matrix(
    clean_scores: dict,
    intervention_scores: Dict[str, dict],
) -> dict:
    """
    Build the identifiability matrix showing how each intervention
    shifts V, L, A relative to the clean baseline.

    Returns:
        Dict with absolute scores and deltas.
    """
    matrix = {"clean": clean_scores}

    for intervention in INTERVENTIONS:
        if intervention in intervention_scores:
            scores = intervention_scores[intervention]
            matrix[intervention] = {
                "v": scores["v"],
                "lp": scores["lp"],
                "a": scores["a"],
                "n": scores["n"],
                "v_delta": scores["v"] - clean_scores["v"],
                "lp_delta": scores["lp"] - clean_scores["lp"],
                "a_delta": scores["a"] - clean_scores["a"],
            }

    return matrix


def check_diagonal_dominance(matrix: dict) -> dict:
    """
    Check if each intervention primarily affects the intended component.

    Returns pass/fail for each intervention.
    """
    checks = {}

    # visual_ruined should spike V the most
    if "visual_ruined" in matrix:
        m = matrix["visual_ruined"]
        v_delta = m["v_delta"]
        checks["visual_ruined"] = {
            "target_delta": v_delta,
            "other_max_delta": max(abs(m["lp_delta"]), abs(m["a_delta"])),
            "dominant": v_delta > max(abs(m["lp_delta"]), abs(m["a_delta"])),
            "target": "V",
        }

    # language_ruined should spike LP the most
    if "language_ruined" in matrix:
        m = matrix["language_ruined"]
        lp_delta = m["lp_delta"]
        checks["language_ruined"] = {
            "target_delta": lp_delta,
            "other_max_delta": max(abs(m["v_delta"]), abs(m["a_delta"])),
            "dominant": abs(lp_delta) > max(abs(m["v_delta"]), abs(m["a_delta"])),
            "target": "L",
        }

    # alignment_ruined should spike A the most
    if "alignment_ruined" in matrix:
        m = matrix["alignment_ruined"]
        a_delta = m["a_delta"]
        checks["alignment_ruined"] = {
            "target_delta": a_delta,
            "other_max_delta": max(abs(m["v_delta"]), abs(m["lp_delta"])),
            "dominant": abs(a_delta) > max(abs(m["v_delta"]), abs(m["lp_delta"])),
            "target": "A",
        }

    return checks


def print_matrix(matrix: dict, model_name: str, dataset: str):
    """Pretty-print the identifiability matrix."""
    print(f"\n  {'='*65}")
    print(f"  IDENTIFIABILITY MATRIX: {model_name} / {dataset}")
    print(f"  {'='*65}")

    # Absolute scores table
    print(f"\n  Absolute Scores:")
    print(f"    {'Condition':<22s} {'V':>8s} {'L':>8s} {'A':>8s} {'N':>6s}")
    print(f"    {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    for condition in ["clean"] + INTERVENTIONS:
        if condition in matrix:
            m = matrix[condition]
            print(f"    {condition:<22s} {m['v']:>8.4f} {m['lp']:>8.4f} "
                  f"{m['a']:>8.4f} {m['n']:>6d}")

    # Delta table (change from clean)
    print(f"\n  Deltas (change from clean baseline):")
    print(f"    {'Intervention':<22s} {'dV':>8s} {'dL':>8s} {'dA':>8s} {'Target':>8s}")
    print(f"    {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for intervention in INTERVENTIONS:
        if intervention in matrix:
            m = matrix[intervention]
            target_map = {
                "visual_ruined": "V",
                "language_ruined": "L",
                "alignment_ruined": "A",
            }
            target = target_map[intervention]
            print(f"    {intervention:<22s} {m['v_delta']:>+8.4f} {m['lp_delta']:>+8.4f} "
                  f"{m['a_delta']:>+8.4f} {target:>8s}")

    # Diagonal dominance check
    checks = check_diagonal_dominance(matrix)
    print(f"\n  Diagonal Dominance Check:")
    all_pass = True
    for intervention, check in checks.items():
        status = "PASS" if check["dominant"] else "FAIL"
        if not check["dominant"]:
            all_pass = False
        print(f"    {intervention:<22s} target={check['target']} "
              f"delta={check['target_delta']:+.4f} "
              f"max_other={check['other_max_delta']:.4f} "
              f"-> {status}")

    overall = "PASS (all diagonal)" if all_pass else "PARTIAL"
    print(f"\n  Overall Identifiability: {overall}")


def main():
    parser = argparse.ArgumentParser(description="Analyze identifiability results (RQ2).")
    parser.add_argument("--clean_dir", required=True,
                        help="Path to clean decomposition results (e.g. results/decomposition/)")
    parser.add_argument("--identifiability_dir", required=True,
                        help="Path to identifiability results (e.g. results/identifiability/)")
    parser.add_argument("--model", default=None,
                        help="Specific model to analyze (e.g. llava_mistral)")
    parser.add_argument("--all_models", action="store_true",
                        help="Analyze all models")
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    ident_dir = Path(args.identifiability_dir)
    
    analysis_dir = ident_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    from analysis.tee_output import TeeOutput
    tee = TeeOutput(str(analysis_dir / "full_report.txt"))
    tee.__enter__()

    if args.all_models:
        # Discover models from clean decomposition dir
        model_names = sorted([
            d.name for d in clean_dir.iterdir()
            if d.is_dir() and not d.name.startswith("log_files")
        ])
    elif args.model:
        model_names = [args.model]
    else:
        print("Error: specify --model or --all_models")
        return

    for model_name in model_names:
        print(f"\n{'#'*70}")
        print(f"  MODEL: {model_name}")
        print(f"{'#'*70}")

        clean_model_dir = clean_dir / model_name
        if not clean_model_dir.exists():
            print(f"  Clean results not found at {clean_model_dir}")
            continue

        # Process each dataset
        for jsonl_file in sorted(clean_model_dir.glob("*.jsonl")):
            if jsonl_file.name.startswith("test_"):
                continue

            dataset = jsonl_file.stem
            clean_scores = load_avg_scores(str(jsonl_file))

            if clean_scores["n"] == 0:
                print(f"  {dataset}: No clean results found, skipping")
                continue

            # Load intervention scores
            intervention_scores = {}
            for intervention in INTERVENTIONS:
                ident_file = ident_dir / intervention / model_name / f"{dataset}.jsonl"
                if ident_file.exists():
                    intervention_scores[intervention] = load_avg_scores(str(ident_file))
                else:
                    print(f"  {dataset}: Missing {intervention} results at {ident_file}")

            if not intervention_scores:
                print(f"  {dataset}: No intervention results found, skipping")
                continue

            # Build and print matrix
            matrix = build_identifiability_matrix(clean_scores, intervention_scores)
            print_matrix(matrix, model_name, dataset)

    # Summary across all models/datasets
    print(f"\n\n{'='*70}")
    print(f"  Analysis complete. Look for PASS/FAIL in Diagonal Dominance Check.")
    print(f"  PASS = your metric correctly detects the intended intervention.")
    print(f"{'='*70}")

    tee.__exit__(None, None, None)
    print(f"Full report saved -> {analysis_dir / 'full_report.txt'}")


if __name__ == "__main__":
    main()
