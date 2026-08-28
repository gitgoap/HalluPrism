"""
RQ4: Source-Aware Abstention / Selective Accuracy

Core question: Does knowing V, L, A help the model decide when to
abstain (refuse to answer) better than scalar confidence alone?

What this script does:
  1. Combines V, L, A into a single "source-aware confidence" score
  2. Compares 3 confidence strategies for selective answering:
       - Scalar confidence (baseline)
       - Source-aware confidence (ours)
       - Oracle (perfect abstention — upper bound)
  3. Produces:
       - Risk-coverage curve data (main figure for the paper)
       - Selective accuracy at multiple coverage thresholds
       - AUROC for correctness prediction
       - VizWiz answerability-specific analysis

Usage:
    # Single model:
    python analysis/rq4_abstention.py \
        --results_dir results/decomposition/llava_mistral/

    # All models:
    python analysis/rq4_abstention.py \
        --results_dir results/decomposition/ \
        --all_models

    # Save risk-coverage curve as CSV for plotting:
    python analysis/rq4_abstention.py \
        --results_dir results/decomposition/ \
        --all_models \
        --save_csv results/figures/

    # Quick-run (recommended after all experiments are done):
    #   python analysis/rq4_abstention.py --results_dir results/decomposition/ --all_models
    #
    # Or run everything at once with:
    #   bash analysis/run_all_analysis.sh
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ======================================================================
# Source-Aware Confidence Combiners
# ======================================================================

def combine_source_aware_confidence(
    scalar_conf: float,
    v_score: float,
    lp_score: float,
    a_score: float,
    method: str = "penalty",
) -> float:
    """
    Combine V, L, A scores with scalar confidence into a single
    source-aware confidence score.

    Methods:
        "penalty": Penalize scalar confidence by the max uncertainty source.
            source_aware = scalar_conf * (1 - max(V, L, A))

        "weighted": Weighted combination.
            source_aware = scalar_conf * (1 - 0.3*V - 0.3*L - 0.4*A)

        "geometric": Geometric mean of (1 - each source).
            source_aware = scalar_conf * ((1-V) * (1-L) * (1-A))^(1/3)

    Returns:
        Float in [0, 1]. Higher = more confident.
    """
    if method == "penalty":
        max_uncertainty = max(v_score, lp_score, a_score)
        return scalar_conf * (1.0 - max_uncertainty)

    elif method == "weighted":
        uncertainty = 0.3 * v_score + 0.3 * lp_score + 0.4 * a_score
        return scalar_conf * max(0.0, 1.0 - uncertainty)

    elif method == "geometric":
        product = (1.0 - v_score) * (1.0 - lp_score) * (1.0 - a_score)
        geo_mean = max(product, 0.0) ** (1.0 / 3.0)
        return scalar_conf * geo_mean

    else:
        raise ValueError(f"Unknown combination method: {method}")


# ======================================================================
# Selective Accuracy and Risk-Coverage
# ======================================================================

def compute_selective_accuracy(
    confidences: np.ndarray,
    correct_flags: np.ndarray,
    coverage: float,
) -> dict:
    """
    Compute accuracy on the top-k% most confident predictions.

    Args:
        confidences: confidence scores (higher = more confident)
        correct_flags: boolean array (True = correct prediction)
        coverage: fraction of examples to keep (0.0 to 1.0)

    Returns:
        Dict with selective_accuracy, risk, n_answered, n_abstained
    """
    n = len(confidences)
    n_keep = max(1, int(n * coverage))

    # Sort by confidence descending
    sorted_indices = np.argsort(-confidences)
    kept_indices = sorted_indices[:n_keep]

    correct_count = correct_flags[kept_indices].sum()
    sel_acc = correct_count / n_keep

    return {
        "coverage": coverage,
        "selective_accuracy": float(sel_acc),
        "risk": float(1.0 - sel_acc),
        "n_answered": n_keep,
        "n_abstained": n - n_keep,
        "n_correct": int(correct_count),
    }


def compute_risk_coverage_curve(
    confidences: np.ndarray,
    correct_flags: np.ndarray,
    n_points: int = 20,
) -> List[dict]:
    """Compute risk-coverage curve data points."""
    points = []
    for i in range(1, n_points + 1):
        cov = i / n_points
        point = compute_selective_accuracy(confidences, correct_flags, cov)
        points.append(point)
    return points


def compute_auroc(confidences: np.ndarray, correct_flags: np.ndarray) -> float:
    """
    AUROC: how well does confidence separate correct from incorrect?
    Higher = better calibrated confidence.
    """
    n = len(confidences)
    if n == 0:
        return 0.5

    n_pos = correct_flags.sum()
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort by confidence descending
    sorted_indices = np.argsort(-confidences)
    sorted_correct = correct_flags[sorted_indices]

    # Compute AUROC via rank-sum
    tp = 0
    auc = 0.0
    for i in range(n):
        if sorted_correct[i]:
            tp += 1
        else:
            auc += tp  # number of positives ranked above this negative

    return auc / (n_pos * n_neg)


def compute_auprc(
    confidences: np.ndarray,
    correct_flags: np.ndarray,
    n_points: int = 20,
) -> float:
    """Area Under the Precision-Risk Coverage curve (higher = better)."""
    curve = compute_risk_coverage_curve(confidences, correct_flags, n_points)
    # Integrate using trapezoidal rule
    coverages = [p["coverage"] for p in curve]
    accuracies = [p["selective_accuracy"] for p in curve]

    auprc = 0.0
    for i in range(1, len(coverages)):
        auprc += (coverages[i] - coverages[i - 1]) * (accuracies[i] + accuracies[i - 1]) / 2

    return float(auprc)


# ======================================================================
# Data Loading
# ======================================================================

def load_decomposition_results(results_dir: str, model_name: str = None) -> List[dict]:
    """Load decomposition JSONL files with V/L/A scores."""
    results_path = Path(results_dir)
    all_results = []

    if model_name:
        dirs = [results_path / model_name]
    else:
        dirs = [d for d in results_path.iterdir()
                if d.is_dir() and not d.name.startswith("log_files")]

    for model_dir in dirs:
        for jf in sorted(model_dir.glob("*.jsonl")):
            if jf.name.startswith("test_"):
                continue
            with open(jf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if (r.get("skipped") or
                        r.get("v_score") is None or
                        r.get("lp_score") is None or
                        r.get("a_score") is None or
                        r.get("scalar_confidence") is None):
                        continue
                    r["_model"] = model_dir.name
                    r["_dataset"] = jf.stem
                    all_results.append(r)

    return all_results


# ======================================================================
# Main Analysis
# ======================================================================

def analyze_abstention(
    results: List[dict],
    combination_methods: List[str] = None,
) -> dict:
    """
    Run the full abstention analysis comparing scalar vs source-aware confidence.
    """
    if combination_methods is None:
        combination_methods = ["penalty", "weighted", "geometric"]

    # Extract arrays
    scalar_conf = np.array([r["scalar_confidence"] for r in results])
    correct_flags = np.array([r.get("is_correct", False) for r in results])
    v_scores = np.array([r.get("v_score", 0.0) for r in results])
    lp_scores = np.array([r.get("lp_score", 0.0) for r in results])
    a_scores = np.array([r.get("a_score", 0.0) for r in results])

    # Compute source-aware confidence for each method
    source_aware_confs = {}
    for method in combination_methods:
        sa_conf = np.array([
            combine_source_aware_confidence(
                sc, v, lp, a, method=method
            )
            for sc, v, lp, a in zip(scalar_conf, v_scores, lp_scores, a_scores)
        ])
        source_aware_confs[method] = sa_conf

    # Compute metrics for each confidence type
    analysis = {
        "n_samples": len(results),
        "accuracy": float(correct_flags.mean()),
        "strategies": {},
    }

    # Strategy 1: Scalar confidence (baseline)
    scalar_curve = compute_risk_coverage_curve(scalar_conf, correct_flags)
    scalar_auroc = compute_auroc(scalar_conf, correct_flags)
    scalar_auprc = compute_auprc(scalar_conf, correct_flags)
    analysis["strategies"]["Scalar Confidence"] = {
        "auroc": scalar_auroc,
        "auprc": scalar_auprc,
        "risk_coverage": scalar_curve,
        "selective_acc_80": compute_selective_accuracy(scalar_conf, correct_flags, 0.8),
        "selective_acc_60": compute_selective_accuracy(scalar_conf, correct_flags, 0.6),
    }

    # Strategy 2+: Source-aware confidence (one per combination method)
    for method, sa_conf in source_aware_confs.items():
        sa_curve = compute_risk_coverage_curve(sa_conf, correct_flags)
        sa_auroc = compute_auroc(sa_conf, correct_flags)
        sa_auprc = compute_auprc(sa_conf, correct_flags)
        label = f"Source-Aware ({method})"
        analysis["strategies"][label] = {
            "auroc": sa_auroc,
            "auprc": sa_auprc,
            "risk_coverage": sa_curve,
            "selective_acc_80": compute_selective_accuracy(sa_conf, correct_flags, 0.8),
            "selective_acc_60": compute_selective_accuracy(sa_conf, correct_flags, 0.6),
        }

    return analysis


def print_analysis(analysis: dict, title: str = ""):
    """Pretty-print abstention analysis results."""
    print(f"\n  {'='*75}")
    print(f"  RQ4: ABSTENTION ANALYSIS{(' — ' + title) if title else ''}")
    print(f"  {'='*75}")
    print(f"  Samples: {analysis['n_samples']}, Accuracy: {analysis['accuracy']:.1%}")

    # AUROC comparison table
    print(f"\n  Confidence Strategy Comparison:")
    print(f"    {'Strategy':<30s} {'AUROC':>7s} {'AUPRC':>7s} {'SelAcc@80%':>11s} {'SelAcc@60%':>11s}")
    print(f"    {'-'*30} {'-'*7} {'-'*7} {'-'*11} {'-'*11}")

    for strategy_name, metrics in analysis["strategies"].items():
        sa80 = metrics["selective_acc_80"]["selective_accuracy"]
        sa60 = metrics["selective_acc_60"]["selective_accuracy"]
        print(f"    {strategy_name:<30s} {metrics['auroc']:>7.3f} {metrics['auprc']:>7.3f} "
              f"{sa80:>10.1%} {sa60:>10.1%}")

    # Highlight best source-aware method
    best_sa = None
    best_auroc = 0
    scalar_auroc = analysis["strategies"]["Scalar Confidence"]["auroc"]

    for name, metrics in analysis["strategies"].items():
        if "Source-Aware" in name and metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            best_sa = name

    if best_sa:
        delta = best_auroc - scalar_auroc
        print(f"\n  Best source-aware method: {best_sa}")
        print(f"  AUROC improvement over scalar: {delta:+.3f} "
              f"({'better' if delta > 0 else 'WORSE'})")

    # Risk-coverage curve summary
    print(f"\n  Risk-Coverage Curve (Scalar vs Best Source-Aware):")
    print(f"    {'Coverage':>9s} {'Scalar Risk':>12s} {'SA Risk':>12s} {'Delta':>8s}")
    print(f"    {'-'*9} {'-'*12} {'-'*12} {'-'*8}")

    scalar_curve = analysis["strategies"]["Scalar Confidence"]["risk_coverage"]
    if best_sa:
        sa_curve = analysis["strategies"][best_sa]["risk_coverage"]
    else:
        sa_curve = scalar_curve

    for sc, sac in zip(scalar_curve, sa_curve):
        delta = sc["risk"] - sac["risk"]
        print(f"    {sc['coverage']:>8.0%} {sc['risk']:>12.3f} {sac['risk']:>12.3f} "
              f"{delta:>+8.3f}")


def save_curves_csv(analysis: dict, output_dir: str, model_name: str):
    """Save risk-coverage curves as CSV for external plotting."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / f"risk_coverage_{model_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["coverage"]
        for strategy_name in analysis["strategies"]:
            header.extend([f"{strategy_name}_risk", f"{strategy_name}_sel_acc"])
        writer.writerow(header)

        # All curves have same coverage points
        n_points = len(next(iter(analysis["strategies"].values()))["risk_coverage"])
        for i in range(n_points):
            row = [analysis["strategies"]["Scalar Confidence"]["risk_coverage"][i]["coverage"]]
            for strategy_name, metrics in analysis["strategies"].items():
                point = metrics["risk_coverage"][i]
                row.extend([point["risk"], point["selective_accuracy"]])
            writer.writerow(row)

    print(f"\n  Risk-coverage CSV saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="RQ4: Abstention analysis.")
    parser.add_argument("--results_dir", required=True,
                        help="Path to decomposition results directory")
    parser.add_argument("--model", default=None,
                        help="Specific model to analyze")
    parser.add_argument("--all_models", action="store_true",
                        help="Analyze all models")
    parser.add_argument("--save_csv", default=None,
                        help="Directory to save risk-coverage CSVs for plotting")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    # Default save location: results/decomposition/analysis/rq4/
    analysis_dir = results_dir / "analysis" / "rq4"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    save_csv_dir = args.save_csv if args.save_csv else str(analysis_dir)

    from analysis.tee_output import TeeOutput
    tee = TeeOutput(str(analysis_dir / "full_report.txt"))
    tee.__enter__()

    if args.all_models:
        model_dirs = sorted([
            d.name for d in results_dir.iterdir()
            if d.is_dir() and not d.name.startswith(("log_files", "analysis"))
        ])
    elif args.model:
        model_dirs = [args.model]
    else:
        print("Error: specify --model or --all_models")
        return

    # Per-model analysis
    for model_name in model_dirs:
        print(f"\n{'#'*75}")
        print(f"  MODEL: {model_name}")
        print(f"{'#'*75}")

        results = load_decomposition_results(str(results_dir), model_name)
        if not results:
            print(f"  No valid results found for {model_name}")
            continue

        print(f"  Loaded {len(results)} samples")

        # Per-dataset analysis
        datasets = sorted(set(r["_dataset"] for r in results))
        for ds in datasets:
            ds_results = [r for r in results if r["_dataset"] == ds]
            analysis = analyze_abstention(ds_results)
            print_analysis(analysis, title=f"{model_name}/{ds}")

            save_curves_csv(analysis, save_csv_dir, f"{model_name}_{ds}")

        # All datasets combined
        if len(datasets) > 1:
            analysis = analyze_abstention(results)
            print_analysis(analysis, title=f"{model_name}/ALL")

            save_curves_csv(analysis, save_csv_dir, f"{model_name}_all")

    # Pooled across all models
    if len(model_dirs) > 1:
        print(f"\n{'#'*75}")
        print(f"  POOLED (all models)")
        print(f"{'#'*75}")

        all_results = load_decomposition_results(str(results_dir))
        if all_results:
            analysis = analyze_abstention(all_results)
            print_analysis(analysis, title="ALL MODELS POOLED")

            save_curves_csv(analysis, save_csv_dir, "pooled_all")

    tee.__exit__(None, None, None)
    print(f"Full report saved -> {analysis_dir / 'full_report.txt'}")


if __name__ == "__main__":
    main()
