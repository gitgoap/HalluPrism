"""
Decomposition Results Analyzer

Reads the decomposition JSONL files and produces:
  1. Average V, L, A scores per dataset
  2. Average V, L, A scores per failure family
  3. Correlation matrix between V, L, A (should be low = they measure different things)
  4. Distribution statistics for correct vs incorrect samples
  5. Dominant source assignment breakdown

Usage:
    # Analyze one model:
    python analysis/analyze_decomposition.py \
        --results_dir results/decomposition/llava_mistral/

    # Analyze all models:
    python analysis/analyze_decomposition.py \
        --results_dir results/decomposition/ \
        --all_models

    # Quick-run (recommended after all experiments are done):
    #   python analysis/analyze_decomposition.py --results_dir results/decomposition/ --all_models
    #
    # Or run everything at once with:
    #   bash analysis/run_all_analysis.sh
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_results(jsonl_path: str) -> List[dict]:
    """Load results from a JSONL file, skipping empty/skipped samples."""
    results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # Skip samples without decomposition scores
            if r.get("skipped") or r.get("model_prediction") in (
                "[skipped_no_image]", "[image_load_error]"
            ):
                continue
            results.append(r)
    return results


def get_scores(result: dict) -> dict:
    """Extract V, L, A scores from a result dict."""
    return {
        "v_score": result.get("v_score", 0.0),
        "lp_score": result.get("lp_score", 0.0),
        "a_score": result.get("a_score", 0.0),
    }


def dominant_source(result: dict) -> str:
    """Determine which uncertainty source dominates for this sample."""
    scores = get_scores(result)
    source = max(scores, key=scores.get)
    name_map = {"v_score": "Visual", "lp_score": "Language-Prior", "a_score": "Alignment"}
    return name_map[source]


def analyze_single_file(jsonl_path: str) -> dict:
    """Analyze one decomposition result file."""
    results = load_results(jsonl_path)
    if not results:
        return {"file": jsonl_path, "error": "No valid results found"}

    # Extract score arrays
    v_scores = np.array([r.get("v_score", 0.0) for r in results])
    lp_scores = np.array([r.get("lp_score", 0.0) for r in results])
    a_scores = np.array([r.get("a_score", 0.0) for r in results])

    # Correct vs Incorrect split
    correct = [r for r in results if r.get("is_correct", False)]
    incorrect = [r for r in results if not r.get("is_correct", False)]

    # Per-failure-family breakdown
    family_stats = defaultdict(lambda: {"v": [], "lp": [], "a": [], "count": 0})
    for r in results:
        fam = r.get("failure_family", "unknown")
        scores = get_scores(r)
        family_stats[fam]["v"].append(scores["v_score"])
        family_stats[fam]["lp"].append(scores["lp_score"])
        family_stats[fam]["a"].append(scores["a_score"])
        family_stats[fam]["count"] += 1

    # Dominant source breakdown
    source_counts = defaultdict(int)
    for r in results:
        source_counts[dominant_source(r)] += 1

    # Correlation between V, L, A
    if len(v_scores) > 2:
        score_matrix = np.column_stack([v_scores, lp_scores, a_scores])
        corr_matrix = np.corrcoef(score_matrix.T)
    else:
        corr_matrix = np.eye(3)

    # LP answer match rate (how often blank-image gives same answer)
    lp_match_count = sum(1 for r in results if r.get("lp_answer_matches", False))

    return {
        "file": jsonl_path,
        "n_samples": len(results),
        "n_correct": len(correct),
        "n_incorrect": len(incorrect),
        "accuracy": len(correct) / len(results) if results else 0,
        "overall_scores": {
            "v_mean": float(np.mean(v_scores)),
            "v_std": float(np.std(v_scores)),
            "lp_mean": float(np.mean(lp_scores)),
            "lp_std": float(np.std(lp_scores)),
            "a_mean": float(np.mean(a_scores)),
            "a_std": float(np.std(a_scores)),
        },
        "correct_scores": {
            "v_mean": float(np.mean([r.get("v_score", 0) for r in correct])) if correct else 0,
            "lp_mean": float(np.mean([r.get("lp_score", 0) for r in correct])) if correct else 0,
            "a_mean": float(np.mean([r.get("a_score", 0) for r in correct])) if correct else 0,
        },
        "incorrect_scores": {
            "v_mean": float(np.mean([r.get("v_score", 0) for r in incorrect])) if incorrect else 0,
            "lp_mean": float(np.mean([r.get("lp_score", 0) for r in incorrect])) if incorrect else 0,
            "a_mean": float(np.mean([r.get("a_score", 0) for r in incorrect])) if incorrect else 0,
        },
        "per_family": {
            fam: {
                "v_mean": float(np.mean(s["v"])),
                "lp_mean": float(np.mean(s["lp"])),
                "a_mean": float(np.mean(s["a"])),
                "count": s["count"],
                "dominant": ["Visual", "Language-Prior", "Alignment"][
                    np.argmax([np.mean(s["v"]), np.mean(s["lp"]), np.mean(s["a"])])
                ],
            }
            for fam, s in family_stats.items()
        },
        "dominant_source_distribution": dict(source_counts),
        "correlation_matrix": {
            "V_LP": float(corr_matrix[0, 1]),
            "V_A": float(corr_matrix[0, 2]),
            "LP_A": float(corr_matrix[1, 2]),
        },
        "lp_answer_match_rate": lp_match_count / len(results) if results else 0,
    }


def print_analysis(summary: dict):
    """Pretty-print decomposition analysis."""
    fname = Path(summary["file"]).parent.name + "/" + Path(summary["file"]).name
    print(f"\n{'='*70}")
    print(f"  DECOMPOSITION ANALYSIS: {fname}")
    print(f"{'='*70}")
    print(f"  Samples: {summary['n_samples']} ({summary['n_correct']} correct, "
          f"{summary['n_incorrect']} incorrect)")
    print(f"  Accuracy: {summary['accuracy']:.1%}")

    # Overall V, L, A
    s = summary["overall_scores"]
    print(f"\n  Overall Uncertainty Scores (mean +/- std):")
    print(f"    V (Visual):         {s['v_mean']:.4f} +/- {s['v_std']:.4f}")
    print(f"    L (Language-Prior): {s['lp_mean']:.4f} +/- {s['lp_std']:.4f}")
    print(f"    A (Alignment):      {s['a_mean']:.4f} +/- {s['a_std']:.4f}")

    # Correct vs Incorrect
    c = summary["correct_scores"]
    ic = summary["incorrect_scores"]
    print(f"\n  Correct vs Incorrect Samples:")
    print(f"    {'':20s} {'V':>8s} {'L':>8s} {'A':>8s}")
    print(f"    {'Correct':20s} {c['v_mean']:>8.4f} {c['lp_mean']:>8.4f} {c['a_mean']:>8.4f}")
    print(f"    {'Incorrect':20s} {ic['v_mean']:>8.4f} {ic['lp_mean']:>8.4f} {ic['a_mean']:>8.4f}")

    # Correlation
    corr = summary["correlation_matrix"]
    print(f"\n  Correlation Between Components (low = good separation):")
    print(f"    V <-> L:  {corr['V_LP']:+.3f}")
    print(f"    V <-> A:  {corr['V_A']:+.3f}")
    print(f"    L <-> A:  {corr['LP_A']:+.3f}")

    # Per failure family
    print(f"\n  Per Failure Family:")
    print(f"    {'Family':<28s} {'V':>7s} {'L':>7s} {'A':>7s} {'Dominant':>15s} {'N':>5s}")
    print(f"    {'-'*28} {'-'*7} {'-'*7} {'-'*7} {'-'*15} {'-'*5}")
    for fam, stats in sorted(summary["per_family"].items()):
        print(f"    {fam:<28s} {stats['v_mean']:>7.4f} {stats['lp_mean']:>7.4f} "
              f"{stats['a_mean']:>7.4f} {stats['dominant']:>15s} {stats['count']:>5d}")

    # Dominant source distribution
    print(f"\n  Dominant Source Distribution:")
    for source, count in sorted(summary["dominant_source_distribution"].items()):
        pct = count / summary["n_samples"]
        print(f"    {source:<20s} {count:>5d} ({pct:.1%})")

    # LP answer match rate
    print(f"\n  Language-Prior: {summary['lp_answer_match_rate']:.1%} of samples gave "
          f"the same answer with a blank image")


def main():
    parser = argparse.ArgumentParser(description="Analyze decomposition results.")
    parser.add_argument("--results_dir", required=True,
                        help="Path to model results dir or top-level decomposition/ dir")
    parser.add_argument("--all_models", action="store_true",
                        help="Analyze all model subdirectories")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    from analysis.tee_output import TeeOutput
    tee = TeeOutput(str(analysis_dir / "full_report.txt"))
    tee.__enter__()


    if args.all_models:
        model_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()
                             and not d.name.startswith(("log_files", "analysis"))])
    else:
        model_dirs = [results_dir]

    all_summaries = []

    for model_dir in model_dirs:
        jsonl_files = sorted(model_dir.glob("*.jsonl"))
        jsonl_files = [f for f in jsonl_files if not f.name.startswith("test_")]

        if not jsonl_files:
            continue

        print(f"\n{'#'*70}")
        print(f"  MODEL: {model_dir.name}")
        print(f"{'#'*70}")

        # Create analysis output dir for this model
        model_analysis_dir = analysis_dir / model_dir.name
        model_analysis_dir.mkdir(parents=True, exist_ok=True)

        for jf in jsonl_files:
            summary = analyze_single_file(str(jf))
            if "error" not in summary:
                print_analysis(summary)
                all_summaries.append(summary)

        # Save per-model summary CSV
        import csv
        report_path = model_analysis_dir / "vla_summary.csv"
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "n_samples", "accuracy", "v_mean", "lp_mean", "a_mean",
                             "v_std", "lp_std", "a_std", "lp_match_rate",
                             "corr_V_LP", "corr_V_A", "corr_LP_A"])
            for s in all_summaries:
                if Path(s["file"]).parent.name == model_dir.name:
                    o = s["overall_scores"]
                    c = s["correlation_matrix"]
                    writer.writerow([
                        Path(s["file"]).stem, s["n_samples"], f"{s['accuracy']:.4f}",
                        f"{o['v_mean']:.4f}", f"{o['lp_mean']:.4f}", f"{o['a_mean']:.4f}",
                        f"{o['v_std']:.4f}", f"{o['lp_std']:.4f}", f"{o['a_std']:.4f}",
                        f"{s['lp_answer_match_rate']:.4f}",
                        f"{c['V_LP']:.4f}", f"{c['V_A']:.4f}", f"{c['LP_A']:.4f}",
                    ])
        print(f"  Summary saved -> {report_path}")

        # Save per-family breakdown CSV
        family_path = model_analysis_dir / "per_family_breakdown.csv"
        with open(family_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "failure_family", "v_mean", "lp_mean", "a_mean",
                             "dominant", "count"])
            for s in all_summaries:
                if Path(s["file"]).parent.name == model_dir.name:
                    ds = Path(s["file"]).stem
                    for fam, stats in sorted(s["per_family"].items()):
                        writer.writerow([
                            ds, fam, f"{stats['v_mean']:.4f}", f"{stats['lp_mean']:.4f}",
                            f"{stats['a_mean']:.4f}", stats["dominant"], stats["count"],
                        ])
        print(f"  Family breakdown saved -> {family_path}")

    # Cross-model comparison
    if len(all_summaries) > 1:
        print(f"\n\n{'='*85}")
        print(f"  CROSS-MODEL V/L/A COMPARISON")
        print(f"{'='*85}")
        print(f"  {'Model/Dataset':<35s} {'V':>7s} {'L':>7s} {'A':>7s} {'LP Match%':>10s}")
        print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*10}")
        for s in all_summaries:
            fname = Path(s["file"]).parent.name + "/" + Path(s["file"]).stem
            o = s["overall_scores"]
            print(f"  {fname:<35s} {o['v_mean']:>7.4f} {o['lp_mean']:>7.4f} "
                  f"{o['a_mean']:>7.4f} {s['lp_answer_match_rate']:>9.1%}")

        # Save cross-model CSV
        import csv
        analysis_dir.mkdir(parents=True, exist_ok=True)
        comparison_csv = analysis_dir / "cross_model_vla.csv"
        with open(comparison_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "dataset", "v_mean", "lp_mean", "a_mean", "lp_match_rate"])
            for s in all_summaries:
                o = s["overall_scores"]
                writer.writerow([
                    Path(s["file"]).parent.name, Path(s["file"]).stem,
                    f"{o['v_mean']:.4f}", f"{o['lp_mean']:.4f}", f"{o['a_mean']:.4f}",
                    f"{s['lp_answer_match_rate']:.4f}",
                ])
        print(f"\n  Cross-model comparison saved -> {comparison_csv}")

    tee.__exit__(None, None, None)
    print(f"Full report saved -> {analysis_dir / 'full_report.txt'}")


if __name__ == "__main__":
    main()

