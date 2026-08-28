"""
Baseline Results Analyzer — Calibration Quadrant Analysis

Reads the .jsonl result files produced by run_baseline.py and generates:
  1. A 2x2 calibration quadrant table (correct/incorrect × high/low confidence)
  2. Per-dataset accuracy summary
  3. Per-failure-family breakdown
  4. ECE (Expected Calibration Error)
  5. CSV export of the "dangerous" samples (incorrect + high confidence)

Usage:
    python analysis/analyze_baselines.py \
        --results_dir results/baselines/llava_mistral/ \
        --threshold 0.5

    # Analyze all models at once:
    python analysis/analyze_baselines.py \
        --results_dir results/baselines/ \
        --all_models

    # Quick-run (recommended after all experiments are done):
    #   python analysis/analyze_baselines.py --results_dir results/baselines/ --all_models --export_dangerous
    #
    # Or run everything at once with:
    #   bash analysis/run_all_analysis.sh
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_results(jsonl_path: str) -> List[dict]:
    """Load results from a JSONL file."""
    results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def classify_quadrant(result: dict, threshold: float = 0.5) -> str:
    """
    Classify a sample into one of the 4 calibration quadrants.

    Returns one of:
        'correct_high'   — Q4: Correct + High Confidence (ideal)
        'correct_low'    — Q1: Correct + Low Confidence (under-confident)
        'incorrect_high' — Q2: Incorrect + High Confidence (DANGEROUS - hallucination)
        'incorrect_low'  — Q3: Incorrect + Low Confidence (appropriately uncertain)
    """
    is_correct = result.get("is_correct", False)
    confidence = result.get("scalar_confidence")

    if confidence is None:
        return "skipped"

    high_conf = confidence >= threshold

    if is_correct and high_conf:
        return "correct_high"
    elif is_correct and not high_conf:
        return "correct_low"
    elif not is_correct and high_conf:
        return "incorrect_high"
    else:
        return "incorrect_low"


def compute_ece(results: List[dict], n_bins: int = 10) -> float:
    """
    Expected Calibration Error.
    Measures how well confidence tracks actual accuracy.
    ECE = 0 means perfectly calibrated. Higher = worse.
    """
    bins = [[] for _ in range(n_bins)]

    for r in results:
        conf = r.get("scalar_confidence")
        if conf is None:
            continue
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append(r)

    ece = 0.0
    total = sum(len(b) for b in bins)
    if total == 0:
        return 0.0

    for b in bins:
        if not b:
            continue
        avg_conf = sum(r["scalar_confidence"] for r in b) / len(b)
        avg_acc = sum(1 for r in b if r.get("is_correct", False)) / len(b)
        ece += (len(b) / total) * abs(avg_acc - avg_conf)

    return ece


def analyze_single_file(jsonl_path: str, threshold: float = 0.5) -> dict:
    """Analyze one result file and return a summary dict."""
    results = load_results(jsonl_path)

    # Filter out skipped samples
    valid = [r for r in results if r.get("model_prediction") not in
             ("[skipped_no_image]", "[image_load_error]", None)]
    skipped = len(results) - len(valid)

    # Overall accuracy
    correct_count = sum(1 for r in valid if r.get("is_correct", False))
    accuracy = correct_count / len(valid) if valid else 0.0

    # Quadrant classification
    quadrants = defaultdict(list)
    for r in valid:
        q = classify_quadrant(r, threshold)
        quadrants[q].append(r)

    # Per-family breakdown
    family_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in valid:
        family = r.get("failure_family", "unknown")
        family_stats[family]["total"] += 1
        if r.get("is_correct", False):
            family_stats[family]["correct"] += 1

    # Confidence statistics
    confs = [r["scalar_confidence"] for r in valid if r.get("scalar_confidence") is not None]
    correct_confs = [r["scalar_confidence"] for r in valid
                     if r.get("is_correct") and r.get("scalar_confidence") is not None]
    wrong_confs = [r["scalar_confidence"] for r in valid
                   if not r.get("is_correct") and r.get("scalar_confidence") is not None]

    # Self-reported stats
    sr_values = [r["self_reported_confidence"] for r in valid
                 if r.get("self_reported_confidence") is not None]
    sr_null_count = sum(1 for r in valid if r.get("self_reported_confidence") is None)

    # ECE
    ece = compute_ece(valid)

    return {
        "file": jsonl_path,
        "total_samples": len(results),
        "valid_samples": len(valid),
        "skipped_samples": skipped,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "ece": ece,
        "quadrants": {
            "Q1_correct_low_conf": len(quadrants["correct_low"]),
            "Q2_incorrect_high_conf": len(quadrants["incorrect_high"]),
            "Q3_incorrect_low_conf": len(quadrants["incorrect_low"]),
            "Q4_correct_high_conf": len(quadrants["correct_high"]),
        },
        "confidence_stats": {
            "mean_all": sum(confs) / len(confs) if confs else 0,
            "mean_correct": sum(correct_confs) / len(correct_confs) if correct_confs else 0,
            "mean_incorrect": sum(wrong_confs) / len(wrong_confs) if wrong_confs else 0,
        },
        "self_reported_stats": {
            "parsed_count": len(sr_values),
            "null_count": sr_null_count,
            "parse_rate": len(sr_values) / len(valid) if valid else 0,
            "mean_when_parsed": sum(sr_values) / len(sr_values) if sr_values else 0,
        },
        "per_family": {
            fam: {
                "accuracy": s["correct"] / s["total"] if s["total"] > 0 else 0,
                "count": s["total"],
            }
            for fam, s in family_stats.items()
        },
        # Keep raw dangerous samples for CSV export
        "_dangerous_samples": quadrants["incorrect_high"],
    }


def print_analysis(summary: dict, verbose: bool = False):
    """Pretty-print analysis results to terminal."""
    fname = Path(summary["file"]).name
    print(f"\n{'='*65}")
    print(f"  ANALYSIS: {fname}")
    print(f"{'='*65}")
    print(f"  Samples: {summary['valid_samples']} valid / {summary['skipped_samples']} skipped")
    print(f"  Accuracy: {summary['accuracy']:.1%} ({summary['correct_count']}/{summary['valid_samples']})")
    print(f"  ECE (Expected Calibration Error): {summary['ece']:.4f}")

    # Quadrant table
    q = summary["quadrants"]
    total = summary["valid_samples"]
    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │        CALIBRATION QUADRANT (threshold=0.5)  │")
    print(f"  ├──────────────────────┬──────────────────────┤")
    print(f"  │   HIGH Confidence    │   LOW Confidence     │")
    print(f"  ├──────────────────────┼──────────────────────┤")
    print(f"  │ Q4: CORRECT + HIGH   │ Q1: CORRECT + LOW    │")
    print(f"  │     {q['Q4_correct_high_conf']:>5d} ({q['Q4_correct_high_conf']/total:.1%})     │"
          f"     {q['Q1_correct_low_conf']:>5d} ({q['Q1_correct_low_conf']/total:.1%})      │")
    print(f"  │     (ideal)          │     ⚠ under-confident │")
    print(f"  ├──────────────────────┼──────────────────────┤")
    print(f"  │ Q2: INCORRECT + HIGH │ Q3: INCORRECT + LOW  │")
    print(f"  │     {q['Q2_incorrect_high_conf']:>5d} ({q['Q2_incorrect_high_conf']/total:.1%})     │"
          f"     {q['Q3_incorrect_low_conf']:>5d} ({q['Q3_incorrect_low_conf']/total:.1%})      │")
    print(f"  │     🔴 DANGEROUS      │     (appropriate)    │")
    print(f"  └──────────────────────┴──────────────────────┘")

    # Confidence gap
    cs = summary["confidence_stats"]
    print(f"\n  Avg Confidence (correct):   {cs['mean_correct']:.3f}")
    print(f"  Avg Confidence (incorrect): {cs['mean_incorrect']:.3f}")
    gap = cs['mean_correct'] - cs['mean_incorrect']
    print(f"  Confidence Gap:             {gap:+.3f} {'(good separation)' if gap > 0.05 else '(POOR separation!)'}")

    # Self-reported
    sr = summary["self_reported_stats"]
    print(f"\n  Self-Reported Confidence:")
    print(f"    Parsed: {sr['parsed_count']}/{sr['parsed_count']+sr['null_count']} ({sr['parse_rate']:.1%})")
    if sr['parsed_count'] > 0:
        print(f"    Mean (when parsed): {sr['mean_when_parsed']:.3f}")

    # Per-family
    if summary["per_family"]:
        print(f"\n  Per-Family Accuracy:")
        for fam, stats in sorted(summary["per_family"].items()):
            print(f"    {fam:<25s} {stats['accuracy']:.1%}  (n={stats['count']})")


def export_dangerous_csv(summary: dict, output_path: str):
    """Export Q2 (incorrect + high confidence) samples to CSV for review."""
    dangerous = summary["_dangerous_samples"]
    if not dangerous:
        print(f"  No dangerous samples to export.")
        return

    import csv
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "dataset", "question", "gold_answer",
            "model_prediction", "scalar_confidence", "self_reported_confidence",
            "failure_family", "image_path"
        ])
        for r in dangerous:
            writer.writerow([
                r.get("sample_id", ""),
                r.get("dataset_name", ""),
                r.get("text_input", "")[:100],
                r.get("gold_answer", ""),
                r.get("model_prediction", ""),
                f"{r.get('scalar_confidence', 0):.3f}",
                r.get("self_reported_confidence", "null"),
                r.get("failure_family", ""),
                r.get("image_path", ""),
            ])
    print(f"  Exported {len(dangerous)} dangerous samples -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze baseline results.")
    parser.add_argument("--results_dir", required=True,
                        help="Path to model results directory or top-level baselines/ dir")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for high/low split (default: 0.5)")
    parser.add_argument("--all_models", action="store_true",
                        help="Analyze all model subdirectories under results_dir")
    parser.add_argument("--export_dangerous", action="store_true",
                        help="Export Q2 (incorrect+high conf) samples to CSV")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    from analysis.tee_output import TeeOutput
    tee = TeeOutput(str(analysis_dir / "full_report.txt"))
    tee.__enter__()

    if args.all_models:
        # Iterate over model subdirectories
        model_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()
                             and not d.name.startswith(("log_files", "analysis"))])
    else:
        model_dirs = [results_dir]

    all_summaries = []

    for model_dir in model_dirs:
        jsonl_files = sorted(model_dir.glob("*.jsonl"))
        # Exclude test files
        jsonl_files = [f for f in jsonl_files if not f.name.startswith("test_")]

        if not jsonl_files:
            continue

        print(f"\n{'#'*65}")
        print(f"  MODEL: {model_dir.name}")
        print(f"{'#'*65}")

        # Create analysis output dir for this model
        model_analysis_dir = analysis_dir / model_dir.name
        model_analysis_dir.mkdir(parents=True, exist_ok=True)

        for jsonl_path in jsonl_files:
            summary = analyze_single_file(str(jsonl_path), threshold=args.threshold)
            print_analysis(summary)
            all_summaries.append(summary)

            if args.export_dangerous:
                csv_path = model_analysis_dir / f"{jsonl_path.stem}.dangerous.csv"
                export_dangerous_csv(summary, str(csv_path))

        # Save per-model summary report
        report_path = model_analysis_dir / "summary.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            for s in all_summaries:
                if Path(s["file"]).parent.name == model_dir.name:
                    ds = Path(s["file"]).stem
                    q = s["quadrants"]
                    f.write(f"{ds}: acc={s['accuracy']:.1%}, ECE={s['ece']:.4f}, "
                            f"Q2_dangerous={q['Q2_incorrect_high_conf']}\n")
        print(f"  Summary saved -> {report_path}")

    # Cross-model comparison table
    if len(all_summaries) > 1:
        print(f"\n\n{'='*80}")
        print(f"  CROSS-MODEL COMPARISON")
        print(f"{'='*80}")
        print(f"  {'File':<35s} {'Acc':>6s} {'ECE':>6s} {'Q1(C+L)':>8s} {'Q2(I+H)':>8s} {'Q3(I+L)':>8s} {'Q4(C+H)':>8s}")
        print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for s in all_summaries:
            fname = Path(s["file"]).parent.name + "/" + Path(s["file"]).name
            q = s["quadrants"]
            print(f"  {fname:<35s} {s['accuracy']:>5.1%} {s['ece']:>6.4f} "
                  f"{q['Q1_correct_low_conf']:>8d} {q['Q2_incorrect_high_conf']:>8d} "
                  f"{q['Q3_incorrect_low_conf']:>8d} {q['Q4_correct_high_conf']:>8d}")

        # Save cross-model comparison CSV
        comparison_csv = analysis_dir / "cross_model_comparison.csv"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        import csv
        with open(comparison_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "dataset", "accuracy", "ece",
                             "Q1_correct_low", "Q2_dangerous", "Q3_incorrect_low", "Q4_correct_high"])
            for s in all_summaries:
                q = s["quadrants"]
                writer.writerow([
                    Path(s["file"]).parent.name,
                    Path(s["file"]).stem,
                    f"{s['accuracy']:.4f}",
                    f"{s['ece']:.4f}",
                    q["Q1_correct_low_conf"],
                    q["Q2_incorrect_high_conf"],
                    q["Q3_incorrect_low_conf"],
                    q["Q4_correct_high_conf"],
                ])
        print(f"\n  Cross-model comparison saved -> {comparison_csv}")

    tee.__exit__(None, None, None)
    print(f"Full report saved -> {analysis_dir / 'full_report.txt'}")


if __name__ == "__main__":
    main()

