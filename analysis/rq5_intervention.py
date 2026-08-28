"""
RQ5: Matched Intervention Analysis

Reads the intervention JSONL files and compares:
  - No intervention (original prediction from decomposition)
  - Matched policy (fix applied based on dominant source)
  - Generic policy (same fix for everyone — CoT prompt)

Key metrics:
  - Fix rate: % of originally-wrong samples that become correct
  - Break rate: % of originally-correct samples that become wrong
  - Net improvement: fix rate - break rate
  - Per-source breakdown: does matching actually help?

Usage:
    python analysis/rq5_intervention.py \
        --results_dir results/intervention/ \
        --all_models

    # Quick-run (recommended after all experiments are done):
    #   python analysis/rq5_intervention.py --results_dir results/intervention/ --all_models
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_results(jsonl_path: str) -> List[dict]:
    """Load intervention results, skipping invalid entries."""
    results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("skipped"):
                continue
            results.append(r)
    return results


def analyze_intervention(results: List[dict]) -> dict:
    """Compute fix/break rates for matched vs generic policy."""
    n_total = len(results)
    if n_total == 0:
        return {}

    # Overall accuracy
    n_orig_correct = sum(1 for r in results if r.get("original_correct", False))
    n_matched_correct = sum(1 for r in results if r.get("matched_correct", False))
    n_generic_correct = sum(1 for r in results if r.get("generic_correct", False))

    # Fix/break counts
    n_matched_fixed = sum(1 for r in results if r.get("matched_fixed", False))
    n_generic_fixed = sum(1 for r in results if r.get("generic_fixed", False))
    n_matched_broke = sum(1 for r in results if r.get("matched_broke", False))
    n_generic_broke = sum(1 for r in results if r.get("generic_broke", False))

    n_originally_wrong = n_total - n_orig_correct

    # Per-source breakdown
    per_source = defaultdict(lambda: {
        "total": 0, "orig_correct": 0,
        "matched_correct": 0, "generic_correct": 0,
        "matched_fixed": 0, "generic_fixed": 0,
        "matched_broke": 0, "generic_broke": 0,
    })

    for r in results:
        source = r.get("dominant_source", "unknown")
        per_source[source]["total"] += 1
        if r.get("original_correct"):
            per_source[source]["orig_correct"] += 1
        if r.get("matched_correct"):
            per_source[source]["matched_correct"] += 1
        if r.get("generic_correct"):
            per_source[source]["generic_correct"] += 1
        if r.get("matched_fixed"):
            per_source[source]["matched_fixed"] += 1
        if r.get("generic_fixed"):
            per_source[source]["generic_fixed"] += 1
        if r.get("matched_broke"):
            per_source[source]["matched_broke"] += 1
        if r.get("generic_broke"):
            per_source[source]["generic_broke"] += 1

    return {
        "n_total": n_total,
        "original_accuracy": n_orig_correct / n_total,
        "matched_accuracy": n_matched_correct / n_total,
        "generic_accuracy": n_generic_correct / n_total,
        "n_originally_wrong": n_originally_wrong,
        "matched_fix_rate": n_matched_fixed / n_originally_wrong if n_originally_wrong > 0 else 0,
        "generic_fix_rate": n_generic_fixed / n_originally_wrong if n_originally_wrong > 0 else 0,
        "matched_break_rate": n_matched_broke / n_orig_correct if n_orig_correct > 0 else 0,
        "generic_break_rate": n_generic_broke / n_orig_correct if n_orig_correct > 0 else 0,
        "matched_net": (n_matched_fixed - n_matched_broke) / n_total,
        "generic_net": (n_generic_fixed - n_generic_broke) / n_total,
        "per_source": dict(per_source),
    }


def print_analysis(analysis: dict, title: str = ""):
    """Pretty-print RQ5 results."""
    if not analysis:
        return

    print(f"\n  {'='*70}")
    print(f"  RQ5: MATCHED INTERVENTION{(' — ' + title) if title else ''}")
    print(f"  {'='*70}")
    print(f"  Total samples: {analysis['n_total']}")
    print(f"  Originally wrong: {analysis['n_originally_wrong']}")

    # Main accuracy comparison
    print(f"\n  Accuracy Comparison:")
    print(f"    {'Policy':<25s} {'Accuracy':>10s} {'Fix Rate':>10s} {'Break Rate':>11s} {'Net':>8s}")
    print(f"    {'-'*25} {'-'*10} {'-'*10} {'-'*11} {'-'*8}")
    print(f"    {'No Intervention':<25s} {analysis['original_accuracy']:>9.1%} {'—':>10s} {'—':>11s} {'—':>8s}")
    print(f"    {'Generic (CoT for all)':<25s} {analysis['generic_accuracy']:>9.1%} "
          f"{analysis['generic_fix_rate']:>9.1%} {analysis['generic_break_rate']:>10.1%} "
          f"{analysis['generic_net']:>+7.1%}")
    print(f"    {'Matched (source-aware)':<25s} {analysis['matched_accuracy']:>9.1%} "
          f"{analysis['matched_fix_rate']:>9.1%} {analysis['matched_break_rate']:>10.1%} "
          f"{analysis['matched_net']:>+7.1%}")

    # Per-source breakdown
    print(f"\n  Per Dominant Source:")
    print(f"    {'Source':<20s} {'N':>5s} {'Orig Acc':>9s} {'Matched':>9s} {'Generic':>9s} "
          f"{'M.Fixed':>8s} {'G.Fixed':>8s}")
    print(f"    {'-'*20} {'-'*5} {'-'*9} {'-'*9} {'-'*9} {'-'*8} {'-'*8}")

    for source, stats in sorted(analysis["per_source"].items()):
        n = stats["total"]
        orig_acc = stats["orig_correct"] / n if n > 0 else 0
        matched_acc = stats["matched_correct"] / n if n > 0 else 0
        generic_acc = stats["generic_correct"] / n if n > 0 else 0
        m_fixed = stats["matched_fixed"]
        g_fixed = stats["generic_fixed"]
        print(f"    {source:<20s} {n:>5d} {orig_acc:>8.1%} {matched_acc:>8.1%} "
              f"{generic_acc:>8.1%} {m_fixed:>8d} {g_fixed:>8d}")

    # Verdict
    matched_better = analysis["matched_net"] > analysis["generic_net"]
    delta = analysis["matched_accuracy"] - analysis["generic_accuracy"]
    print(f"\n  Matched vs Generic accuracy delta: {delta:+.1%}")
    print(f"  Verdict: Matched policy {'OUTPERFORMS' if matched_better else 'UNDERPERFORMS'} "
          f"generic policy")
    if matched_better:
        print(f"  => Knowing the uncertainty source IS actionable!")


def main():
    parser = argparse.ArgumentParser(description="RQ5: Intervention analysis.")
    parser.add_argument("--results_dir", required=True,
                        help="Path to intervention results directory")
    parser.add_argument("--model", default=None)
    parser.add_argument("--all_models", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    from analysis.tee_output import TeeOutput
    tee = TeeOutput(str(analysis_dir / "full_report.txt"))
    tee.__enter__()

    if args.all_models:
        model_dirs = sorted([
            d for d in results_dir.iterdir()
            if d.is_dir() and not d.name.startswith(("log_files", "analysis"))
        ])
    elif args.model:
        model_dirs = [results_dir / args.model]
    else:
        print("Error: specify --model or --all_models")
        return

    all_results_flat = []
    all_rows = []

    for model_dir in model_dirs:
        jsonl_files = sorted(model_dir.glob("*.jsonl"))
        jsonl_files = [f for f in jsonl_files if not f.name.startswith("test_")]

        if not jsonl_files:
            continue

        print(f"\n{'#'*70}")
        print(f"  MODEL: {model_dir.name}")
        print(f"{'#'*70}")

        model_results = []
        for jf in jsonl_files:
            results = load_results(str(jf))
            if not results:
                continue
            model_results.extend(results)

            # Per-dataset analysis
            analysis = analyze_intervention(results)
            print_analysis(analysis, title=f"{model_dir.name}/{jf.stem}")

            # Collect for CSV
            all_rows.append({
                "model": model_dir.name, "dataset": jf.stem,
                "original_accuracy": analysis["original_accuracy"],
                "matched_accuracy": analysis["matched_accuracy"],
                "generic_accuracy": analysis["generic_accuracy"],
                "matched_fix_rate": analysis["matched_fix_rate"],
                "generic_fix_rate": analysis["generic_fix_rate"],
                "matched_net": analysis["matched_net"],
                "generic_net": analysis["generic_net"],
                "n_total": analysis["n_total"],
            })

        # Per-model aggregate
        if len(jsonl_files) > 1 and model_results:
            analysis = analyze_intervention(model_results)
            print_analysis(analysis, title=f"{model_dir.name}/ALL")

        all_results_flat.extend(model_results)

    # Pooled analysis
    if len(model_dirs) > 1 and all_results_flat:
        print(f"\n{'#'*70}")
        print(f"  POOLED (all models)")
        print(f"{'#'*70}")
        analysis = analyze_intervention(all_results_flat)
        print_analysis(analysis, title="ALL MODELS POOLED")

    # Save comparison CSV
    if all_rows:
        import csv
        csv_path = analysis_dir / "intervention_comparison.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "model", "dataset", "original_accuracy", "matched_accuracy",
                "generic_accuracy", "matched_fix_rate", "generic_fix_rate",
                "matched_net", "generic_net", "n_total",
            ])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n  RQ5 results saved -> {csv_path}")

    tee.__exit__(None, None, None)
    print(f"Full report saved -> {analysis_dir / 'full_report.txt'}")


if __name__ == "__main__":
    main()

