"""
Rebuttal Experiment 1: Routing Control Analysis

Reads the routing control JSONL results and prints a comparison table showing
accuracy, fix rate, and break rate for each routing policy (matched, random,
permuted, generic) vs. no intervention.

Usage:
    python rebuttal/exp1_routing_control/analyze_routing_control.py \\
        --results_dir results/rebuttal_exp1_routing/

    # Quick-run (after merging chunks):
    #   python rebuttal/exp1_routing_control/analyze_routing_control.py --results_dir results/rebuttal_exp1_routing/
"""

import argparse
import json
import sys
from pathlib import Path

def analyze_routing(results_path):
    with open(results_path, "r") as f:
        data = [json.loads(x) for x in f if x.strip()]
    
    valid = [r for r in data if not r.get("skipped")]
    n = len(valid)
    if n == 0:
        return
        
    orig_acc = sum(r["original_correct"] for r in valid) / n
    stats = {}
    
    for prefix in ["matched", "permuted", "random", "generic"]:
        acc = sum(r[f"{prefix}_correct"] for r in valid) / n
        fixed = sum(r[f"{prefix}_fixed"] for r in valid)
        broke = sum(r[f"{prefix}_broke"] for r in valid)
        orig_wrong = sum(not r["original_correct"] for r in valid)
        orig_right = sum(r["original_correct"] for r in valid)
        
        fix_rate = fixed / orig_wrong if orig_wrong else 0
        break_rate = broke / orig_right if orig_right else 0
        
        stats[prefix] = {
            "acc": acc,
            "fix_rate": fix_rate,
            "break_rate": break_rate
        }
    
    print(f"--- Results for {results_path} ({n} samples) ---")
    print(f"{'Policy':<15} | {'Accuracy':<10} | {'Fix Rate':<10} | {'Break Rate':<10}")
    print("-" * 55)
    print(f"{'No Intervention':<15} | {orig_acc:<10.3f} | {'-':<10} | {'-':<10}")
    for prefix in ["matched", "random", "permuted", "generic"]:
        st = stats[prefix]
        print(f"{prefix:<15} | {st['acc']:<10.3f} | {st['fix_rate']:<10.3f} | {st['break_rate']:<10.3f}")
    print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    
    for f in Path(args.results_dir).rglob("*.jsonl"):
        analyze_routing(str(f))
