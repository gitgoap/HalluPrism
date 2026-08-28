#!/bin/bash
# ============================================================
# Master Analysis Runner — Run this after ALL experiments are done
#
# Runs all 6 analysis scripts in sequence.
# Because analysis only reads JSONL files and does math (no MLLM inference),
# this does NOT need a GPU and finishes in seconds.
#
# Usage:
#   bash analysis/run_all_analysis.sh
#
# Individual commands (if you prefer to run them one at a time):
#   1. python analysis/analyze_baselines.py --results_dir results/baselines/ --all_models --export_dangerous
#   2. python analysis/analyze_decomposition.py --results_dir results/decomposition/ --all_models
#   3. python analysis/rq3_failure_family.py --results_dir results/decomposition/ --all_models
#   4. python analysis/rq4_abstention.py --results_dir results/decomposition/ --all_models
#   5. python analysis/rq5_intervention.py --results_dir results/intervention/ --all_models
#
# For rebuttal analysis, see: rebuttal/run_rebuttal_4gpus.sh
# ============================================================

set -e

echo "========================================"
echo "1. BASELINE ANALYSIS"
echo "========================================"
python analysis/analyze_baselines.py \
    --results_dir results/baselines/ \
    --all_models \
    --export_dangerous

echo "========================================"
echo "2. DECOMPOSITION ANALYSIS"
echo "========================================"
python analysis/analyze_decomposition.py \
    --results_dir results/decomposition/ \
    --all_models

echo "========================================"
echo "3. IDENTIFIABILITY (RQ2) ANALYSIS"
echo "========================================"
# Requires both clean decomposition results and the ruined identifiability results
python analysis/analyze_identifiability.py \
    --clean_dir results/decomposition/ \
    --identifiability_dir results/identifiability/ \
    --all_models

echo "========================================"
echo "4. FAILURE FAMILY PREDICTION (RQ3)"
echo "========================================"
python analysis/rq3_failure_family.py \
    --results_dir results/decomposition/ \
    --all_models

echo "========================================"
echo "5. SOURCE-AWARE ABSTENTION (RQ4)"
echo "========================================"
python analysis/rq4_abstention.py \
    --results_dir results/decomposition/ \
    --all_models

echo "========================================"
echo "6. MATCHED INTERVENTION (RQ5)"
echo "========================================"
# This will gracefully skip if the intervention folder doesn't exist yet
if [ -d "results/intervention/" ]; then
    python analysis/rq5_intervention.py \
        --results_dir results/intervention/ \
        --all_models
else
    echo "Skipping RQ5: results/intervention/ directory not found yet."
fi

echo "========================================"
echo "ALL ANALYSIS COMPLETE!"
echo "Check the results/*/analysis/ folders for CSVs and full_report.txt files."
echo "========================================"
