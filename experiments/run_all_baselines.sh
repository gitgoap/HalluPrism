#!/bin/bash
# ============================================================
# Run ALL baseline experiments: 4 models x 4 datasets = 16 runs
# ============================================================
#
# Usage:
#   bash experiments/run_all_baselines.sh
#
# Prerequisites:
#   - Model weights downloaded to /home/models/
#   - Datasets in data/ (HallusionBench, POPE, VSR, VizWiz)
#   - CUDA available
#
# Each run saves to results/baselines/<model>/<dataset>.jsonl
# Logs go to      results/baselines/<model>/log_files_<model>/<dataset>.log
# ============================================================

set -e  # Exit on first error

DATA_ROOT="data/"
RESULTS_DIR="results/baselines"

CUDA_DEVICE="cuda:1"

# Model paths — update these to match your server layout
LLAVA_MISTRAL_PATH="/home/models/llava-v1.6-mistral-7b-hf"
LLAVA_VICUNA_PATH="/home/models/llava-v1.6-vicuna-7b-hf"
QWEN_VL_PATH="/home/models/Qwen3-VL-8B-Instruct"
GEMMA_VL_PATH="/home/models/gemma-4-E4B-it"
DATASETS=("hallusionbench" "pope" "vsr" "vizwiz")

mkdir -p "$RESULTS_DIR"

# ============================================================
# Model 1: LLaVA-1.6-Mistral-7B
# ============================================================
echo "========== LLaVA-1.6-Mistral-7B =========="
for ds in "${DATASETS[@]}"; do
    echo "[LLaVA-Mistral] Running $ds..."
    python -m experiments.run_baseline \
        --dataset "$ds" \
        --model llava_mistral \
        --model_path "$LLAVA_MISTRAL_PATH" \
        --cuda_device "$CUDA_DEVICE" \
        --data_root "$DATA_ROOT" \
        --output "$RESULTS_DIR/llava_mistral/${ds}.jsonl" \
        --self_reported
    echo "[LLaVA-Mistral] $ds done."
done

# ============================================================
# Model 2: LLaVA-1.6-Vicuna-7B
# ============================================================
echo "========== LLaVA-1.6-Vicuna-7B =========="
for ds in "${DATASETS[@]}"; do
    echo "[LLaVA-Vicuna] Running $ds..."
    python -m experiments.run_baseline \
        --dataset "$ds" \
        --model llava_vicuna \
        --model_path "$LLAVA_VICUNA_PATH" \
        --cuda_device "$CUDA_DEVICE" \
        --data_root "$DATA_ROOT" \
        --output "$RESULTS_DIR/llava_vicuna/${ds}.jsonl" \
        --self_reported
    echo "[LLaVA-Vicuna] $ds done."
done

# ============================================================
# Model 3: Qwen3-VL-8B-Instruct
# ============================================================
echo "========== Qwen3-VL-8B-Instruct =========="
for ds in "${DATASETS[@]}"; do
    echo "[Qwen3-VL] Running $ds..."
    python -m experiments.run_baseline \
        --dataset "$ds" \
        --model qwen_vl \
        --model_path "$QWEN_VL_PATH" \
        --cuda_device "$CUDA_DEVICE" \
        --data_root "$DATA_ROOT" \
        --output "$RESULTS_DIR/qwen_vl/${ds}.jsonl" \
        --self_reported
    echo "[Qwen3-VL] $ds done."
done

# ============================================================
# Model 4: Gemma 4 E4B
# ============================================================
echo "========== Gemma 4 E4B =========="
for ds in "${DATASETS[@]}"; do
    echo "[Gemma4] Running $ds..."
    python -m experiments.run_baseline \
        --dataset "$ds" \
        --model gemma_vl \
        --model_path "$GEMMA_VL_PATH" \
        --cuda_device "$CUDA_DEVICE" \
        --data_root "$DATA_ROOT" \
        --output "$RESULTS_DIR/gemma_vl/${ds}.jsonl" \
        --self_reported
    echo "[Gemma4] $ds done."
done



echo ""
echo "=========================================="
echo "All baseline runs complete!"
echo "Results in: $RESULTS_DIR/"
echo "=========================================="
ls -la "$RESULTS_DIR/"