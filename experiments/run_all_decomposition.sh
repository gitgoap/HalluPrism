#!/bin/bash
# ============================================================
# Run Uncertainty Decomposition: 4 models x 4 datasets = 16 runs
# ============================================================
#
# Each sample requires ~7-8 forward passes (vs 1 for baseline),
# so this takes 7-8x longer than run_all_baselines.sh.
#
# Results go to: results/decomposition/<model>/<dataset>.jsonl
# Logs go to:    results/decomposition/<model>/log_files_<model>/<dataset>.log
# ============================================================

set -e

DATA_ROOT="data/"
RESULTS_DIR="results/decomposition"
CUDA_DEVICE="cuda:2"

# Model paths
LLAVA_MISTRAL_PATH="/home/models/llava-v1.6-mistral-7b-hf"
LLAVA_VICUNA_PATH="/home/models/llava-v1.6-vicuna-7b-hf"
QWEN_VL_PATH="/home/models/Qwen3-VL-8B-Instruct"
GEMMA_VL_PATH="/home/models/gemma-4-E4B-it"

DATASETS=("hallusionbench" "pope" "vsr" "vizwiz")

mkdir -p "$RESULTS_DIR"

# ============================================================
# Model 1: LLaVA-1.6-Mistral-7B
# ============================================================
echo "========== [DECOMP] LLaVA-1.6-Mistral-7B =========="
for ds in "${DATASETS[@]}"; do
    echo "[LLaVA-Mistral DECOMP] Running $ds..."
    python -m experiments.run_decomposition \
        --dataset "$ds" \
        --model llava_mistral \
        --model_path "$LLAVA_MISTRAL_PATH" \
        --cuda_device "$CUDA_DEVICE" \
        --data_root "$DATA_ROOT" \
        --output "$RESULTS_DIR/llava_mistral/${ds}.jsonl"
    echo "[LLaVA-Mistral DECOMP] $ds done."
done

# ============================================================
# Model 2: LLaVA-1.6-Vicuna-7B
# ============================================================
echo "========== [DECOMP] LLaVA-1.6-Vicuna-7B =========="
for ds in "${DATASETS[@]}"; do
    echo "[LLaVA-Vicuna DECOMP] Running $ds..."
    python -m experiments.run_decomposition \
        --dataset "$ds" \
        --model llava_vicuna \
        --model_path "$LLAVA_VICUNA_PATH" \
        --cuda_device "$CUDA_DEVICE" \
        --data_root "$DATA_ROOT" \
        --output "$RESULTS_DIR/llava_vicuna/${ds}.jsonl"
    echo "[LLaVA-Vicuna DECOMP] $ds done."
done

# ============================================================
# Model 3: Qwen3-VL-8B-Instruct
# ============================================================
echo "========== [DECOMP] Qwen3-VL-8B-Instruct =========="
for ds in "${DATASETS[@]}"; do
    echo "[Qwen3-VL DECOMP] Running $ds..."
    python -m experiments.run_decomposition \
        --dataset "$ds" \
        --model qwen_vl \
        --model_path "$QWEN_VL_PATH" \
        --cuda_device "$CUDA_DEVICE" \
        --data_root "$DATA_ROOT" \
        --output "$RESULTS_DIR/qwen_vl/${ds}.jsonl"
    echo "[Qwen3-VL DECOMP] $ds done."
done

# ============================================================
# Model 4: Gemma 4 E4B
# ============================================================
echo "========== [DECOMP] Gemma 4 E4B =========="
for ds in "${DATASETS[@]}"; do
    echo "[Gemma4 DECOMP] Running $ds..."
    python -m experiments.run_decomposition \
        --dataset "$ds" \
        --model gemma_vl \
        --model_path "$GEMMA_VL_PATH" \
        --cuda_device "$CUDA_DEVICE" \
        --data_root "$DATA_ROOT" \
        --output "$RESULTS_DIR/gemma_vl/${ds}.jsonl"
    echo "[Gemma4 DECOMP] $ds done."
done

echo ""
echo "=========================================="
echo "All decomposition runs complete!"
echo "Results in: $RESULTS_DIR/"
echo "=========================================="
ls -la "$RESULTS_DIR/"
