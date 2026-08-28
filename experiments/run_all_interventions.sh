#!/bin/bash
# ============================================================
# Run Matched Intervention (RQ5): 4 models x 4 datasets = 16 runs
#
# Reads decomposition results to know each sample's dominant source,
# then applies the matched fix and a generic fix.
#
# Prerequisites: Decomposition results must exist in results/decomposition/
# ============================================================

set -e

DATA_ROOT="data/"
DECOMP_DIR="results/decomposition"
RESULTS_DIR="results/intervention"
CUDA_DEVICE="cuda:2"

LLAVA_MISTRAL_PATH="/home/models/llava-v1.6-mistral-7b-hf"
LLAVA_VICUNA_PATH="/home/models/llava-v1.6-vicuna-7b-hf"
QWEN_VL_PATH="/home/models/Qwen3-VL-8B-Instruct"
GEMMA_VL_PATH="/home/models/gemma-4-E4B-it"

DATASETS=("hallusionbench" "pope" "vsr" "vizwiz")

mkdir -p "$RESULTS_DIR"

run_intervention() {
    local MODEL_NAME=$1
    local MODEL_PATH=$2
    local DS=$3

    local DECOMP_FILE="$DECOMP_DIR/$MODEL_NAME/${DS}.jsonl"

    if [ ! -f "$DECOMP_FILE" ]; then
        echo "  [SKIP] Decomposition file not found: $DECOMP_FILE"
        return
    fi

    echo "  [${MODEL_NAME}] Running matched intervention on ${DS}..."
    python -m experiments.run_matched_intervention \
        --decomp_results "$DECOMP_FILE" \
        --model "$MODEL_NAME" \
        --model_path "$MODEL_PATH" \
        --cuda_device "$CUDA_DEVICE" \
        --data_root "$DATA_ROOT" \
        --output "$RESULTS_DIR/$MODEL_NAME/${DS}.jsonl"
    echo "  [${MODEL_NAME}] ${DS} done."
}

echo "========== LLaVA-Mistral =========="
for ds in "${DATASETS[@]}"; do
    run_intervention "llava_mistral" "$LLAVA_MISTRAL_PATH" "$ds"
done

echo "========== LLaVA-Vicuna =========="
for ds in "${DATASETS[@]}"; do
    run_intervention "llava_vicuna" "$LLAVA_VICUNA_PATH" "$ds"
done

echo "========== Qwen3-VL =========="
for ds in "${DATASETS[@]}"; do
    run_intervention "qwen_vl" "$QWEN_VL_PATH" "$ds"
done

echo "========== Gemma 4 =========="
for ds in "${DATASETS[@]}"; do
    run_intervention "gemma_vl" "$GEMMA_VL_PATH" "$ds"
done

echo ""
echo "========================================================"
echo "All matched intervention runs complete!"
echo "Results in: $RESULTS_DIR/"
echo "========================================================"
ls -la "$RESULTS_DIR/"
