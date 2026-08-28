#!/bin/bash
# ============================================================
# Run Identifiability Study (RQ2) — Week 5
# 3 interventions x 4 models x 4 datasets = 48 total runs
#
# IMPORTANT: Original data in data/ is NEVER touched.
# All interventions operate entirely in RAM.
#
# Results go to:
#   results/identifiability/<intervention>/<model>/<dataset>.jsonl
# ============================================================

set -e

DATA_ROOT="data/"
RESULTS_DIR="results/identifiability"
CUDA_DEVICE="cuda:0"

# Model paths — update to match your server
LLAVA_MISTRAL_PATH="/home/models/llava-v1.6-mistral-7b-hf"
LLAVA_VICUNA_PATH="/home/models/llava-v1.6-vicuna-7b-hf"
QWEN_VL_PATH="/home/models/Qwen3-VL-8B-Instruct"
GEMMA_VL_PATH="/home/models/gemma-4-E4B-it"

DATASETS=("hallusionbench" "pope" "vsr" "vizwiz")
INTERVENTIONS=("visual_ruined" "language_ruined" "alignment_ruined")

mkdir -p "$RESULTS_DIR"

run_identifiability() {
    local MODEL_NAME=$1
    local MODEL_PATH=$2
    local INTERVENTION=$3
    local DS=$4

    echo "  [${MODEL_NAME} | ${INTERVENTION}] Running ${DS}..."
    python -m experiments.run_identifiability \
        --dataset "$DS" \
        --model "$MODEL_NAME" \
        --model_path "$MODEL_PATH" \
        --cuda_device "$CUDA_DEVICE" \
        --intervention "$INTERVENTION" \
        --data_root "$DATA_ROOT" \
        --output "$RESULTS_DIR/$INTERVENTION/$MODEL_NAME/${DS}.jsonl"
    echo "  [${MODEL_NAME} | ${INTERVENTION}] ${DS} done."
}

# ============================================================
# Loop over all 3 interventions
# ============================================================
for INTERVENTION in "${INTERVENTIONS[@]}"; do
    echo ""
    echo "######################################################"
    echo " INTERVENTION: $INTERVENTION"
    echo "######################################################"

    # Model 1: LLaVA-Mistral
    echo "--- LLaVA-Mistral ---"
    for ds in "${DATASETS[@]}"; do
        run_identifiability "llava_mistral" "$LLAVA_MISTRAL_PATH" "$INTERVENTION" "$ds"
    done

    # Model 2: LLaVA-Vicuna
    echo "--- LLaVA-Vicuna ---"
    for ds in "${DATASETS[@]}"; do
        run_identifiability "llava_vicuna" "$LLAVA_VICUNA_PATH" "$INTERVENTION" "$ds"
    done

    # Model 3: Qwen3-VL
    echo "--- Qwen3-VL ---"
    for ds in "${DATASETS[@]}"; do
        run_identifiability "qwen_vl" "$QWEN_VL_PATH" "$INTERVENTION" "$ds"
    done

    # Model 4: Gemma 4
    echo "--- Gemma 4 ---"
    for ds in "${DATASETS[@]}"; do
        run_identifiability "gemma_vl" "$GEMMA_VL_PATH" "$INTERVENTION" "$ds"
    done
done

echo ""
echo "========================================================"
echo "All identifiability runs complete!"
echo "Results in: $RESULTS_DIR/"
echo "========================================================"
ls -la "$RESULTS_DIR/"
