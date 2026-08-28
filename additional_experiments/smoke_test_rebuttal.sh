#!/bin/bash
# Smoke test for Rebuttal Experiments
# This runs the pipeline on just 5 samples to ensure no errors occur.
# Usage: bash rebuttal/smoke_test_rebuttal.sh

GPU_ID=0
MAX_SAMPLES=5

echo "=========================================================="
echo " Starting SMOKE TEST on GPU $GPU_ID ($MAX_SAMPLES samples)"
echo "=========================================================="

get_model_path() {
    case $1 in
        "qwen_vl") echo "/home/models/Qwen3-VL-8B-Instruct" ;;
        "llava_mistral") echo "/home/models/llava-v1.6-mistral-7b-hf" ;;
        "llava_vicuna") echo "/home/models/llava-v1.6-vicuna-7b-hf" ;;
        "gemma_vl") echo "/home/models/gemma-4-E4B-it" ;;
        *) echo "" ;;
    esac
}

# ---------------------------------------------------------
# EXPERIMENT 1 SMOKE TEST (Qwen on VSR)
# ---------------------------------------------------------
echo "Smoke Test Exp 1 (Routing Control): Qwen on VSR..."
decomp_file="results/decomposition/qwen_vl/vsr.jsonl"
output_file="results/rebuttal_exp1_routing/qwen_vl/smoke_test_vsr.jsonl"

if [ -f "$decomp_file" ]; then
    python rebuttal/exp1_routing_control/run_routing_control.py \
        --decomp_results ${decomp_file} \
        --model qwen_vl \
        --model_path $(get_model_path qwen_vl) \
        --cuda_device cuda:${GPU_ID} \
        --max_samples ${MAX_SAMPLES} \
        --output ${output_file}
else
    echo "Warning: Base decomposition file missing. Smoke test for Exp 1 skipped."
fi

# ---------------------------------------------------------
# EXPERIMENT 2 SMOKE TEST (Qwen on VizWiz - Weak)
# ---------------------------------------------------------
echo "Smoke Test Exp 2 (Sensitivity): Qwen on VizWiz (Weak)..."
output_decomp="results/rebuttal_exp2_sensitivity/weak/decomposition/qwen_vl/smoke_test_vizwiz.jsonl"
output_interv="results/rebuttal_exp2_sensitivity/weak/intervention/qwen_vl/smoke_test_vizwiz.jsonl"

python rebuttal/exp2_perturbation_sensitivity/run_sensitivity_decomposition.py \
    --strength weak \
    --dataset vizwiz \
    --model qwen_vl \
    --model_path $(get_model_path qwen_vl) \
    --cuda_device cuda:${GPU_ID} \
    --max_samples ${MAX_SAMPLES} \
    --output ${output_decomp}
    
python experiments/run_matched_intervention.py \
    --decomp_results ${output_decomp} \
    --model qwen_vl \
    --model_path $(get_model_path qwen_vl) \
    --cuda_device cuda:${GPU_ID} \
    --max_samples ${MAX_SAMPLES} \
    --output ${output_interv}

echo "=========================================================="
echo " SMOKE TEST COMPLETED SUCCESSFULLY!"
echo "=========================================================="
