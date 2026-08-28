#!/bin/bash
# Rebuttal Experiment 1: Random/permuted routing control
# This script runs the routing control on the minimal informative settings specified by the reviewer.
get_model_path() {
    case $1 in
        "qwen_vl") echo "/home/models/Qwen3-VL-8B-Instruct" ;;
        "llava_mistral") echo "/home/models/llava-v1.6-mistral-7b-hf" ;;
        "llava_vicuna") echo "/home/models/llava-v1.6-vicuna-7b-hf" ;;
        "gemma_vl") echo "/home/models/gemma-4-E4B-it" ;;
        *) echo "" ;;
    esac
}

echo "=============================================="
echo " Starting Routing Control Experiment"
echo "=============================================="

# Define the 4 minimal settings requested:
# 1. Qwen3-VL/VSR
# 2. Gemma 4/HallusionBench
# 3. LLaVA-Mistral/VizWiz
# 4. Qwen3-VL/HallusionBench (Negative control where matched routing doesn't help as much)

SETTINGS=(
    "qwen_vl vsr"
    "gemma_vl hallusionbench"
    "llava_mistral vizwiz"
    "qwen_vl hallusionbench"
)

for setting in "${SETTINGS[@]}"; do
    set -- $setting
    model=$1
    dataset=$2
    
    decomp_file="results/decomposition/${model}/${dataset}.jsonl"
    output_file="results/rebuttal_exp1_routing/${model}/${dataset}.jsonl"
    
    if [ ! -f "$decomp_file" ]; then
        echo "Warning: Decomposition file $decomp_file not found. Skipping $model on $dataset."
        continue
    fi
    
    ACTUAL_MODEL_PATH=$(get_model_path $model)
    
    echo "Running Routing Control for $model on $dataset..."
    python rebuttal/exp1_routing_control/run_routing_control.py \
        --decomp_results ${decomp_file} \
        --model ${model} \
        --model_path ${ACTUAL_MODEL_PATH} \
        --cuda_device cuda:0 \
        --data_root data/ \
        --output ${output_file}
done

echo "=============================================="
echo " Running Analysis for Routing Control"
echo "=============================================="

python rebuttal/exp1_routing_control/analyze_routing_control.py \
    --results_dir results/rebuttal_exp1_routing/

echo "Routing Control Experiment Complete."
