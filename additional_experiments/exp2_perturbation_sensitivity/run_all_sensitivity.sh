#!/bin/bash
# Rebuttal Experiment 2: Small perturbation-strength sensitivity study
# This script runs the weak and strong perturbation pipelines and their analyses.
get_model_path() {
    case $1 in
        "qwen_vl") echo "/home/models/Qwen3-VL-8B-Instruct" ;;
        "llava_mistral") echo "/home/models/llava-v1.6-mistral-7b-hf" ;;
        "llava_vicuna") echo "/home/models/llava-v1.6-vicuna-7b-hf" ;;
        "gemma_vl") echo "/home/models/gemma-4-E4B-it" ;;
        *) echo "" ;;
    esac
}

MODELS=("qwen_vl" "llava_mistral")
DATASETS=("vizwiz" "hallusionbench")
STRENGTHS=("weak" "strong")

echo "=============================================="
echo " Starting Perturbation Sensitivity Experiment"
echo "=============================================="

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for strength in "${STRENGTHS[@]}"; do
            
            output_decomp="results/rebuttal_exp2_sensitivity/${strength}/decomposition/${model}/${dataset}.jsonl"
            output_interv="results/rebuttal_exp2_sensitivity/${strength}/intervention/${model}/${dataset}.jsonl"
            
            ACTUAL_MODEL_PATH=$(get_model_path $model)
            
            echo "Running Decomposition for $model on $dataset with $strength strength..."
            python rebuttal/exp2_perturbation_sensitivity/run_sensitivity_decomposition.py \
                --strength ${strength} \
                --dataset ${dataset} \
                --model ${model} \
                --model_path ${ACTUAL_MODEL_PATH} \
                --cuda_device cuda:0 \
                --data_root data/ \
                --output ${output_decomp}
                
            echo "Running Matched Intervention for $model on $dataset with $strength strength..."
            python experiments/run_matched_intervention.py \
                --decomp_results ${output_decomp} \
                --model ${model} \
                --model_path ${ACTUAL_MODEL_PATH} \
                --cuda_device cuda:0 \
                --data_root data/ \
                --output ${output_interv}
                
        done
    done
done

echo "=============================================="
echo " Running Analyses for Sensitivity Study"
echo "=============================================="

for strength in "${STRENGTHS[@]}"; do
    echo "--- Analysis for ${strength} strength ---"
    python analysis/rq3_failure_family.py \
        --results_dir results/rebuttal_exp2_sensitivity/${strength}/decomposition/ \
        --all_models
        
    python analysis/rq5_intervention.py \
        --results_dir results/rebuttal_exp2_sensitivity/${strength}/intervention/ \
        --all_models
done

echo "Sensitivity study complete. Compare the RQ3 and RQ5 outputs between weak and strong."
