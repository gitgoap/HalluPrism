#!/bin/bash
# ============================================================
# Master Script for running the Rebuttal on 4 GPUs
#
# FULL REBUTTAL WORKFLOW:
#   Step 1: Run inference in 4 tmux panes (one per GPU):
#     tmux pane 0:  bash rebuttal/run_rebuttal_4gpus.sh 0
#     tmux pane 1:  bash rebuttal/run_rebuttal_4gpus.sh 1
#     tmux pane 2:  bash rebuttal/run_rebuttal_4gpus.sh 2
#     tmux pane 3:  bash rebuttal/run_rebuttal_4gpus.sh 3
#
#   Step 2: Wait for all 4 panes to print "GPU X DONE!"
#
#   Step 3: Merge the VizWiz chunks back into single files:
#     bash rebuttal/merge_jsonl_chunks.sh
#
#   Step 4: Run analysis for Experiment 1 (Routing Control):
#     python rebuttal/exp1_routing_control/analyze_routing_control.py --results_dir results/rebuttal_exp1_routing/
#
#   Step 5: Run analysis for Experiment 2 (Perturbation Sensitivity):
#     python analysis/rq3_failure_family.py --results_dir results/rebuttal_exp2_sensitivity/weak/decomposition/ --all_models
#     python analysis/rq3_failure_family.py --results_dir results/rebuttal_exp2_sensitivity/strong/decomposition/ --all_models
#
# ============================================================
#
# Usage (single GPU):  bash rebuttal/run_rebuttal_4gpus.sh <gpu_id>
# e.g., run this command with 0, 1, 2, and 3 in four separate tmux panes.
# ============================================================
GPU_ID=$1

if [ -z "$GPU_ID" ]; then
    echo "Usage: bash rebuttal/run_rebuttal_4gpus.sh <gpu_id> (0, 1, 2, or 3)"
    exit 1
fi

get_model_path() {
    case $1 in
        "qwen_vl") echo "/home/models/Qwen3-VL-8B-Instruct" ;;
        "llava_mistral") echo "/home/models/llava-v1.6-mistral-7b-hf" ;;
        "llava_vicuna") echo "/home/models/llava-v1.6-vicuna-7b-hf" ;;
        "gemma_vl") echo "/home/models/gemma-4-E4B-it" ;;
        *) echo "" ;;
    esac
}

# Define Chunk boundaries for VizWiz (total 4319)
if [ "$GPU_ID" == "0" ]; then
    START=0
    END=1080
elif [ "$GPU_ID" == "1" ]; then
    START=1080
    END=2160
elif [ "$GPU_ID" == "2" ]; then
    START=2160
    END=3240
elif [ "$GPU_ID" == "3" ]; then
    START=3240
    END=4320
fi

echo "=========================================================="
echo " Starting execution on GPU $GPU_ID (VizWiz Chunk: $START to $END)"
echo "=========================================================="

# ---------------------------------------------------------
# EXPERIMENT 1: ROUTING CONTROL (Chunked for VizWiz, Static for others)
# ---------------------------------------------------------
if [ "$GPU_ID" == "0" ]; then
    echo "Running Exp 1: Static Datasets on GPU 0..."
    for setting in "qwen_vl vsr" "gemma_vl hallusionbench" "qwen_vl hallusionbench"; do
        set -- $setting
        model=$1
        dataset=$2
        decomp_file="results/decomposition/${model}/${dataset}.jsonl"
        output_file="results/rebuttal_exp1_routing/${model}/${dataset}.jsonl"
        
        if [ -f "$decomp_file" ]; then
            python rebuttal/exp1_routing_control/run_routing_control.py \
                --decomp_results ${decomp_file} \
                --model ${model} \
                --model_path $(get_model_path $model) \
                --cuda_device cuda:${GPU_ID} \
                --output ${output_file}
        fi
    done
fi

echo "Running Exp 1: VizWiz Chunk on GPU $GPU_ID..."
python rebuttal/exp1_routing_control/run_routing_control.py \
    --decomp_results results/decomposition/llava_mistral/vizwiz.jsonl \
    --model llava_mistral \
    --model_path $(get_model_path llava_mistral) \
    --cuda_device cuda:${GPU_ID} \
    --start_index ${START} \
    --end_index ${END} \
    --output results/rebuttal_exp1_routing/llava_mistral/vizwiz_chunk_${GPU_ID}.jsonl

# ---------------------------------------------------------
# EXPERIMENT 2: SENSITIVITY (Chunked for VizWiz, Static for HallusionBench)
# ---------------------------------------------------------

# Static HallusionBench allocations
if [ "$GPU_ID" == "1" ]; then
    echo "Running Exp 2: HallusionBench for Qwen3-VL on GPU 1..."
    for strength in "weak" "strong"; do
        output_decomp="results/rebuttal_exp2_sensitivity/${strength}/decomposition/qwen_vl/hallusionbench.jsonl"
        output_interv="results/rebuttal_exp2_sensitivity/${strength}/intervention/qwen_vl/hallusionbench.jsonl"
        
        python rebuttal/exp2_perturbation_sensitivity/run_sensitivity_decomposition.py \
            --strength ${strength} \
            --dataset hallusionbench \
            --model qwen_vl \
            --model_path $(get_model_path qwen_vl) \
            --cuda_device cuda:${GPU_ID} \
            --output ${output_decomp}
            
        python experiments/run_matched_intervention.py \
            --decomp_results ${output_decomp} \
            --model qwen_vl \
            --model_path $(get_model_path qwen_vl) \
            --cuda_device cuda:${GPU_ID} \
            --output ${output_interv}
    done
fi

if [ "$GPU_ID" == "2" ]; then
    echo "Running Exp 2: HallusionBench for LLaVA-Mistral on GPU 2..."
    for strength in "weak" "strong"; do
        output_decomp="results/rebuttal_exp2_sensitivity/${strength}/decomposition/llava_mistral/hallusionbench.jsonl"
        output_interv="results/rebuttal_exp2_sensitivity/${strength}/intervention/llava_mistral/hallusionbench.jsonl"
        
        python rebuttal/exp2_perturbation_sensitivity/run_sensitivity_decomposition.py \
            --strength ${strength} \
            --dataset hallusionbench \
            --model llava_mistral \
            --model_path $(get_model_path llava_mistral) \
            --cuda_device cuda:${GPU_ID} \
            --output ${output_decomp}
            
        python experiments/run_matched_intervention.py \
            --decomp_results ${output_decomp} \
            --model llava_mistral \
            --model_path $(get_model_path llava_mistral) \
            --cuda_device cuda:${GPU_ID} \
            --output ${output_interv}
    done
fi

# Distributed VizWiz for Exp 2
echo "Running Exp 2: VizWiz Chunk on GPU $GPU_ID..."
for model in "qwen_vl" "llava_mistral"; do
    for strength in "weak" "strong"; do
        output_decomp="results/rebuttal_exp2_sensitivity/${strength}/decomposition/${model}/vizwiz_chunk_${GPU_ID}.jsonl"
        output_interv="results/rebuttal_exp2_sensitivity/${strength}/intervention/${model}/vizwiz_chunk_${GPU_ID}.jsonl"
        
        python rebuttal/exp2_perturbation_sensitivity/run_sensitivity_decomposition.py \
            --strength ${strength} \
            --dataset vizwiz \
            --model ${model} \
            --model_path $(get_model_path $model) \
            --cuda_device cuda:${GPU_ID} \
            --start_index ${START} \
            --end_index ${END} \
            --output ${output_decomp}
            
        python experiments/run_matched_intervention.py \
            --decomp_results ${output_decomp} \
            --model ${model} \
            --model_path $(get_model_path $model) \
            --cuda_device cuda:${GPU_ID} \
            --start_index ${START} \
            --end_index ${END} \
            --output ${output_interv}
    done
done

echo "=========================================================="
echo " GPU $GPU_ID DONE! "
echo "=========================================================="
