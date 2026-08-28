#!/bin/bash
# Merge VizWiz chunked JSONL files back into a single file for analysis
#
# WHEN TO RUN: After all 4 GPU panes from run_rebuttal_4gpus.sh announce "GPU X DONE!"
#   bash rebuttal/merge_jsonl_chunks.sh
#
# WHAT TO DO NEXT (run the analysis):
#   python rebuttal/exp1_routing_control/analyze_routing_control.py --results_dir results/rebuttal_exp1_routing/
#   python analysis/rq3_failure_family.py --results_dir results/rebuttal_exp2_sensitivity/weak/decomposition/ --all_models
#   python analysis/rq3_failure_family.py --results_dir results/rebuttal_exp2_sensitivity/strong/decomposition/ --all_models

echo "Merging Exp 1 Routing Control chunks..."
cat results/rebuttal_exp1_routing/llava_mistral/vizwiz_chunk_*.jsonl > results/rebuttal_exp1_routing/llava_mistral/vizwiz.jsonl
rm results/rebuttal_exp1_routing/llava_mistral/vizwiz_chunk_*.jsonl

echo "Merging Exp 2 Sensitivity chunks..."
for model in "qwen_vl" "llava_mistral"; do
    for strength in "weak" "strong"; do
        # Decomposition
        cat results/rebuttal_exp2_sensitivity/${strength}/decomposition/${model}/vizwiz_chunk_*.jsonl > results/rebuttal_exp2_sensitivity/${strength}/decomposition/${model}/vizwiz.jsonl
        rm results/rebuttal_exp2_sensitivity/${strength}/decomposition/${model}/vizwiz_chunk_*.jsonl
        
        # Intervention
        cat results/rebuttal_exp2_sensitivity/${strength}/intervention/${model}/vizwiz_chunk_*.jsonl > results/rebuttal_exp2_sensitivity/${strength}/intervention/${model}/vizwiz.jsonl
        rm results/rebuttal_exp2_sensitivity/${strength}/intervention/${model}/vizwiz_chunk_*.jsonl
    done
done

echo "Chunks merged! You can now run the analysis scripts:"
echo "python rebuttal/exp1_routing_control/analyze_routing_control.py --results_dir results/rebuttal_exp1_routing/"
echo "python analysis/rq3_failure_family.py --results_dir results/rebuttal_exp2_sensitivity/weak/decomposition/ --all_models"
