import os
import json
import shutil
import random
import pandas as pd
from pathlib import Path

# Paths
RESULTS_DIR = Path("../results/decomposition")
AUDIT_DIR = Path(".")
IMAGES_DIR = AUDIT_DIR / "images"
os.makedirs(IMAGES_DIR, exist_ok=True)

DATASETS = ["hallusionbench", "pope", "vizwiz", "vsr"]
MODELS = ["qwen_vl", "gemma_vl", "llava_mistral", "llava_vicuna"]

# Target counts per dataset
TARGET_INCORRECT = 35
TARGET_CORRECT_LOW_CONF = 10
TARGET_CORRECT_HIGH_CONF = 5

def load_all_data(dataset_name):
    """Loads all logs for a specific dataset across all 4 models."""
    data = []
    for model in MODELS:
        log_file = RESULTS_DIR / model / f"{dataset_name}.jsonl"
        if not log_file.exists():
            continue
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                row['model'] = model  # Add model name for tracking
                data.append(row)
    return pd.DataFrame(data)

def sample_dataset(df):
    """Samples 50 items according to the requested distribution."""
    # Split by correctness
    incorrect = df[df['is_correct'] == False]
    correct = df[df['is_correct'] == True]
    
    # Calculate median confidence of correct answers to split high/low
    if not correct.empty:
        median_conf = correct['scalar_confidence'].median()
        correct_low = correct[correct['scalar_confidence'] <= median_conf]
        correct_high = correct[correct['scalar_confidence'] > median_conf]
    else:
        correct_low = pd.DataFrame()
        correct_high = pd.DataFrame()

    # Sample safely (handling cases where we might not have enough samples)
    n_inc = min(TARGET_INCORRECT, len(incorrect))
    n_clow = min(TARGET_CORRECT_LOW_CONF, len(correct_low))
    n_chigh = min(TARGET_CORRECT_HIGH_CONF, len(correct_high))
    
    sampled = pd.concat([
        incorrect.sample(n=n_inc, random_state=42),
        correct_low.sample(n=n_clow, random_state=42),
        correct_high.sample(n=n_chigh, random_state=42)
    ])
    
    # If we are short (e.g. not enough incorrect), just fill with random others
    shortfall = 50 - len(sampled)
    if shortfall > 0:
        remaining = df.drop(sampled.index)
        sampled = pd.concat([sampled, remaining.sample(n=shortfall, random_state=42)])
        
    return sampled.sample(frac=1, random_state=42) # Shuffle final 50

def process_and_export():
    master_records = []
    annotator_records = []
    
    # We will assume image paths in JSON are relative to the workspace root.
    workspace_root = Path("D:/LCS2-Internship-work/MultiModal-Uncertainty/Original-Github-Repo/MMUQ") 
    
    for dataset in DATASETS:
        print(f"Processing {dataset}...")
        df = load_all_data(dataset)
        if df.empty:
            print(f"  Warning: No data found for {dataset}")
            continue
            
        sampled_df = sample_dataset(df)
        
        for _, row in sampled_df.iterrows():
            orig_image_path = Path(row['image_path'])
            # Ensure it resolves correctly whether it's absolute or relative
            if not orig_image_path.is_absolute():
                orig_image_path = workspace_root / orig_image_path
                
            # Create a clean unique image name for the annotator
            ext = orig_image_path.suffix if orig_image_path.suffix else '.jpg'
            clean_image_name = f"{row['dataset_name']}_{row['sample_id']}_{row['model']}{ext}".lower()
            clean_image_name = clean_image_name.replace("/", "_").replace("\\", "_")
            dest_image_path = IMAGES_DIR / clean_image_name
            
            # Copy image
            try:
                if orig_image_path.exists():
                    shutil.copy2(orig_image_path, dest_image_path)
                else:
                    print(f"  Missing image: {orig_image_path}")
            except Exception as e:
                print(f"  Error copying {orig_image_path}: {e}")
                
            # Prepare Annotator Row
            annotator_row = {
                "sample_id": row['sample_id'],
                "dataset_name": row['dataset_name'],
                "model": row['model'],
                "image_file": clean_image_name,
                "question": row.get('text_input', ''),
                "gold_answer": row.get('gold_answer', ''),
                "model_answer": row.get('model_prediction', ''),
                "annotator_label": "",
                "annotator_confidence": "",
                "short_justification": ""
            }
            annotator_records.append(annotator_row)
            
            # Prepare Master Row
            master_row = annotator_row.copy()
            master_row.update({
                "original_image_path": row['image_path'],
                "is_correct": row.get('is_correct', ''),
                "scalar_confidence": row.get('scalar_confidence', ''),
                "visual_score_V": row.get('v_score', ''),
                "language_score_L": row.get('lp_score', ''),
                "alignment_score_A": row.get('a_score', '')
            })
            master_records.append(master_row)
            
    # Export
    df_annotator = pd.DataFrame(annotator_records)
    df_master = pd.DataFrame(master_records)
    
    df_annotator.to_csv("annotator_sheet.csv", index=False)
    df_master.to_csv("master_analysis_sheet.csv", index=False)
    
    print("\nDone! Exported 200 samples.")
    print(" - annotator_sheet.csv")
    print(" - master_analysis_sheet.csv")
    print(" - Copied images to human_audit/images/")

if __name__ == "__main__":
    process_and_export()
