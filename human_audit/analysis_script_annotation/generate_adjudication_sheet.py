import pandas as pd
import numpy as np
import glob
import os
from collections import Counter

BASE_DIR = r"C:\Users\aman\.gemini\antigravity\scratch\mllm-uncertainty\human_audit"
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations")
MASTER_FILE = os.path.join(BASE_DIR, "master_analysis_sheet.csv")
OUT_FILE = os.path.join(BASE_DIR, "manual_adjudication_sheet.csv")

def clean_label(label):
    if pd.isna(label): return "UNCLEAR"
    l = str(label).strip().lower()
    if "visual" in l: return "VISUAL"
    if "language" in l or "prior" in l: return "LANGUAGE_PRIOR"
    if "align" in l or "alin" in l: return "ALIGNMENT"
    if "mixed" in l: return "MIXED"
    if "no failure" in l or "none" in l: return "NO_FAILURE"
    return "UNCLEAR"

def main():
    print("Generating Manual Adjudication Sheet...")
    master_df = pd.read_csv(MASTER_FILE)
    
    annotator_files = sorted(glob.glob(os.path.join(ANNOTATIONS_DIR, "*.csv")))
    
    anno_cols = []
    
    for i, f in enumerate(annotator_files):
        df = pd.read_csv(f)
        idx = i + 1
        
        col_name = f'anno_{idx}_label'
        conf_col = f'anno_{idx}_confidence'
        just_col = f'anno_{idx}_justification'
        
        df[col_name] = df['annotator_label'].apply(clean_label)
        df[conf_col] = df['annotator_confidence']
        df[just_col] = df['short_justification']
        
        # Merge safely on BOTH sample_id and model because sample_id is duplicated across models
        master_df = master_df.merge(df[['sample_id', 'model', col_name, conf_col, just_col]], on=['sample_id', 'model'], how='left')
        anno_cols.append(col_name)

    # Find disagreements
    def get_status(row):
        labels = [row[c] for c in anno_cols]
        counts = Counter(labels)
        most_common, max_count = counts.most_common(1)[0]
        
        if max_count >= 2:
            return pd.Series([False, most_common])
        return pd.Series([True, ""]) # Needs adjudication
        
    master_df[['needs_adjudication', 'final_adjudicated_label']] = master_df.apply(get_status, axis=1)
    
    # Filter to only the ones that need adjudication for the user to easily review
    disagreements_df = master_df[master_df['needs_adjudication'] == True].copy()
    
    # Reorder columns to make it easy for the human adjudicator
    front_cols = [
        'sample_id', 'dataset_name', 'model', 'image_file', 'question', 'gold_answer', 'model_answer',
        'final_adjudicated_label', # <--- User will fill this in
        'anno_1_label', 'anno_1_confidence', 'anno_1_justification',
        'anno_2_label', 'anno_2_confidence', 'anno_2_justification',
        'anno_3_label', 'anno_3_confidence', 'anno_3_justification'
    ]
    
    disagreements_df = disagreements_df[front_cols]
    
    disagreements_df.to_csv(OUT_FILE, index=False)
    
    print(f"Total samples: {len(master_df)}")
    print(f"Samples with clear majority: {len(master_df) - len(disagreements_df)}")
    print(f"Samples needing manual adjudication: {len(disagreements_df)}")
    print(f"Saved adjudication sheet to: {OUT_FILE}")

if __name__ == "__main__":
    main()
