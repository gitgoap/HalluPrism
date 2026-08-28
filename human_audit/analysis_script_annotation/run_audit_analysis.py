import pandas as pd
import numpy as np
import glob
import os
from collections import Counter

BASE_DIR = r"C:\Users\aman\.gemini\antigravity\scratch\mllm-uncertainty\human_audit"
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations")
MASTER_FILE = os.path.join(BASE_DIR, "master_analysis_sheet.csv")

VALID_LABELS = ["VISUAL", "LANGUAGE_PRIOR", "ALIGNMENT", "MIXED", "UNCLEAR", "NO_FAILURE"]

def clean_label(label):
    if pd.isna(label): return "UNCLEAR"
    l = str(label).strip().lower()
    if "visual" in l: return "VISUAL"
    if "language" in l or "prior" in l: return "LANGUAGE_PRIOR"
    if "align" in l or "alin" in l: return "ALIGNMENT"
    if "mixed" in l: return "MIXED"
    if "no failure" in l or "none" in l: return "NO_FAILURE"
    return "UNCLEAR"

def fleiss_kappa(ratings_matrix, n_categories):
    N = ratings_matrix.shape[0]
    n = np.sum(ratings_matrix[0])
    
    p_j = np.sum(ratings_matrix, axis=0) / (N * n)
    P_i = (np.sum(ratings_matrix * ratings_matrix, axis=1) - n) / (n * (n - 1))
    
    P_bar = np.sum(P_i) / N
    P_e_bar = np.sum(p_j * p_j)
    
    if P_e_bar == 1: return 1.0
    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
    return kappa

def compute_kappas(df, annotator_cols):
    label_to_idx = {l: i for i, l in enumerate(VALID_LABELS)}
    ratings_full = np.zeros((len(df), len(VALID_LABELS)))
    ratings_collapsed = np.zeros((len(df), 2))
    
    for i, row in df.iterrows():
        counts = Counter([row[c] for c in annotator_cols])
        for label, count in counts.items():
            ratings_full[i, label_to_idx[label]] += count
            if label in ["VISUAL", "LANGUAGE_PRIOR", "ALIGNMENT"]:
                ratings_collapsed[i, 0] += count
            else:
                ratings_collapsed[i, 1] += count
                
    kappa_full = fleiss_kappa(ratings_full, len(VALID_LABELS))
    kappa_collapsed = fleiss_kappa(ratings_collapsed, 2)
    return kappa_full, kappa_collapsed

def adjudicate(row, annotator_cols, manual_labels_dict):
    labels = [row[c] for c in annotator_cols]
    counts = Counter(labels)
    most_common, max_count = counts.most_common(1)[0]
    if max_count >= 2:
        return most_common
        
    # Tie-break: check manual labels
    key = (row['sample_id'], row['model'])
    if key in manual_labels_dict and not pd.isna(manual_labels_dict[key]):
        return clean_label(manual_labels_dict[key])
        
    return "UNCLEAR"

def normalize_sources(df):
    def z_score(series):
        if series.std() == 0: return np.zeros_like(series)
        return (series - series.mean()) / series.std()

    df['z_V'] = df.groupby(['dataset_name', 'model'])['visual_score_V'].transform(z_score)
    df['z_L'] = df.groupby(['dataset_name', 'model'])['language_score_L'].transform(z_score)
    df['z_A'] = df.groupby(['dataset_name', 'model'])['alignment_score_A'].transform(z_score)
    
    sources = ["VISUAL", "LANGUAGE_PRIOR", "ALIGNMENT"]
    
    def get_dominant(row, cols):
        vals = [row[c] for c in cols]
        return sources[np.argmax(vals)]
        
    df['raw_dominant'] = df.apply(lambda r: get_dominant(r, ['visual_score_V', 'language_score_L', 'alignment_score_A']), axis=1)
    df['normalized_dominant'] = df.apply(lambda r: get_dominant(r, ['z_V', 'z_L', 'z_A']), axis=1)
    return df

def my_metrics(y_true, y_pred, labels):
    cm = pd.DataFrame(0, index=[f"Human {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
    for t, p in zip(y_true, y_pred):
        cm.loc[f"Human {t}", f"Pred {p}"] += 1
        
    res = {}
    for l in labels:
        tp = cm.loc[f"Human {l}", f"Pred {l}"]
        fp = cm[f"Pred {l}"].sum() - tp
        fn = cm.loc[f"Human {l}"].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        res[l] = {"Precision": precision, "Recall": recall, "F1": f1}
        
    return cm, pd.DataFrame(res).T

def main():
    print("Loading Master Sheet...")
    master_df = pd.read_csv(MASTER_FILE)
    
    annotator_files = sorted(glob.glob(os.path.join(ANNOTATIONS_DIR, "*.csv")))
    annotator_cols = []
    
    for i, f in enumerate(annotator_files):
        df = pd.read_csv(f)
        col_name = f"anno_{i}_label"
        annotator_cols.append(col_name)
        # Direct assignment since rows are perfectly aligned (200 rows each)
        master_df[col_name] = df['annotator_label'].apply(clean_label)
        
    print(f"Merged Dataset Shape: {master_df.shape}")
        
    k_full, k_coll = compute_kappas(master_df, annotator_cols)
    print(f"\n[Inter-Annotator Agreement - {len(annotator_files)} Annotators]")
    print(f"Fleiss' Kappa (Full 6-Labels): {k_full:.3f}")
    print(f"Fleiss' Kappa (Collapsed Source vs Non-Source): {k_coll:.3f}")
    
    manual_sheet_path = os.path.join(BASE_DIR, "full_manual_adjudication_sheet.csv")
    manual_labels_dict = {}
    if os.path.exists(manual_sheet_path):
        try:
            man_df = pd.read_csv(manual_sheet_path)
            for _, r in man_df.iterrows():
                manual_labels_dict[(r['sample_id'], r['model'])] = r['final_adjudicated_label']
            print(f"Loaded {len(manual_labels_dict)} manual adjudications from {manual_sheet_path}")
        except Exception as e:
            print(f"Warning: Could not read manual sheet: {e}")
            
    master_df['adjudicated_label'] = master_df.apply(lambda r: adjudicate(r, annotator_cols, manual_labels_dict), axis=1)
    
    print(f"\n[Adjudicated Label Distribution]")
    print(master_df['adjudicated_label'].value_counts())
    
    master_df = normalize_sources(master_df)
    
    clear_df = master_df[master_df['adjudicated_label'].isin(["VISUAL", "LANGUAGE_PRIOR", "ALIGNMENT"])]
    y_true = clear_df['adjudicated_label'].values
    y_pred_norm = clear_df['normalized_dominant'].values
    
    print(f"\n[Evaluation on Clear-Source Examples (N={len(clear_df)})]")
    labels = ["VISUAL", "LANGUAGE_PRIOR", "ALIGNMENT"]
    
    print("\n--- Normalized Dominant Source ---")
    cm, metrics = my_metrics(y_true, y_pred_norm, labels)
    print("Confusion Matrix:")
    print(cm)
    print("\nMetrics:")
    print(metrics.round(3))
    
    out_path = os.path.join(BASE_DIR, "adjudicated_audit_results.csv")
    master_df.to_csv(out_path, index=False)
    print(f"\nSaved adjudicated dataset to: {out_path}")

if __name__ == "__main__":
    main()
