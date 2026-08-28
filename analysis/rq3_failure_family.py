"""
RQ3: Failure-Family Prediction

Trains 3 classifiers (Logistic Regression, XGBoost, LightGBM) to predict
which failure family a sample belongs to, using:
  - Feature Set A: [scalar_confidence] only
  - Feature Set B: [V, L, A] only
  - Feature Set C: [V, L, A, scalar_confidence]

If Feature Set B/C consistently outperforms A across all 3 classifiers,
that proves source-aware uncertainty is more informative than scalar confidence.

Output:
  - Main comparison table (AUROC per failure family per feature set)
  - Per-classifier breakdown
  - Feature importance from tree-based models

Usage:
    # Single model:
    python analysis/rq3_failure_family.py \
        --results_dir results/decomposition/llava_mistral/

    # All models:
    python analysis/rq3_failure_family.py \
        --results_dir results/decomposition/ \
        --all_models

    # Quick-run (recommended after all experiments are done):
    #   python analysis/rq3_failure_family.py --results_dir results/decomposition/ --all_models
    #
    # Or run everything at once with:
    #   bash analysis/run_all_analysis.sh

   
    pip install scikit-learn xgboost lightgbm
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore", category=UserWarning)


def load_all_results(results_dir: str, model_name: str = None) -> List[dict]:
    """
    Load all decomposition JSONL files for a model (or all models).
    Only keeps samples that have V, L, A scores and a failure family.
    """
    results_path = Path(results_dir)
    all_results = []

    if model_name:
        dirs = [results_path / model_name]
    else:
        dirs = [d for d in results_path.iterdir()
                if d.is_dir() and not d.name.startswith("log_files")]

    for model_dir in dirs:
        for jf in sorted(model_dir.glob("*.jsonl")):
            if jf.name.startswith("test_"):
                continue
            with open(jf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    # Must have decomposition scores and a failure family
                    if (r.get("skipped") or
                        r.get("v_score") is None or
                        r.get("lp_score") is None or
                        r.get("a_score") is None or
                        r.get("failure_family") is None):
                        continue
                    r["_model"] = model_dir.name
                    r["_dataset"] = jf.stem
                    all_results.append(r)

    return all_results


def prepare_features_and_labels(results: List[dict]) -> dict:
    """
    Extract feature matrices and label vectors for classification.

    Returns:
        Dict with feature sets and labels.
    """
    # Feature arrays
    scalar_conf = np.array([r.get("scalar_confidence", 0.0) for r in results]).reshape(-1, 1)
    v_scores = np.array([r.get("v_score", 0.0) for r in results])
    lp_scores = np.array([r.get("lp_score", 0.0) for r in results])
    a_scores = np.array([r.get("a_score", 0.0) for r in results])

    # Feature sets
    X_scalar = scalar_conf                                          # [scalar_conf]
    X_vla = np.column_stack([v_scores, lp_scores, a_scores])        # [V, L, A]
    X_combined = np.column_stack([v_scores, lp_scores, a_scores,    # [V, L, A, scalar]
                                  scalar_conf.ravel()])

    # Labels
    labels = np.array([r["failure_family"] for r in results])

    # Correctness (for filtering to incorrect-only analysis)
    is_correct = np.array([r.get("is_correct", False) for r in results])

    return {
        "X_scalar": X_scalar,
        "X_vla": X_vla,
        "X_combined": X_combined,
        "labels": labels,
        "is_correct": is_correct,
        "feature_names_vla": ["V_score", "LP_score", "A_score"],
        "feature_names_combined": ["V_score", "LP_score", "A_score", "Scalar_conf"],
    }


def evaluate_classifiers(data: dict, use_incorrect_only: bool = False) -> dict:
    """
    Train and evaluate 3 classifiers with 5-fold cross-validation.

    Returns results for each (classifier, feature_set) combination.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.pipeline import Pipeline

    try:
        from xgboost import XGBClassifier
        has_xgb = True
    except ImportError:
        has_xgb = False
        print("  [WARNING] xgboost not installed. Skipping XGBoost.")

    try:
        from lightgbm import LGBMClassifier
        has_lgbm = True
    except ImportError:
        has_lgbm = False
        print("  [WARNING] lightgbm not installed. Skipping LightGBM.")

    # Filter to incorrect samples only if requested
    if use_incorrect_only:
        mask = ~data["is_correct"]
        X_scalar = data["X_scalar"][mask]
        X_vla = data["X_vla"][mask]
        X_combined = data["X_combined"][mask]
        labels = data["labels"][mask]
    else:
        X_scalar = data["X_scalar"]
        X_vla = data["X_vla"]
        X_combined = data["X_combined"]
        labels = data["labels"]

    if len(labels) < 10:
        print("  Too few samples for classification. Skipping.")
        return {}

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = le.classes_

    # Skip if only 1 class
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print("  Only 1 class present. Skipping classification.")
        return {}

    # Cross-validation setup
    n_splits = min(5, min(np.bincount(y)))
    if n_splits < 2:
        n_splits = 2
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    feature_sets = {
        "Scalar Only": X_scalar,
        "V+L+A": X_vla,
        "V+L+A+Scalar": X_combined,
    }

    # Define classifiers (all with default hyperparameters)
    classifiers = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
    }
    if has_xgb:
        classifiers["XGBoost"] = XGBClassifier(
            n_estimators=100, max_depth=4, random_state=42,
            eval_metric="mlogloss", verbosity=0,
        )
    if has_lgbm:
        classifiers["LightGBM"] = LGBMClassifier(
            n_estimators=100, max_depth=4, random_state=42, verbose=-1,
        )

    # Evaluate each combination
    results = {}
    for clf_name, clf in classifiers.items():
        results[clf_name] = {}
        for feat_name, X in feature_sets.items():
            try:
                scores = cross_val_score(
                    clf, X, y, cv=cv, scoring="roc_auc_ovr_weighted"
                )
                results[clf_name][feat_name] = {
                    "auroc_mean": float(np.mean(scores)),
                    "auroc_std": float(np.std(scores)),
                }
            except Exception as e:
                results[clf_name][feat_name] = {
                    "auroc_mean": 0.0,
                    "auroc_std": 0.0,
                    "error": str(e),
                }

    # Feature importance from tree models (trained on full data)
    feature_importance = {}
    if has_xgb:
        try:
            xgb = XGBClassifier(n_estimators=100, max_depth=4, random_state=42,
                                eval_metric="mlogloss", verbosity=0)
            xgb.fit(X_combined, y)
            importance = xgb.feature_importances_
            feature_importance["XGBoost"] = dict(
                zip(data["feature_names_combined"], importance.tolist())
            )
        except Exception:
            pass

    if has_lgbm:
        try:
            lgbm = LGBMClassifier(n_estimators=100, max_depth=4,
                                   random_state=42, verbose=-1)
            lgbm.fit(X_combined, y)
            importance = lgbm.feature_importances_ / lgbm.feature_importances_.sum()
            feature_importance["LightGBM"] = dict(
                zip(data["feature_names_combined"], importance.tolist())
            )
        except Exception:
            pass

    return {
        "classification_results": results,
        "feature_importance": feature_importance,
        "class_names": class_names.tolist(),
        "n_samples": len(y),
        "class_distribution": {
            name: int(count) for name, count in
            zip(class_names, np.bincount(y))
        },
    }


def print_results(eval_results: dict, title: str = ""):
    """Pretty-print the RQ3 comparison table."""
    if not eval_results:
        return

    print(f"\n  {'='*75}")
    print(f"  RQ3 FAILURE-FAMILY PREDICTION{(' — ' + title) if title else ''}")
    print(f"  {'='*75}")

    print(f"\n  Samples: {eval_results['n_samples']}")
    print(f"  Classes: {eval_results['class_names']}")
    print(f"  Distribution: {eval_results['class_distribution']}")

    # Main comparison table
    clf_results = eval_results["classification_results"]
    print(f"\n  AUROC Comparison (weighted, {len(clf_results)} classifiers):")
    print(f"  {'Classifier':<22s} {'Scalar Only':>14s} {'V+L+A':>14s} {'V+L+A+Scalar':>14s}")
    print(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*14}")

    for clf_name, feat_results in clf_results.items():
        cols = []
        for feat_name in ["Scalar Only", "V+L+A", "V+L+A+Scalar"]:
            r = feat_results.get(feat_name, {})
            if "error" in r:
                cols.append("ERROR")
            else:
                cols.append(f"{r['auroc_mean']:.3f}+/-{r['auroc_std']:.3f}")
        print(f"  {clf_name:<22s} {cols[0]:>14s} {cols[1]:>14s} {cols[2]:>14s}")

    # Highlight the key finding
    print(f"\n  Key Question: Does V+L+A outperform Scalar Only across ALL classifiers?")
    all_better = True
    for clf_name, feat_results in clf_results.items():
        scalar = feat_results.get("Scalar Only", {}).get("auroc_mean", 0)
        vla = feat_results.get("V+L+A", {}).get("auroc_mean", 0)
        delta = vla - scalar
        status = "YES" if delta > 0 else "NO"
        if delta <= 0:
            all_better = False
        print(f"    {clf_name:<22s}: V+L+A = {vla:.3f}, Scalar = {scalar:.3f}, "
              f"delta = {delta:+.3f} -> {status}")

    verdict = "CONFIRMED" if all_better else "NOT CONFIRMED"
    print(f"\n  Verdict: Source-aware > Scalar across all classifiers: {verdict}")

    # Feature importance
    if eval_results.get("feature_importance"):
        print(f"\n  Feature Importance (from tree-based classifiers):")
        for clf_name, importance in eval_results["feature_importance"].items():
            print(f"    {clf_name}:")
            for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
                bar = "█" * int(imp * 40)
                print(f"      {feat:<15s} {imp:.3f} {bar}")


def main():
    parser = argparse.ArgumentParser(description="RQ3: Failure-family prediction.")
    parser.add_argument("--results_dir", required=True,
                        help="Path to decomposition results directory")
    parser.add_argument("--model", default=None,
                        help="Specific model to analyze")
    parser.add_argument("--all_models", action="store_true",
                        help="Analyze all models (pooled)")
    parser.add_argument("--incorrect_only", action="store_true",
                        help="Only use incorrect predictions for classification")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    analysis_dir = results_dir / "analysis" / "rq3"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    from analysis.tee_output import TeeOutput
    tee = TeeOutput(str(analysis_dir / "full_report.txt"))
    tee.__enter__()

    if args.all_models:
        model_dirs = sorted([
            d.name for d in results_dir.iterdir()
            if d.is_dir() and not d.name.startswith(("log_files", "analysis"))
        ])
    elif args.model:
        model_dirs = [args.model]
    else:
        print("Error: specify --model or --all_models")
        return

    # Per-model analysis
    all_rows = []
    for model_name in model_dirs:
        print(f"\n{'#'*75}")
        print(f"  MODEL: {model_name}")
        print(f"{'#'*75}")

        results = load_all_results(str(results_dir), model_name)
        if not results:
            print(f"  No valid results found for {model_name}")
            continue

        print(f"  Loaded {len(results)} samples with V/L/A scores")

        data = prepare_features_and_labels(results)
        eval_results = evaluate_classifiers(data, use_incorrect_only=args.incorrect_only)
        print_results(eval_results, title=model_name)

        # Collect rows for CSV
        if eval_results and "classification_results" in eval_results:
            for clf_name, feat_results in eval_results["classification_results"].items():
                for feat_name, metrics in feat_results.items():
                    all_rows.append({
                        "model": model_name,
                        "classifier": clf_name,
                        "features": feat_name,
                        "auroc_mean": metrics.get("auroc_mean", 0),
                        "auroc_std": metrics.get("auroc_std", 0),
                    })

    # Pooled analysis across all models
    if len(model_dirs) > 1:
        print(f"\n{'#'*75}")
        print(f"  POOLED ANALYSIS (all models combined)")
        print(f"{'#'*75}")

        all_results = load_all_results(str(results_dir))
        if all_results:
            print(f"  Loaded {len(all_results)} total samples")
            data = prepare_features_and_labels(all_results)
            eval_results = evaluate_classifiers(data, use_incorrect_only=args.incorrect_only)
            print_results(eval_results, title="ALL MODELS POOLED")

            if eval_results and "classification_results" in eval_results:
                for clf_name, feat_results in eval_results["classification_results"].items():
                    for feat_name, metrics in feat_results.items():
                        all_rows.append({
                            "model": "POOLED",
                            "classifier": clf_name,
                            "features": feat_name,
                            "auroc_mean": metrics.get("auroc_mean", 0),
                            "auroc_std": metrics.get("auroc_std", 0),
                        })

    # Save results CSV
    if all_rows:
        import csv
        csv_path = analysis_dir / "classifier_comparison.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "classifier", "features",
                                                    "auroc_mean", "auroc_std"])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n  RQ3 results saved -> {csv_path}")

    tee.__exit__(None, None, None)
    print(f"Full report saved -> {analysis_dir / 'full_report.txt'}")


if __name__ == "__main__":
    main()

