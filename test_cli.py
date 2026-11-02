#!/usr/bin/env python3
"""
CLI Test Utility for Decision Tree Classification & Explainability
Run directly in the terminal to inspect datasets, train trees, and test sample predictions.
"""

import sys
import os

# Auto-activate workspace virtual environment if run with global system python
VENV_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
if os.path.exists(VENV_PYTHON) and sys.executable != VENV_PYTHON:
    try:
        import pandas
    except ImportError:
        os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text


def run_pipeline(csv_path: str, target_col: str = None, max_depth: int = 3, sample_dict: dict = None):
    print("=" * 68)
    print(f" 🌳 DECISION TREE CLI STUDIO: {csv_path} ")
    print("=" * 68)

    # 1. Load CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Could not read {csv_path}: {e}")
        sys.exit(1)

    print(f"\n[1] DATASET OVERVIEW")
    print(f"    • Total Rows:     {df.shape[0]}")
    print(f"    • Total Columns:  {df.shape[1]}")
    print(f"    • Columns List:   {list(df.columns)}")

    # Auto-detect target if not provided
    if target_col is None:
        target_col = df.columns[-1]
    
    if target_col not in df.columns:
        print(f"[ERROR] Target column '{target_col}' not found in {list(df.columns)}")
        sys.exit(1)

    feature_cols = [c for c in df.columns if c != target_col]
    print(f"    • Target Column:  '{target_col}'")
    print(f"    • Feature Count:  {len(feature_cols)}")

    # 2. Preprocessing
    X_raw = df[feature_cols].copy()
    y_raw = df[target_col].copy()

    # Drop missing target rows
    valid_idx = y_raw.dropna().index
    X_raw = X_raw.loc[valid_idx]
    y_raw = y_raw.loc[valid_idx]

    # Target encoding
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y_raw.astype(str))
    class_names = [str(c) for c in target_le.classes_]
    print(f"    • Target Classes: {class_names}")

    # Feature imputation & encoding
    encoders = {}
    feature_types = {}
    X_processed = pd.DataFrame(index=X_raw.index)

    for col in feature_cols:
        series = X_raw[col]
        is_cat = not pd.api.types.is_numeric_dtype(series)

        if is_cat:
            mode_val = series.mode()[0] if not series.mode().empty else "Missing"
            imputed = series.fillna(mode_val).astype(str)
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(imputed)
            encoders[col] = le
            feature_types[col] = "categorical"
        else:
            median_val = float(series.median()) if not np.isnan(series.median()) else 0.0
            X_processed[col] = series.fillna(median_val).astype(float)
            feature_types[col] = "numeric"

    # 3. Train/Test Split & Modeling
    stratify_opt = y_encoded if min(np.bincount(y_encoded)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=0.2, random_state=42, stratify=stratify_opt
    )

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    print("\n" + "=" * 68)
    print(f"[2] MODEL PERFORMANCE & PRECISION / RECALL COMPARISON")
    print("=" * 68)
    print(f"    • Out-of-Sample Holdout Accuracy: {accuracy * 100:.2f}% (max_depth={max_depth})")
    print(f"    • Training Samples: {len(X_train)} | Holdout Test Samples: {len(X_test)}")
    
    # Side-by-side comparative table
    hdr_class = "Class Label"
    hdr_p = "Precision"
    hdr_r = "Recall"
    hdr_f1 = "F1-Score"
    hdr_s = "Support"

    print("\n    ⚖️  SIDE-BY-SIDE CLASS PERFORMANCE BREAKDOWN:")
    print("    " + "-" * 62)
    print(f"    {hdr_class:<20} | {hdr_p:<10} | {hdr_r:<10} | {hdr_f1:<8} | {hdr_s}")
    print("    " + "-" * 62)
    for c in class_names:
        p_val = report_dict[c]["precision"] * 100
        r_val = report_dict[c]["recall"] * 100
        f1_val = report_dict[c]["f1-score"]
        supp = int(report_dict[c]["support"])
        print(f"    {c:<20} | {p_val:>8.1f}% | {r_val:>8.1f}% | {f1_val:>8.2f} | {supp:>7d}")
    print("    " + "-" * 62)

    if "macro avg" in report_dict:
        mp = report_dict["macro avg"]["precision"] * 100
        mr = report_dict["macro avg"]["recall"] * 100
        mf1 = report_dict["macro avg"]["f1-score"]
        print(f"    {'Macro Average':<20} | {mp:>8.1f}% | {mr:>8.1f}% | {mf1:>8.2f} | {len(y_test):>7d}")
        print("    " + "-" * 62)

    # Comparison takeaways for binary classification
    if len(class_names) == 2:
        c0, c1 = class_names[0], class_names[1]
        p0, p1 = report_dict[c0]["precision"] * 100, report_dict[c1]["precision"] * 100
        r0, r1 = report_dict[c0]["recall"] * 100, report_dict[c1]["recall"] * 100
        print(f"\n    🔍 Precision vs. Recall Comparison:")
        print(f"       • Precision (Purity) : {c0} = {p0:.1f}%  vs  {c1} = {p1:.1f}%")
        print(f"       • Recall (Coverage)  : {c0} = {r0:.1f}%  vs  {c1} = {r1:.1f}%")
        higher_recall = c0 if r0 > r1 else c1
        lower_recall = c1 if r0 > r1 else c0
        print(f"       • Takeaway: The model detects '{higher_recall}' with higher sensitivity ({max(r0, r1):.1f}%),")
        print(f"         while '{lower_recall}' achieves {min(r0, r1):.1f}% coverage.")

    print("\n    Confusion Matrix (Actual Rows vs Predicted Columns):")
    hdr_cm = "  ".join([f"Pred_{str(c)[:6]:>7}" for c in class_names])
    print(f"                 {hdr_cm}")
    for i, row in enumerate(cm):
        row_str = "  ".join([f"{val:>12d}" for val in row])
        print(f"      Actual_{str(class_names[i])[:6]:<6}: {row_str}")

    # 4. Text Decision Tree Architecture
    print("-" * 50)
    print("[3] DECISION TREE LOGIC FLOWCHART")
    print("-" * 50)
    tree_text = export_text(clf, feature_names=feature_cols)
    print(tree_text)

    # 5. Live Sample Prediction & Step-by-Step Rule Traversal
    print("-" * 50)
    print("[4] LIVE SAMPLE TEST & RULE EXPLAINABILITY")
    print("-" * 50)

    # If no sample provided, pick the first row of test set
    if sample_dict is None:
        sample_row = X_test.iloc[0:1]
        print(f"    Testing with Sample Row #1 from Holdout:")
    else:
        sample_dict_clean = {}
        for col in feature_cols:
            if col in sample_dict:
                val = sample_dict[col]
                if feature_types[col] == "categorical":
                    val = encoders[col].transform([str(val)])[0]
                else:
                    val = float(val)
                sample_dict_clean[col] = val
            else:
                sample_dict_clean[col] = float(X_processed[col].median())
        sample_row = pd.DataFrame([sample_dict_clean])
        print(f"    Testing with Custom Patient Input:")

    for col in feature_cols:
        print(f"      • {col:18s} = {sample_row[col].values[0]}")

    pred_idx = clf.predict(sample_row)[0]
    pred_label = class_names[pred_idx]
    probs = clf.predict_proba(sample_row)[0]
    confidence = probs[pred_idx] * 100

    print(f"\n    🎯 PREDICTION RESULT: {pred_label} (Confidence: {confidence:.1f}%)")
    print("    🔍 RULE SEQUENCE TRAVERSED:")

    node_indicator = clf.decision_path(sample_row)
    leaf_id = clf.apply(sample_row)[0]
    node_indices = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

    for step_i, node_id in enumerate(node_indices):
        if node_id == leaf_id:
            break
        feat_idx = clf.tree_.feature[node_id]
        feat_name = feature_cols[feat_idx]
        threshold = clf.tree_.threshold[node_id]
        actual_val = sample_row.iloc[0, feat_idx]
        op = "<=" if actual_val <= threshold else ">"
        branch = "LEFT" if actual_val <= threshold else "RIGHT"
        print(f"      Step {step_i + 1}: {feat_name} ({actual_val}) {op} {threshold:.2f}  [{branch} branch taken]")

    print("\n" + "=" * 68)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Decision Tree on any dataset in the terminal.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="sample_datasets/diabetes.csv",
        help="Path to CSV dataset (default: sample_datasets/diabetes.csv)"
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target column name (default: last column of CSV)"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Decision Tree max_depth (default: 3)"
    )
    args = parser.parse_args()

    run_pipeline(csv_path=args.dataset, target_col=args.target, max_depth=args.depth)
