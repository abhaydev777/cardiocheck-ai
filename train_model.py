"""
Train a Decision Tree Classifier for Heart Disease Risk Prediction.
Generates model artifact heart_disease_model.pkl and visualization heart_tree.png.
"""

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

from data_loader import COLUMNS, get_dataset

MODEL_PATH = "heart_disease_model.pkl"
TREE_IMAGE_PATH = "heart_tree.png"

FEATURE_COLUMNS = [c for c in COLUMNS if c != "target"]
CLASS_NAMES = ["Healthy (Low Risk)", "Heart Disease (High Risk)"]


def train():
    print("=" * 60)
    print(" HEART DISEASE RISK CLASSIFICATION - MODEL TRAINING ")
    print("=" * 60)

    # 1. Load data
    df = get_dataset()
    X = df[FEATURE_COLUMNS]
    y = df["target"]

    print(f"\n[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    print(f"[INFO] Target distribution:\n{y.value_counts(normalize=True).rename({0: 'Healthy (0)', 1: 'Disease (1)'})}")

    # 2. Train/Test Split (80/20 with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[INFO] Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # 3. Train Decision Tree Classifier (max_depth=4 for maximum explainability)
    clf = DecisionTreeClassifier(
        max_depth=4,
        criterion="gini",
        random_state=42,
    )
    clf.fit(X_train, y_train)
    # Store feature names inside the estimator for easy downstream reference
    clf.feature_names_in_ = np.array(FEATURE_COLUMNS)

    # 4. Evaluation
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=["Healthy (0)", "Disease (1)"])

    print("\n" + "-" * 50)
    print(f" MODEL PERFORMANCE METRICS")
    print("-" * 50)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(f"[[TN={cm[0, 0]:3d}, FP={cm[0, 1]:3d}]")
    print(f" [FN={cm[1, 0]:3d}, TP={cm[1, 1]:3d}]]")
    print("\nClassification Report:")
    print(cr)

    # Feature Importances
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print("Feature Importances:")
    for feat, imp in importances.items():
        if imp > 0.001:
            print(f"  - {feat:10s}: {imp:.4f}")

    # 5. Generate High-Resolution Decision Tree Visualization
    print(f"\n[INFO] Rendering decision tree diagram to {TREE_IMAGE_PATH}...")
    plt.figure(figsize=(26, 13), dpi=300)
    plot_tree(
        clf,
        feature_names=FEATURE_COLUMNS,
        class_names=CLASS_NAMES,
        filled=True,
        rounded=True,
        impurity=True,
        proportion=True,
        precision=2,
        fontsize=9,
    )
    plt.title(
        "Explainable Decision Tree Architecture - Heart Disease Risk Classifier (max_depth=4)",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    plt.savefig(TREE_IMAGE_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] High-resolution tree diagram saved at {TREE_IMAGE_PATH}")

    # 6. Save Model to heart_disease_model.pkl
    joblib.dump(clf, MODEL_PATH)
    print(f"[SUCCESS] Trained model saved to {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train()

