import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, log_loss, confusion_matrix)

# 1. Load the dataset
out_dir = Path(__file__).resolve().parent
data = load_breast_cancer()
X, y = data.data, data.target
names = data.target_names

print("Dataset shape:", X.shape)
print("Classes:", names)
print("Class counts (malignant, benign):", np.bincount(y))

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training samples:", len(y_train))
print("Testing samples :", len(y_test))

# 3. Create the four classifiers
models = {
    "SVM": Pipeline([
        ("scale", StandardScaler()),
        ("model", SVC(kernel="rbf", C=1.0, probability=True, random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "KNN": Pipeline([
        ("scale", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Naive Bayes": GaussianNB()
}

results = {}
conf_mats = {}

# 4. Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = np.clip(model.predict_proba(X_test), 1e-15, 1 - 1e-15)

    # Benign (label 1) is the positive class
    results[name] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, pred, pos_label=1, zero_division=0)),
        "log_loss": float(log_loss(y_test, proba, labels=model.classes_))
    }
    conf_mats[name] = confusion_matrix(y_test, pred, labels=[0, 1]).tolist()

# 5. Display metrics and confusion matrices
print("\nCLASSIFICATION PERFORMANCE (positive class = benign)")
print("Model           Accuracy  Precision  Recall   F1-score  Log loss")
for name, values in results.items():
    print(f"{name:<15} {values['accuracy']:.4f}    {values['precision']:.4f}     "
          f"{values['recall']:.4f}   {values['f1']:.4f}    {values['log_loss']:.4f}")

print("\nCONFUSION MATRICES (rows = actual; columns = predicted; labels=[malignant, benign])")
for name, cm in conf_mats.items():
    print(f"\n{name}:\n{np.array(cm)}")

# 6. Plot confusion matrices
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
for ax, (name, cm) in zip(axes, conf_mats.items()):
    cm = np.array(cm)
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(name)
    ax.set_xticks([0, 1], labels=["Malignant", "Benign"], rotation=20)
    ax.set_yticks([0, 1], labels=["Malignant", "Benign"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
fig.suptitle("Confusion Matrices of Classification Models")
fig.tight_layout()
fig.savefig(out_dir / "confusion_matrices.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# 7. Plot comparison of accuracy, precision, recall and F1-score
metric_names = ["accuracy", "precision", "recall", "f1"]
model_names = list(results.keys())
x = np.arange(len(model_names))
width = 0.19
fig, ax = plt.subplots(figsize=(9, 4.8))
for i, metric in enumerate(metric_names):
    vals = [results[name][metric] for name in model_names]
    ax.bar(x + (i - 1.5) * width, vals, width, label=metric.capitalize())
ax.set_xticks(x, labels=model_names)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Performance Comparison of Classifiers")
ax.legend(ncol=4)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(out_dir / "metrics_comparison.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# 8. Plot log loss comparison
fig, ax = plt.subplots(figsize=(7, 4.5))
losses = [results[name]["log_loss"] for name in model_names]
ax.bar(model_names, losses)
ax.set_ylabel("Log loss (lower is better)")
ax.set_title("Log Loss Comparison")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(out_dir / "log_loss_comparison.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# 9. Save machine-readable results
with open(out_dir / "results.json", "w", encoding="utf-8") as f:
    json.dump({
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "class_counts_malignant_benign": np.bincount(y).tolist(),
        "positive_class": "benign",
        "results": results,
        "confusion_matrices_rows_actual_cols_predicted_labels_malignant_benign": conf_mats
    }, f, indent=2)

print("\nSaved: confusion_matrices.png, metrics_comparison.png, log_loss_comparison.png, results.json")
