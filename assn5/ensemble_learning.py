"""Practical Assignment 5: Ensemble Learning and Performance Evaluation."""

import inspect
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)
# Save generated tables and figures beside this script.
os.chdir(Path(__file__).resolve().parent)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# 1. Load the built-in dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")
class_names = ["Malignant", "Benign"]

print("=" * 64)
print("ENSEMBLE LEARNING AND PERFORMANCE EVALUATION")
print("=" * 64)
print("Dataset shape:", X.shape)
print("Number of samples:", len(X))
print("Number of features:", X.shape[1])
print("Class distribution:")
print(y.value_counts().sort_index().rename(index={0: "Malignant", 1: "Benign"}))
print("Missing values:", int(X.isnull().sum().sum()))

# 2. Use one stratified split for every model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 3. Define base classifiers
log_reg = make_pipeline(
    StandardScaler(), LogisticRegression(max_iter=5000, random_state=42)
)
decision_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
random_forest = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# 4. Define stacking ensemble
stacking_model = StackingClassifier(
    estimators=[
        ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=42))),
        ("dt", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ],
    final_estimator=LogisticRegression(max_iter=5000, random_state=42),
    cv=5,
    n_jobs=-1,
)

# 5. Define AdaBoost with a shallow decision tree as the weak learner
stump = DecisionTreeClassifier(max_depth=1, random_state=42)
ada_params = {
    "n_estimators": 50,
    "learning_rate": 1.0,
    "random_state": 42,
}
if "estimator" in inspect.signature(AdaBoostClassifier).parameters:
    adaboost_model = AdaBoostClassifier(estimator=stump, **ada_params)
else:
    adaboost_model = AdaBoostClassifier(base_estimator=stump, **ada_params)

# 6. Define hard voting ensemble
voting_model = VotingClassifier(
    estimators=[
        ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=42))),
        ("dt", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ],
    voting="hard",
    n_jobs=-1,
)

# 7. Combine individual and ensemble models for a fair comparison
models = {
    "Logistic Regression": log_reg,
    "Decision Tree": decision_tree,
    "Random Forest": random_forest,
    "Stacking": stacking_model,
    "AdaBoost": adaboost_model,
    "Voting": voting_model,
}

results = []
predictions = {}
confusion_matrices = {}

# 8. Train and evaluate each model
for name, model in models.items():
    print("\n" + "=" * 64)
    print("Training:", name)
    print("=" * 64)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    confusion_matrices[name] = cm

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
    })

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("Confusion Matrix (rows=actual, columns=predicted):")
    print(cm)
    print("Classification Report:")
    print(classification_report(
        y_test, pred, labels=[0, 1], target_names=class_names, zero_division=0
    ))

# 9. Print and save the metric comparison table
results_df = pd.DataFrame(results).sort_values(
    by="Accuracy", ascending=False
).reset_index(drop=True)
print("\n" + "=" * 78)
print("FINAL MODEL PERFORMANCE COMPARISON")
print("=" * 78)
print(results_df.round(4).to_string(index=False))
results_df.to_csv("ensemble_performance_results.csv", index=False)
print("\nSaved: ensemble_performance_results.csv")

# 10. Plot the performance metrics
metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
ax = results_df.set_index("Model")[metric_cols].plot(kind="bar", figsize=(12, 6))
ax.set_title("Performance Comparison of Base and Ensemble Models")
ax.set_xlabel("Machine Learning Models")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
plt.xticks(rotation=30, ha="right")
plt.legend(loc="lower right")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("model_performance_comparison.png", dpi=300, bbox_inches="tight")

# 11. Plot confusion matrices for all six models
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()
for ax, (name, cm) in zip(axes, confusion_matrices.items()):
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(name)
    ax.set_xticks([0, 1], labels=class_names)
    ax.set_yticks([0, 1], labels=class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
fig.suptitle("Confusion Matrices of All Six Models", fontsize=15)
fig.tight_layout()
fig.savefig("confusion_matrices.png", dpi=300, bbox_inches="tight")

# 12. Save confusion matrix counts in a simple CSV file
cm_rows = []
for name, cm in confusion_matrices.items():
    cm_rows.append({
        "Model": name,
        "Actual malignant / predicted malignant": int(cm[0, 0]),
        "Actual malignant / predicted benign": int(cm[0, 1]),
        "Actual benign / predicted malignant": int(cm[1, 0]),
        "Actual benign / predicted benign": int(cm[1, 1]),
    })
pd.DataFrame(cm_rows).to_csv("ensemble_confusion_matrices.csv", index=False)
print("Saved: ensemble_confusion_matrices.csv")

# 13. Identify the top model by test accuracy
best = results_df.iloc[0]
print("\n" + "=" * 64)
print("BEST MODEL BASED ON TEST ACCURACY")
print("=" * 64)
print("Model:", best["Model"])
print(f"Accuracy: {best['Accuracy']:.4f}")
print("\nGraphs saved as model_performance_comparison.png and confusion_matrices.png")
print("Experiment completed successfully.")

# Show graphs when running in a local desktop Python environment
plt.show()
