"""Practical Assignment 6: Clustering Algorithms
K-Means, Agglomerative Clustering, DBSCAN and Gaussian Mixture Model
Dataset: Wine recognition dataset available in scikit-learn.
"""

import warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)
from scipy.cluster.hierarchy import linkage, dendrogram

OUT_DIR = Path(__file__).resolve().parent / "clustering_outputs"
OUT_DIR.mkdir(exist_ok=True)

# 1. Load dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target  # used only for external evaluation (ARI), not for clustering
class_names = wine.target_names

print("=" * 72)
print("CLUSTERING ALGORITHMS - WINE DATASET")
print("=" * 72)
print("Dataset shape:", X.shape)
print("Samples:", len(X))
print("Features:", X.shape[1])
print("Missing values:", int(X.isnull().sum().sum()))
print("Reference class counts (used only for ARI):")
for i, name in enumerate(class_names):
    print(f"  {name}: {int(np.sum(y == i))}")

# 2. Standardize features; clustering is distance-sensitive
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# 3. PCA is only for plotting, not model training
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(Xs)
print("\nPCA variance explained by first two components:",
      f"{pca.explained_variance_ratio_.sum():.4f}")

# 4. Explore K-Means cluster counts
k_rows = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(Xs)
    k_rows.append({
        "k": k,
        "Inertia": km.inertia_,
        "Silhouette": silhouette_score(Xs, labels),
        "Davies-Bouldin": davies_bouldin_score(Xs, labels),
        "Calinski-Harabasz": calinski_harabasz_score(Xs, labels),
    })
k_df = pd.DataFrame(k_rows)
k_df.to_csv(OUT_DIR / "kmeans_k_selection.csv", index=False)
print("\nK-MEANS: CHOOSING k")
print(k_df.round(4).to_string(index=False))

# 5. Fit four clustering algorithms
models = {
    "K-Means": KMeans(n_clusters=3, n_init=10, random_state=42),
    "Agglomerative (Ward)": AgglomerativeClustering(n_clusters=3, linkage="ward"),
    "DBSCAN": DBSCAN(eps=2.3, min_samples=7),
    "Gaussian Mixture": GaussianMixture(n_components=3, covariance_type="spherical", random_state=42),
}
labels_by_model = {}
results = []

for name, model in models.items():
    if name == "Gaussian Mixture":
        labels = model.fit_predict(Xs)
    else:
        labels = model.fit_predict(Xs)
    labels_by_model[name] = labels

    # Ignore DBSCAN's noise points for internal cluster metrics, while keeping them for ARI.
    if name == "DBSCAN":
        mask = labels != -1
        eval_X = Xs[mask]
        eval_labels = labels[mask]
        noise_count = int(np.sum(~mask))
    else:
        mask = np.ones(len(labels), dtype=bool)
        eval_X = Xs
        eval_labels = labels
        noise_count = 0

    cluster_count = len(set(eval_labels))
    if cluster_count >= 2 and len(eval_labels) > cluster_count:
        sil = silhouette_score(eval_X, eval_labels)
        dbi = davies_bouldin_score(eval_X, eval_labels)
        ch = calinski_harabasz_score(eval_X, eval_labels)
    else:
        sil, dbi, ch = np.nan, np.nan, np.nan

    results.append({
        "Algorithm": name,
        "Clusters (excluding noise)": cluster_count,
        "Noise points": noise_count,
        "Silhouette": sil,
        "Davies-Bouldin": dbi,
        "Calinski-Harabasz": ch,
        "Adjusted Rand Index": adjusted_rand_score(y, labels),
    })

    print("\n" + "-" * 72)
    print("Model:", name)
    counts = pd.Series(labels).value_counts().sort_index()
    print("Cluster sizes (-1 means noise for DBSCAN):")
    print(counts.to_string())
    print("Silhouette:", "N/A" if np.isnan(sil) else f"{sil:.4f}")
    print("Davies-Bouldin:", "N/A" if np.isnan(dbi) else f"{dbi:.4f}")
    print("Calinski-Harabasz:", "N/A" if np.isnan(ch) else f"{ch:.4f}")
    print("Adjusted Rand Index (external check):", f"{adjusted_rand_score(y, labels):.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "clustering_metrics.csv", index=False)
print("\n" + "=" * 72)
print("CLUSTERING MODEL COMPARISON")
print("=" * 72)
print(results_df.round(4).to_string(index=False))
print("\nNote: DBSCAN silhouette/DB/CH metrics exclude noise points; ARI includes all points.")
print("Lower Davies-Bouldin is preferred; higher Silhouette, Calinski-Harabasz and ARI are preferred.")

# 6. K-Means model selection plots
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
plot_specs = [
    ("Inertia", "Elbow: Inertia (lower)"),
    ("Silhouette", "Silhouette (higher)"),
    ("Davies-Bouldin", "Davies-Bouldin (lower)"),
    ("Calinski-Harabasz", "Calinski-Harabasz (higher)"),
]
for ax, (col, title) in zip(axes, plot_specs):
    ax.plot(k_df["k"], k_df[col], marker="o")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("k")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "kmeans_k_selection.png", dpi=180)

# 7. Visualize all four clusterings in PCA space
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, (name, labels) in zip(axes.flat, labels_by_model.items()):
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=32, alpha=0.85)
    ax.set_title(name)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)
fig.suptitle("Clustering Results in PCA Space (PCA used for visualization only)")
plt.tight_layout()
plt.savefig(OUT_DIR / "cluster_visualization_pca.png", dpi=180)

# 8. Plot DBSCAN with noise marked separately
if "DBSCAN" in labels_by_model:
    db_labels = labels_by_model["DBSCAN"]
    fig, ax = plt.subplots(figsize=(7, 5))
    noise = db_labels == -1
    ax.scatter(X_pca[noise, 0], X_pca[noise, 1], c="black", marker="x", s=38, label="Noise")
    for cluster_id in sorted(set(db_labels) - {-1}):
        mask = db_labels == cluster_id
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=32, label=f"Cluster {cluster_id}")
    ax.set_title("DBSCAN Clusters and Noise (PCA view)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dbscan_noise.png", dpi=180)

# 9. Hierarchical clustering dendrogram (subset shown for readability)
Z = linkage(Xs, method="ward")
fig, ax = plt.subplots(figsize=(10, 5))
dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=8, ax=ax)
ax.set_title("Hierarchical Clustering Dendrogram (last 30 merged groups)")
ax.set_xlabel("Cluster / sample group")
ax.set_ylabel("Ward distance")
plt.tight_layout()
plt.savefig(OUT_DIR / "hierarchical_dendrogram.png", dpi=180)

# 10. Cluster profile heatmap for K-Means; means of original standardized features
km_labels = labels_by_model["K-Means"]
profile = pd.DataFrame(Xs, columns=X.columns).assign(Cluster=km_labels).groupby("Cluster").mean()
profile.to_csv(OUT_DIR / "kmeans_cluster_profiles.csv")
fig, ax = plt.subplots(figsize=(12, 4.6))
sns.heatmap(profile, cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Mean standardized feature value"})
ax.set_title("K-Means Cluster Profiles (standardized feature means)")
ax.set_xlabel("Wine features")
ax.set_ylabel("Cluster")
plt.xticks(rotation=75, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "cluster_profiles.png", dpi=180)

# 11. Plot internal metrics and external ARI comparison
metric_plot = results_df.set_index("Algorithm")[["Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Adjusted Rand Index"]]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, metric in zip(axes.flat, metric_plot.columns):
    metric_plot[metric].plot(kind="bar", ax=ax, rot=20)
    ax.set_title(metric)
    ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "clustering_metric_comparison.png", dpi=180)

# Show the generated figures for interactive environments
plt.show()

print("\nFiles saved:")
for filename in [
    "clustering_metrics.csv",
    "kmeans_k_selection.csv",
    "kmeans_cluster_profiles.csv",
    "kmeans_k_selection.png",
    "cluster_visualization_pca.png",
    "dbscan_noise.png",
    "hierarchical_dendrogram.png",
    "cluster_profiles.png",
    "clustering_metric_comparison.png",
]:
    print(" -", OUT_DIR / filename)
print("\nExperiment completed successfully.")
