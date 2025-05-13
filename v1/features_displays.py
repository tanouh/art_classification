import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import os
import random

# ---------------------
# ARGUMENT PARSING
# ---------------------
parser = argparse.ArgumentParser(description="Feature Clustering with Text Labels")
parser.add_argument("--features-path", type=str, required=True, help="Path to features .pt file")
parser.add_argument("--labels-path", type=str, required=True, help="Path to labels .pt file")
parser.add_argument("--n-clusters", type=int, default=5, help="Number of clusters")
parser.add_argument("--plot-dir", type=str, required=True, help="Directory to save the t-SNE plot")
parser.add_argument("--labels-per-cluster", type=int, default=5, help="Number of labels to show per cluster")
args = parser.parse_args()

# ---------------------
# LOAD FEATURES & LABELS
# ---------------------
features = torch.load(args.features_path).numpy()
labels = torch.load(args.labels_path).numpy()

# ---------------------
# KMeans Clustering
# ---------------------
kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
clusters = kmeans.fit_predict(features)

# ---------------------
# t-SNE Projection
# ---------------------
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
features_2d = tsne.fit_transform(features)

# ---------------------
# PLOT WITH TEXT LABELS
# ---------------------
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], c=clusters, cmap='tab10', alpha=0.6)
plt.title(f"t-SNE Visualization with KMeans Clusters (k={args.n_clusters})")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.grid(True)

# Add a few text labels per cluster
for cluster_id in range(args.n_clusters):
    cluster_indices = np.where(clusters == cluster_id)[0]
    selected_indices = random.sample(list(cluster_indices), min(args.labels_per_cluster, len(cluster_indices)))

    for idx in selected_indices:
        x, y = features_2d[idx]
        label = str(labels[idx])
        ax.text(x, y, label, fontsize=9, weight='bold', color='black', ha='center', va='center')

# ---------------------
# SAVE PLOT
# ---------------------
os.makedirs(args.plot_dir, exist_ok=True)
output_file = os.path.join(args.plot_dir, f"tsne_k{args.n_clusters}_labels.png")
plt.tight_layout()
plt.savefig(output_file, dpi=300)
print(f"Labeled t-SNE visualization saved to: {output_file}")
