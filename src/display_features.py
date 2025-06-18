import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import os

# ---------------------
# ARGUMENT PARSING
# ---------------------
parser = argparse.ArgumentParser(description="Feature Clustering with Text Labels")
parser.add_argument("--features-path", type=str, required=True, help="Path to features .pt file")
parser.add_argument("--labels-path", type=str, required=True, help="Path to labels .pt file")
parser.add_argument("--n-clusters", type=int, default=10, help="Number of clusters")
parser.add_argument("--plot-dir", type=str, required=True, help="Directory to save the t-SNE plot")
parser.add_argument("--labels-per-cluster", type=int, default=10, help="Number of labels to show per cluster")
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
# PCA analysis
#----------------------
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features)
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")


# ---------------------
# PLOT WITH TEXT LABELS
# ---------------------
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], c=clusters, cmap='tab10', alpha=0.6)
plt.title(f"t-SNE Visualization with KMeans Clusters (k={clusters.shape[0]})")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.grid(True)

# Add a few text labels per cluster
# for cluster_id in range(clusters.shape[0]):
#     cluster_indices = np.where(clusters == cluster_id)[0]
#     selected_indices = random.sample(list(cluster_indices), min(args.labels_per_cluster, len(cluster_indices)))

#     for idx in selected_indices:
#         x, y = features_2d[idx]
#         label = str(labels[idx])
#         ax.text(x, y, label, fontsize=9, weight='bold', color='black', ha='center', va='center')

# Affiche un point représentatif par cluster (le plus proche du centroïde)
for cluster_id in np.unique(clusters):
    cluster_indices = np.where(clusters == cluster_id)[0]
    
    # Trouve le point le plus proche du centroïde du cluster
    cluster_features = features[cluster_indices]
    centroid = kmeans.cluster_centers_[cluster_id]
    dists = np.linalg.norm(cluster_features - centroid, axis=1)
    representative_idx = cluster_indices[np.argmin(dists)]

    # Récupérer coordonnées t-SNE et label
    x, y = features_2d[representative_idx]
    label = str(labels[representative_idx])
    
    ax.text(x, y, label, fontsize=10, weight='bold', color='black', ha='center', va='center')

# ---------------------
# SAVE PLOT
# ---------------------
os.makedirs(args.plot_dir, exist_ok=True)
output_file = os.path.join(args.plot_dir, f"tsne_k{args.n_clusters}_labels.png")
plt.tight_layout()
plt.savefig(output_file, dpi=300)
print(f"Labeled t-SNE visualization saved to: {output_file}")
