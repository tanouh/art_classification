import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import os
# --output-dir $OUTPUT_DIR --features-path $FEATURES_PATH --labels-path $LABEL_PATH --n-clusters $N_CLUSTERS --plot-dir $PLOT_DIR

parser = argparse.ArgumentParser(description="Feature Extraction and Clustering")
parser.add_argument("--features-path", type=str, required=True, help="Path to the features file")
parser.add_argument("--labels-path", type=str, required=True, help="Path to the labels file")
parser.add_argument("--n-clusters", type=int, default=5, help="Number of clusters for KMeans")
parser.add_argument("--plot-dir", type=str, required=True, help="Directory to save the plot")
args = parser.parse_args()

# Charger les features et labels
features = torch.load(args.features_path)
labels = torch.load(args.labels_path)

features_np = features.numpy()
labels_np = labels.numpy()

# -------------------------------
# CLUSTERING USING KMEANS
# -------------------------------
n_clusters = 5  # First try we'll change later
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(features_np)

# -------------------------------
# t-SNE FOR VISUALIZATION
# -------------------------------
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
features_2d = tsne.fit_transform(features_np)

# -------------------------------
# PLOT RESULTS
# -------------------------------
plt.figure(figsize=(10, 7))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=clusters, cmap='tab10', alpha=0.7)
plt.title(f"t-SNE Visualization with KMeans Clusters (k={n_clusters})")
plt.colorbar(scatter, label="Cluster ID")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.grid(True)
plt.tight_layout()
output_file = args.plot_dir
os.makedirs(args.plot_dir, exist_ok=True)
output_file = os.path.join(args.plot_dir, f"tsne_clusters_cluster_{args.n_clusters}.png")
plt.savefig(output_file)
print(f"Visualization saved to {output_file}")