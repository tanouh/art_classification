import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load extracted features (assumed to be only abstract images)
features = torch.load("output/abstract_features.pt")
features_np = features.numpy()

# -------------------------------
# CLUSTERING USING KMEANS
# -------------------------------
n_clusters = 5  # First try we'll change later
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(features_np)

# -------------------------------
# t-SNE FOR VISUALIZATION
# -------------------------------
print("Running t-SNE...")
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
plt.show()
