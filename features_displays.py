import torch
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np

# Charger les features et labels
features = torch.load("output/abstract_features.pt")
labels = torch.load("output/abstract_labels.pt")

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
print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
features_2d = tsne.fit_transform(features_np)