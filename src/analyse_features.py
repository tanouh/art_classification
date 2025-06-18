import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from collections import Counter

# --------------------
# PARAMÈTRES DE BASE
# --------------------
parser = argparse.ArgumentParser(description="Feature Clustering with Text Labels")
parser.add_argument("--features-path", type=str, required=True, help="Path to features .pt file")
parser.add_argument("--labels-path", type=str, required=True, help="Path to labels .pt file")
parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the output")
args = parser.parse_args()

features_path = args.features_path
labels_path = args.labels_path
output_dir = args.output_dir

os.makedirs(output_dir, exist_ok=True)

# --------------------
# CHARGEMENT DES DONNÉES
# --------------------
features = torch.load(features_path).numpy()
labels = torch.load(labels_path).numpy()
num_samples, num_dims = features.shape
num_classes = len(np.unique(labels))

# --------------------
# RAPPORT TEXTE DE BASE
# --------------------
report = []
report.append(f"# Synthesis Report\n")
report.append(f"- Nb of samples: {num_samples}")
report.append(f"- Dim of feautures : {num_dims}")
report.append(f"- Nb classes : {num_classes}")
report.append(f"- Class repartition : {dict(Counter(labels))}\n")

# --------------------
# 1. t-SNE PLOT
# --------------------
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
features_2d = tsne.fit_transform(features)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=labels, palette='tab10', alpha=0.7)
plt.title("t-SNE Projection")
plt.savefig(f"{output_dir}/tsne.png", dpi=300)
plt.close()
report.append("![t-SNE](tsne.png)")

# --------------------
# 2. HEATMAP
# --------------------
sampled = features[np.random.choice(features.shape[0], size=min(300, num_samples), replace=False)]
plt.figure(figsize=(14, 6))
sns.heatmap(sampled, cmap="viridis")
plt.title("Feature Heatmap (sampled)")
plt.savefig(f"{output_dir}/heatmap.png", dpi=300)
plt.close()
report.append("![Heatmap](heatmap.png)")

# --------------------
# 3. PCA → RGB (Texture abstraite)
# --------------------
pca = PCA(n_components=3)
rgb_proj = pca.fit_transform(features)
rgb_proj = (rgb_proj - rgb_proj.min(0)) / (rgb_proj.max(0) - rgb_proj.min(0) + 1e-8)

plt.figure(figsize=(10, 1))
for i in range(rgb_proj.shape[0]):
    plt.scatter(i, 0, color=rgb_proj[i], s=100)
plt.axis("off")
plt.title("PCA RGB Texture")
plt.savefig(f"{output_dir}/texture_rgb.png", dpi=300)
plt.close()
report.append("![PCA Texture](texture_rgb.png)")

# --------------------
# 4. DISTANCES INTRA / INTER
# --------------------
dists = pairwise_distances(features)
intra_dists = []
inter_dists = []

for i in range(num_samples):
    for j in range(i + 1, num_samples):
        if labels[i] == labels[j]:
            intra_dists.append(dists[i, j])
        else:
            inter_dists.append(dists[i, j])

plt.figure(figsize=(8, 5))
sns.histplot(intra_dists, color='blue', label='Intra-cluster', kde=True, stat='density')
sns.histplot(inter_dists, color='red', label='Inter-cluster', kde=True, stat='density')
plt.legend()
plt.title("Distances Intra vs Inter Cluster")
plt.savefig(f"{output_dir}/distances.png", dpi=300)
plt.close()
report.append("![Distances](distances.png)")

# --------------------
# 5. SAUVEGARDE DU RAPPORT
# --------------------
report_path = os.path.join(output_dir, "report.md")
with open(report_path, "w") as f:
    f.write("\n".join(report))

print(f"Report : {report_path}")
