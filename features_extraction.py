import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import os
import argparse
from datetime import datetime
from tqdm import tqdm
from log_utils import *  # Assurez-vous que ce fichier existe pour la gestion des logs.

# --------------------------
# ARGUMENT PARSING
# --------------------------
parser = argparse.ArgumentParser(description="Extract features from truncated VGG-16 model.")
parser.add_argument("--data", type=str, required=True, help="Path to dataset directory containing 'abstrait' images")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader")
parser.add_argument("--output-dir", type=str, required=True, help="Directory to save extracted features and logs")
args = parser.parse_args()

# --------------------------
# DEVICE SETUP
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up log directory and log file
run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(args.output_dir, "features_logs", run_name)
log_path = os.path.join(log_dir, f"{run_name}.log")
os.makedirs(log_dir, exist_ok=True)

log_message(f"Using device: {device}", log_path)

# --------------------------
# IMAGE TRANSFORMATIONS
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize input to match VGG16 requirements
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                         std=[0.229, 0.224, 0.225])   # ImageNet stds
])

# --------------------------
# LOAD DATASET (ONLY ABSTRACT PAINTINGS)
# --------------------------
# Assumes the 'abstrait' folder contains the abstract images
dataset = datasets.ImageFolder(root=os.path.join(args.data, "abstract_only"), transform=transform)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

log_message(f"Dataset loaded. Found {len(dataset)} abstract images.", log_path)

# --------------------------
# TRUNCATE VGG16
# --------------------------
vgg16 = models.vgg16(pretrained=True).to(device)
vgg16.eval()

# Truncate the classifier: keep everything except the last fully connected layer
# Classifier layers: [fc1, ReLU, Dropout, fc2, ReLU, Dropout, fc3]
# We keep up to index -1 to remove fc3 (final layer)
truncated = nn.Sequential(*list(vgg16.classifier.children())[:-1])
model = nn.Sequential(vgg16.features, nn.Flatten(), truncated).to(device)

log_message("VGG16 model truncated and ready for feature extraction.", log_path)

# --------------------------
# FEATURE EXTRACTION
# --------------------------
all_features = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(loader, desc="Extracting features"):
        inputs = inputs.to(device)
        features = model(inputs)
        all_features.append(features.cpu())  # Collect all features in CPU memory
        all_labels.append(labels)  # Collect corresponding labels

# Concatenate all batches into one tensor
all_features = torch.cat(all_features)
all_labels = torch.cat(all_labels)

# --------------------------
# SAVE FEATURES AND LABELS
# --------------------------
features_path = os.path.join(args.output_dir, "abstract_features.pt")
labels_path = os.path.join(args.output_dir, "abstract_labels.pt")

torch.save(all_features, features_path)
torch.save(all_labels, labels_path)

log_message(f"Features saved to {features_path}", log_path)
log_message(f"Labels saved to {labels_path}", log_path)
log_message("Feature extraction completed successfully!", log_path)
