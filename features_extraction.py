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
from log_utils import *  

# --------------------------
# ARGUMENT PARSING
# --------------------------
parser = argparse.ArgumentParser(description="Extract features from truncated VGG-16 model.")
parser.add_argument("--data", type=str, required=True, help="Path to dataset directory containing 'abstrait' images")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader")
parser.add_argument("--output-dir", type=str, required=True, help="Directory to save extracted features and logs")
parser.add_argument("--model-path", type=str, required=True, help="Path to the pre-trained VGG16 model")
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
model = models.vgg16(pretrained=False)  # Load VGG16 model
model.classifier[6] = nn.Linear(in_features=4096, out_features=2)  # Modify the last layer to have 2 outputs (for binary classification)

weights = torch.load(args.model_path, weights_only=True)  # Load the pre-trained weights
model.load_state_dict(weights)  # Load the pre-trained weights
model = model.to(device)
model.eval()

# Truncate the classifier: keep everything except the last fully connected layer
# Classifier layers: [fc1, ReLU, Dropout, fc2, ReLU, Dropout, fc3]
# We keep up to index -1 to remove fc3 (final layer)
truncated = nn.Sequential(
    model.features,
    nn.AdaptiveAvgPool2d(output_size=(7, 7)),  # Ensure the output size is consistent
    nn.Flatten(),
    nn.Sequential(*list(model.classifier.children())[:-1])  # Keep all but the last layer
).to(device)
# --------------------------
# FEATURE EXTRACTION
# --------------------------
all_features = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(loader, desc="Extracting features"):
        inputs = inputs.to(device)
        features = truncated(inputs)
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
