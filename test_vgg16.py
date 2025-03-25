import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import argparse
import os

# -------------------------------
# ARGUMENTS PAR LIGNE DE COMMANDE
# -------------------------------
parser = argparse.ArgumentParser(description="Test a fine-tuned VGG-16 model.")
parser.add_argument("--data", type=str, required=True, help="Path to dataset directory")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model file")
args = parser.parse_args()

# -----------------
# CONFIGURATION GPU
# -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------
# PRÉTRAITEMENT IMAGES
# ----------------------
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------
# CHARGEMENT DONNÉES
# -------------------
test_dataset = datasets.ImageFolder(root=os.path.join(args.data, "test"), transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# -------------------
# CHARGEMENT DU MODÈLE
# -------------------
model = models.vgg16(pretrained=False)  # Désactiver le téléchargement du modèle
n_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(n_features, 2)  # 2 classes
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)
model.eval()

# -----------------
# ÉVALUATION DU MODÈLE
# -----------------
print("Evaluating model on test set...")
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")