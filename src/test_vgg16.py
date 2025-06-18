import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import argparse
import os
from datetime import datetime
from functions import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# -------------------------------
# ARGUMENTS PAR LIGNE DE COMMANDE
# -------------------------------
parser = argparse.ArgumentParser(description="Test a fine-tuned VGG-16 model.")
parser.add_argument("--data", type=str, required=True, help="Path to dataset directory")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model file")
parser.add_argument("--output-dir", type=str, required=True, help="Directory to save logs")
parser.add_argument("--run-name", type=str, required=True, help="Optional run name for logging")
args = parser.parse_args()

# ---------------------
# LOGGING SETUP
# ---------------------
run_name = args.run_name if args.run_name else datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join(args.output_dir, "test_logs", run_name)
log_path = os.path.join(log_dir, f"{run_name}.log")
os.makedirs(log_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_message(f"Using device: {device}", log_path)

# ---------------------
# PRÉTRAITEMENT DES DONNÉES
# ---------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root=os.path.join(args.data, "test"), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

log_message(f"Test dataset loaded. Total samples: {len(test_dataset)}", log_path)

# ---------------------
# CHARGEMENT DU MODÈLE
# ---------------------
model = models.vgg16(pretrained=False)
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 2)
)
model.load_state_dict(torch.load(args.model_path, weights_only=True, map_location=device))
model.to(device)
model.eval()

# ---------------------
# ÉVALUATION DU MODÈLE
# ---------------------
log_message("Evaluating model on test set...", log_path)

criterion = nn.CrossEntropyLoss()
test_loss = 0.0
correct = 0
total = 0

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

avg_test_loss = test_loss / len(test_loader)
accuracy = 100.0 * correct / total

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.savefig(os.path.join(log_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')  # You can change the filename and format



log_message(f"Test Accuracy: {accuracy:.2f}%", log_path)
log_message(f"Test Loss: {avg_test_loss:.4f}", log_path)
