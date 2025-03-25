import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from datetime import datetime
from log_utils import *

# -------------------------------
# ARGUMENTS PAR LIGNE DE COMMANDE
# -------------------------------
parser = argparse.ArgumentParser(description="Fine-tuning VGG-16 on a custom dataset.")
parser.add_argument("--data", type=str, required=True, help="Path to dataset directory")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--output-dir", type=str, default="./models", help="Directory to save the trained model")
args = parser.parse_args()

# ---------------------
# TENSORBOARD WRITER
# ---------------------
run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(args.output_dir, "train_logs", run_name)
log_path = f"{log_dir}/{run_name}.log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)


# -----------------
# CONFIGURATION GPU
# -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_message(f"Using device {device}", log_path)

# ----------------------
# PRÉTRAITEMENT IMAGES
# ----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------
# CHARGEMENT DONNÉES
# -------------------
train_dataset = datasets.ImageFolder(root=os.path.join(args.data, "train"), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(args.data, "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# -------------------
# CHARGEMENT DU MODÈLE
# -------------------
model = models.vgg16(pretrained=True)

# Geler les poids du modèle pré-entraîné
for param in model.parameters():
    param.requires_grad = False

# Modifier la dernière couche pour 2 classes
n_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(n_features, 2)
model.to(device)

# ---------------------
# DÉFINITION DE LA PERTE
# ---------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[6].parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


# --------------------
# BOUCLE D'ENTRAÎNEMENT
# --------------------
log_message("Training started.", log_path)

best_val_acc = 0.0  # Pour suivre la meilleure précision sur validation
best_model_path = os.path.join(log_dir, "best_vgg16.pth")

for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    
    # -----------------
    # PHASE DE VALIDATION
    # -----------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    log_message(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f} - Val Accuracy: {val_acc:.2f}%", log_path)

    # -----------------
    # TENSORBOARD LOGGING
    # -----------------
    writer.add_scalar("Loss/train", avg_train_loss, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)


    # Sauvegarder le meilleur modèle
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        log_message(f"New best model saved at {best_model_path}", log_path)

    scheduler.step()

log_message(f"Best validation accuracy: {best_val_acc:.2f}%", log_path)
writer.flush()
writer.close()