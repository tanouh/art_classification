import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
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
parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the trained model")
parser.add_argument("--model-name", type=str, required=True, help="Model name")
args = parser.parse_args()

# ---------------------
# CONFIGURATION & LOGGING
# ---------------------
run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(args.output_dir, "train_logs", run_name)
log_path = os.path.join(log_dir, f"{run_name}.log")
os.makedirs(log_dir, exist_ok=True)

model_dir = os.path.join(args.output_dir, "model")
best_model_path = os.path.join(model_dir, args.model_name)
os.makedirs(model_dir, exist_ok=True)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_message(f"Using device: {device}", log_path)
log_message("Starting training process...\n"
            f"n_epoch : {args.epochs}, batch size : {args.batch_size}, learning rate : {args.lr}", log_path)

# ----------------------
# TRANSFORMATIONS & DATA
# ----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(args.data, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(args.data, "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

log_message(f"Dataset loaded. Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}", log_path)

# -------------------
# MODÈLE VGG-16
# -------------------
model = models.vgg16(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = False

# Modification du classifieur pour 2 classes
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 2)
model.to(device)

# --------------------
# ENTRAÎNEMENT
# --------------------
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.classifier[6].parameters(), lr=args.lr)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
log_message(f"Optimizer: SGD", log_path)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

best_val_acc = 0.0

for epoch in range(args.epochs):
    model.train()
    train_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # -----------------
    # VALIDATION
    # -----------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    val_acc = 100.0 * correct / total

    log_message(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f} - Val Accuracy: {val_acc:.2f}%", log_path)


    # Sauvegarde du meilleur modèle
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        log_message(f"New best model saved at {best_model_path}", log_path)

    scheduler.step()

log_message(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%", log_path)

