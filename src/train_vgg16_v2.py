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
from functions import *
from torchinfo import summary

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
parser.add_argument("--run-name", type=str, required=True, help="Run name for logging")
args = parser.parse_args()

# ---------------------
# CONFIGURATION & LOGGING
# ---------------------
run_name = args.run_name if args.run_name else datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(args.output_dir, "train_logs", run_name)
log_path = os.path.join(log_dir, f"{run_name}.log")
os.makedirs(log_dir, exist_ok=True)

model_dir = os.path.join(args.output_dir, "model")
best_model_path = os.path.join(log_dir, args.model_name)
os.makedirs(model_dir, exist_ok=True)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_message(f"Using device: {device}", log_path)
log_message("Starting training process...\n"
            f"n_epoch : {args.epochs}, batch size : {args.batch_size}, learning rate : {args.lr}", log_path)

# ----------------------
# TRANSFORMATIONS & DATA
# ----------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # LetterboxPad(size=(224, 224), fill_mode='reflect'),  # Resize with letterbox padding
    transforms.RandomRotation(degrees=90),               # Rotate images randomly up to ±90°
    transforms.RandomHorizontalFlip(p=0.5),               # Flip horizontally with 50% chance
    transforms.RandomVerticalFlip(p=0.5),                 # Flip vertically with 50% chance
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(args.data, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(args.data, "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

log_message(f"Dataset loaded. Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}", log_path)

# -------------------
# MODÈLE VGG-16
# -------------------
model = models.vgg16(pretrained=True)

threshold = 32
# Fine tune convolutional layers for feature extraction
log_message(f"Freezing convolutional layers for feature extraction (N {threshold} and beyond)", log_path)
# Freezing all layers except the last two convolutional layers
# This allows the model to retain learned features while adapting the last layers for the new task.
for idx, (name, param) in enumerate(model.features.named_parameters()):
    if idx >= threshold:  # Block 5
        param.requires_grad = True
    else:
        param.requires_grad = False

# Freezing all layers 
# for param in model.features.parameters():
#     param.requires_grad = False


for param in model.classifier.parameters():
    param.requires_grad = True


num_features = model.classifier[6].in_features
model.classifier[6] = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 2)
)
model.to(device)

# --------------------
# MODEL SUMMARY
# --------------------
# Créer le résumé sous forme de chaîne
model_summary_str = str(summary(model, input_size=(args.batch_size, 3, 224, 224), 
                                col_names=["input_size", "output_size", "num_params"], 
                                depth=3))

# Log le résumé
log_message("Model Summary:\n" + model_summary_str, log_path)

# --------------------
# ENTRAÎNEMENT
# --------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[6].parameters(), lr=args.lr)
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
# log_message(f"Optimizer: SGD", log_path)
log_message(f"Optimizer: Adam", log_path)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

best_val_acc = 0.0

for epoch in range(args.epochs):
    model.train()
    train_loss = 0.0
    val_loss = 0.0
    train_acc = 0.0
    val_acc = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += (outputs.argmax(dim=1) == labels).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = (train_acc * 100) / len(train_loader.dataset)
    

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
            val_loss += criterion(outputs, labels).item()
            correct += (preds == labels).sum().item()

    val_acc = 100.0 * correct / total
    val_loss /= len(val_loader)

    log_message(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.2f} - Train accuracy: {avg_train_acc:.2f}% - Val loss: {val_loss:.2f} - Val Accuracy: {val_acc:.2f}%", log_path)


    # Sauvegarde du meilleur modèle
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        log_message(f"New best model saved at {best_model_path}", log_path)

    scheduler.step()

log_message(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%", log_path)

