import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import argparse
import os
from datetime import datetime
from log_utils import *

# ---------------------
# ARGUMENTS PAR LIGNE DE COMMANDE
# ---------------------
parser = argparse.ArgumentParser(description="Test a fine-tuned VGG-16 model.")
parser.add_argument("--data", type=str, required=True, help="Path to dataset directory")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model file")
args = parser.parse_args()

# ---------------------
# LOGGING
# ---------------------
run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(args.output_dir, "test_logs", run_name)
log_path = f"{log_dir}/{run_name}.log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# -------------------
# PRÉTRAITEMENT DES DONNÉES
# -------------------
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    os.path.join(args.data, 'test'),
    target_size=(224, 224),
    batch_size=args.batch_size,
    class_mode='categorical',
    shuffle=False
)

# -------------------
# CHARGEMENT DU MODÈLE
# -------------------
model = load_model(args.model_path)

# -----------------
# ÉVALUATION DU MODÈLE
# -----------------
log_message("Evaluating model on test set...", log_path)

# Evaluation du modèle sur l'ensemble de test
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

# Affichage des résultats
log_message(f"Test Accuracy: {test_accuracy * 100:.2f}%", log_path)
log_message(f"Test Loss: {test_loss:.4f}", log_path)
