import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import TensorBoard
import os
from datetime import datetime
import argparse
from log_utils import *

# -------------------------------
# ARGUMENTS PAR LIGNE DE COMMANDE
# -------------------------------
parser = argparse.ArgumentParser(description="Fine-tuning VGG-16 on a custom dataset.")
parser.add_argument("--data", type=str, required=True, help="Path to dataset directory")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--output-dir", type=str, default="./output", help="Directory to save the trained model")
args = parser.parse_args()


# -----------------
# TENSORBOARD CALLBACK
# -----------------
run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(args.output_dir, "train_logs", run_name)
log_path = os.path.join(log_dir, f"{run_name}.log")
model_path = os.path.join(args.output_dir,f"model/{run_name}" ,"vgg16.h5")

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

log_message("Starting training process...", log_path)
# -------------------------
# PRÉTRAITEMENT DES DONNÉES
# -------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(args.data, 'train'),
    target_size=(224, 224),
    batch_size=args.batch_size,
    class_mode='categorical'
)

val_generator = test_datagen.flow_from_directory(
    os.path.join(args.data, 'val'),
    target_size=(224, 224),
    batch_size=args.batch_size,
    class_mode='categorical'
)
log_message(f"Dataset loaded. Training samples: {train_generator.samples}, Validation samples: {val_generator.samples}", log_path)

# ------------------------
# CHARGEMENT DU MODÈLE
# ------------------------
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Geler les couches du modèle préentraîné
for layer in base_model.layers:
    layer.trainable = False

# Ajouter des couches personnalisées
x = layers.Flatten()(base_model.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(2, activation='softmax')(x)  # Supposons 2 classes

model = models.Model(inputs=base_model.input, outputs=x)

# ------------------------
# COMPILATION DU MODÈLE
# ------------------------
model.compile(optimizer=optimizers.Adam(learning_rate=args.lr), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

log_message("Model compiled successfully.", log_path)

# ----------------------
# BOUCLE D'ENTRAÎNEMENT
# ----------------------
model.fit(
    train_generator,
    epochs=args.epochs,
    validation_data=val_generator,
    callbacks=[tensorboard_callback]
)

# Sauvegarder le meilleur modèle
model.save(os.path.join(model_path))
log_message(f"Model saved at {model_path}", log_path)
