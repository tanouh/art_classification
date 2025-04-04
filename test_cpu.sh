#!/bin/bash

# Définition des variables
DATA_DIR="${HOME}/data/art"
OUTPUT_DIR="$HOME/projet/art_classification/output"
EPOCHS=1
BATCH_SIZE=32
LR=0.0005
MODEL_PATH="${OUTPUT_DIR}/model/best_vgg16.pth"

# Afficher les infos du job
echo "Starting job on: $(hostname)"
echo "Job started at: $(date)"

# Activer l'environnement Conda
source $HOME/.bashrc
conda activate projenv

# Étape 1 : Entraîner le modèle
python train_vgg16.py --data $DATA_DIR --epochs $EPOCHS --batch-size $BATCH_SIZE --lr $LR --output-dir $OUTPUT_DIR

# Étape 2 : Tester le meilleur modèle entraîné
python test_vgg16.py --data $DATA_DIR --batch-size $BATCH_SIZE --model-path $MODEL_PATH --output-dir $OUTPUT_DIR

# Afficher l'heure de fin
echo "Job finished at: $(date)"