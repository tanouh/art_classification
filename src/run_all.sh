#!/bin/bash
#SBATCH --job-name=vgg16_finetune      # Nom du job
#SBATCH --output=../out/run/vgg16_%j.out          # Fichier de sortie (%j = ID du job)
#SBATCH --error=../out/run/vgg16_%j.err           # Fichier d'erreur
#SBATCH --partition=P100               # Partition GPU (à adapter selon le cluster)
#SBATCH --gres=gpu:1                   # Demander 1 GPU
#SBATCH --cpus-per-task=8              # 8 CPU par tâche
#SBATCH --mem=32G                      # 32 Go de RAM
#SBATCH --time=24:00:00                # Temps max (24h)

# Afficher les infos du job
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Variables
DATA_DIR="${HOME}/data/art"
OUTPUT_DIR="${HOME}/projet/art_classification/output"
EPOCHS=10
BATCH_SIZE=32
LR=0.001

RUN_NAME=$(date +%Y%m%d-%H%M%S)
MODEL_NAME="best_vgg16.pth"
MODEL_PATH="${OUTPUT_DIR}/train_logs/${RUN_NAME}/${MODEL_NAME}"

FEATURES_PATH="${OUTPUT_DIR}/features_logs/${RUN_NAME}/abstract_features.pt"
LABEL_PATH="${OUTPUT_DIR}/features_logs/${RUN_NAME}/abstract_labels.pt"

PLOT_DIR="${OUTPUT_DIR}/features_logs/${RUN_NAME}"
N_CLUSTERS=1

# Activer l'environnement Conda
source ${HOME}/.bashrc
conda activate projenv

# Étape 1 : Entraîner le modèle
srun python train_vgg16_v2.py --data $DATA_DIR --epochs $EPOCHS --batch-size $BATCH_SIZE --lr $LR --output-dir $OUTPUT_DIR --model-name $MODEL_NAME --run-name $RUN_NAME

# Étape 2 : Tester le meilleur modèle entraîné
srun python test_vgg16.py --data $DATA_DIR --batch-size $BATCH_SIZE --model-path $MODEL_PATH --output-dir $OUTPUT_DIR --run-name $RUN_NAME

# Étape 3 : Extraire les caractéristiques
srun python extract_features.py --data $DATA_DIR --batch-size $BATCH_SIZE --output-dir $OUTPUT_DIR --model-path $MODEL_PATH --run-name $RUN_NAME

# Etape 4 : Visualiser les caractéristiques
srun python display_features.py --features-path $FEATURES_PATH --labels-path $LABEL_PATH --n-clusters $N_CLUSTERS --plot-dir $PLOT_DIR

# Afficher l'heure de fin
echo "Job finished at: $(date)"
