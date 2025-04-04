#!/bin/bash
#SBATCH --job-name=vgg16_finetune      # Nom du job
#SBATCH --output=out/vgg16_%j.out          # Fichier de sortie (%j = ID du job)
#SBATCH --error=out/vgg16_%j.err           # Fichier d'erreur
#SBATCH --partition=P100               # Partition GPU (à adapter selon le cluster)
#SBATCH --gres=gpu:1                   # Demander 1 GPU
#SBATCH --cpus-per-task=8              # 8 CPU par tâche
#SBATCH --mem=32G                      # 32 Go de RAM
#SBATCH --time=24:00:00                # Temps max (24h)

# Afficher les infos du job
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Définition des variables
DATA_DIR="${HOME}/data/art"
OUTPUT_DIR="${HOME}/projet/art_classification/output"
EPOCHS=10
BATCH_SIZE=32
LR=0.0005
MODEL_PATH="${OUTPUT_DIR}/model/best_vgg16.pth"


# Activer l'environnement Conda
source ${HOME}/.bashrc
conda activate projenv

# Tester le meilleur modèle entraîné
srun python test_vgg16.py --data $DATA_DIR --batch-size $BATCH_SIZE --model-path $MODEL_PATH --output-dir $OUTPUT_DIR

# Afficher l'heure de fin
echo "Job finished at: $(date)"