#!/bin/bash
#SBATCH --job-name=features_clustering     # Nom du job
#SBATCH --output=out/clusters_%j.out          # Fichier de sortie (%j = ID du job)
#SBATCH --error=out/clusters_%j.err           # Fichier d'erreur
#SBATCH --partition=P100               # Partition GPU (à adapter selon le cluster)
#SBATCH --gres=gpu:1                   # Demander 1 GPU
#SBATCH --cpus-per-task=8              # 8 CPU par tâche
#SBATCH --mem=32G                      # 32 Go de RAM
#SBATCH --time=24:00:00                # Temps max (24h)

# Afficher les infos du job
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Définition des variables
OUTPUT_DIR="${HOME}/projet/art_classification/output"
N_CLUSTERS=10
PLOT_DIR="${OUTPUT_DIR}/plots"
FEATURES_PATH="${OUTPUT_DIR}/features/abstract_features.pt"
LABEL_PATH="${OUTPUT_DIR}/labels/abstract_labels.pt"


# Activer l'environnement Conda
source ${HOME}/.bashrc
conda activate projenv

# Tester le meilleur modèle entraîné
srun python features_displays.py --features-path $FEATURES_PATH --labels-path $LABEL_PATH --n-clusters $N_CLUSTERS --plot-dir $PLOT_DIR

# Afficher l'heure de fin
echo "Job finished at: $(date)"