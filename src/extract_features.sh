#!/bin/bash
#SBATCH --job-name=extract_vgg16_features      # Job name
#SBATCH --output=out/extract_%j.out            # Output file (%j = job ID)
#SBATCH --error=out/extract_%j.err             # Error file
#SBATCH --partition=P100                       # GPU partition (adapt to your cluster)
#SBATCH --gres=gpu:1                           # Request 1 GPU
#SBATCH --cpus-per-task=8                      # 8 CPU cores
#SBATCH --mem=32G                              # 32 GB of RAM
#SBATCH --time=4:00:00                         # Max execution time

# Show job info
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Define variables (adjust paths)

DATA_DIR="${HOME}/data/art"
OUTPUT_DIR="${HOME}/projet/art_classification/output"
BATCH_SIZE=32
MODEL_NAME="best_vgg16.pth"
RUN_NAME="20250605-201040"
MODEL_PATH="${OUTPUT_DIR}/train_logs/${RUN_NAME}/${MODEL_NAME}"

# Activate conda environment
source $HOME/.bashrc
conda activate projenv

# Run feature extraction
srun python extract_features.py --data $DATA_DIR --batch-size $BATCH_SIZE --output-dir $OUTPUT_DIR --model-path $MODEL_PATH --run-name $RUN_NAME

# Show job end time
echo "Job finished at: $(date)"
