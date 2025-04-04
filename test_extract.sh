# Show job info
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Define variables (adjust paths)
DATA_DIR="$HOME/data/art/train/"
OUTPUT_DIR="$HOME/projet/art_classification/output/"
BATCH_SIZE=32

# Activate conda environment
source $HOME/.bashrc
conda activate projenv

# Run feature extraction
python features_extraction.py --data "$DATA_DIR" --batch-size $BATCH_SIZE --output-dir "$OUTPUT_DIR"

# Show job end time
echo "Job finished at: $(date)"
