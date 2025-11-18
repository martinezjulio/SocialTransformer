#!/bin/bash
# Script to train the Social Bandit Guider model
# on generated demonstration data

# Activate conda environment
source ~/anaconda3/bin/activate
conda activate banditguider

# Set parameters
DATASET_NAME="demonstrations_data_01"
BASE_DATA_DIR="data"
TRAIN_PATH="${BASE_DATA_DIR}/${DATASET_NAME}/train/manifest.json"
VAL_PATH="${BASE_DATA_DIR}/${DATASET_NAME}/val/manifest.json"

# Model hyperparameters
BATCH_SIZE=32
LEARNING_RATE=3e-4
EPOCHS=10
DEVICE="cpu"

# Transformer hyperparameters
OPTIMIZER="adamw"
DROPOUT=0.1
NHEAD=8
NUM_LAYERS=4
DIM_FEEDFORWARD=768
D_MODEL=256
DEMO_LOSS_WEIGHT=0.5

# Weights & Biases configuration
WANDB_PROJECT="social-belief-modeling"
WANDB_ENTITY="juliomartinez-individual"

# Checkpoint configuration
CHECKPOINT_DIR="checkpoints"
SAVE_EVERY=2  # Save only best and final (set to N to save every N epochs)
ARTIFACT_NAME="sbi-transformer-checkpoint"

# Print configuration
echo "=================================="
echo "Training Configuration"
echo "=================================="
echo "Dataset: $DATASET_NAME"
echo "Train path: $TRAIN_PATH"
echo "Val path: $VAL_PATH"
echo ""
echo "Model Hyperparameters:"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Epochs: $EPOCHS"
echo "  Device: $DEVICE"
echo "  (num_arms, max_bandits auto-detected from manifest)"
echo ""
echo "Transformer Architecture:"
echo "  Optimizer: $OPTIMIZER"
echo "  Dropout: $DROPOUT"
echo "  Attention heads: $NHEAD"
echo "  Num layers: $NUM_LAYERS"
echo "  FFN dimension: $DIM_FEEDFORWARD"
echo "  Model dimension: $D_MODEL"
echo "  Demo loss weight: $DEMO_LOSS_WEIGHT"
echo ""
echo "Weights & Biases:"
echo "  Project: $WANDB_PROJECT"
echo "  Entity: $WANDB_ENTITY"
echo ""
echo "Checkpoints:"
echo "  Directory: $CHECKPOINT_DIR"
echo "  Save every: $SAVE_EVERY epochs"
echo "  Artifact name: $ARTIFACT_NAME"
echo "=================================="
echo ""

# Check if dataset exists
if [ ! -f "$TRAIN_PATH" ]; then
    echo "ERROR: Training manifest not found at $TRAIN_PATH"
    echo "Please generate data first using scripts/generate_simulation.sh"
    exit 1
fi

if [ ! -f "$VAL_PATH" ]; then
    echo "ERROR: Validation manifest not found at $VAL_PATH"
    echo "Please generate data first using scripts/generate_simulation.sh"
    exit 1
fi

# Run training
echo "Starting training..."
python src/train.py \
    --train_path "$TRAIN_PATH" \
    --val_path "$VAL_PATH" \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --epochs $EPOCHS \
    --device $DEVICE \
    --optimizer $OPTIMIZER \
    --dropout $DROPOUT \
    --nhead $NHEAD \
    --num_layers $NUM_LAYERS \
    --dim_feedforward $DIM_FEEDFORWARD \
    --d_model $D_MODEL \
    --demo_loss_weight $DEMO_LOSS_WEIGHT \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --save_every $SAVE_EVERY \
    --artifact_name "$ARTIFACT_NAME"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "Training completed successfully!"
    echo "=================================="
    echo "Model state dict saved to: sbi_transformer.pt"
    echo "Checkpoints saved to: $CHECKPOINT_DIR/{run-id}/"
    echo "  - best_model.pt (best validation loss)"
    echo "  - final_model.pt (final epoch)"
    if [ "$SAVE_EVERY" -gt 0 ]; then
        echo "  - checkpoint_epoch_N.pt (periodic checkpoints)"
    fi
    echo ""
    echo "Note: When using --use_wandb, checkpoints are organized by run ID"
    echo "All checkpoints logged to wandb artifacts!"
    echo "View at: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT/artifacts"
    echo ""
else
    echo ""
    echo "=================================="
    echo "ERROR: Training failed!"
    echo "=================================="
    exit 1
fi

