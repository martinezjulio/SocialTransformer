#!/bin/bash
# Script to train the Social Bandit Guider model
# on generated demonstration data

# Activate conda environment
source ~/anaconda3/bin/activate
conda activate banditguider

# Set parameters
DATASET_NAME="demonstrations_data_01"
BASE_DATA_DIR="data"
DATASET_PATH="${BASE_DATA_DIR}/${DATASET_NAME}"

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
DEMO_BELIEF_LOSS="log_mse"  # Options: log_mse, kl_divergence, expected_value_mse, cross_entropy

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
echo "Dataset path: $DATASET_PATH"
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
echo "  Demo belief loss: $DEMO_BELIEF_LOSS"
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
DATASET_METADATA="${DATASET_PATH}/dataset_metadata.json"
TRAIN_MANIFEST="${DATASET_PATH}/train/manifest.json"
VAL_MANIFEST="${DATASET_PATH}/val/manifest.json"

if [ ! -f "$DATASET_METADATA" ]; then
    echo "ERROR: Dataset metadata not found at $DATASET_METADATA"
    echo "Please generate data first using scripts/generate_simulation.sh"
    exit 1
fi

if [ ! -f "$TRAIN_MANIFEST" ]; then
    echo "ERROR: Training manifest not found at $TRAIN_MANIFEST"
    echo "Please generate data first using scripts/generate_simulation.sh"
    exit 1
fi

if [ ! -f "$VAL_MANIFEST" ]; then
    echo "ERROR: Validation manifest not found at $VAL_MANIFEST"
    echo "Please generate data first using scripts/generate_simulation.sh"
    exit 1
fi

# Run training
echo "Starting training..."
python src/train.py \
    --dataset_path "$DATASET_PATH" \
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
    --demo_belief_loss $DEMO_BELIEF_LOSS \
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

