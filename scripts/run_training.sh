#!/bin/bash
# Script to train the Social Bandit Guider model
# on generated demonstration data

# Activate conda environment
#source ~/anaconda3/bin/activate
source /home/julio/miniforge3/etc/profile.d/conda.sh
conda activate SocialTransformer

# Set parameters
DATASET_NAME="demonstrations_data_02"
BASE_DATA_DIR="data"
DATASET_PATH="${BASE_DATA_DIR}/${DATASET_NAME}"

# Model hyperparameters
BATCH_SIZE=256
LEARNING_RATE=1e-4
EPOCHS=200
DEVICE="cuda"
WEIGHT_DECAY=0.01  # Weight decay for regularization

# Transformer hyperparameters
OPTIMIZER="adamw"
DROPOUT=0.1
NHEAD=16
NUM_LAYERS=8
DIM_FEEDFORWARD=1024
D_MODEL=512
DEMO_LOSS_WEIGHT=1.0
DEMO_BELIEF_LOSS="cross_entropy"  # Options: log_mse, kl_divergence, expected_value_mse, cross_entropy
EXPERIENCE_LOSS_WEIGHT=1.0  # Weight for N_tot experience loss (0.0 = disabled, 1.0 = equal weight)
MASKING_STRATEGY="all_bandits"  # Options: all_bandits, current_bandit_only

# Profile and Experience Head Configuration
PROFILE_METHOD="no_profile"  # Options: no_profile, learned_profile, agent_pooling, cross_attention
EXPERIENCE_HEAD_LOSS_WEIGHT=1.0  # Weight for direct experience head loss (only used for profile methods)
EXPERIENCE_HEAD_LOSS_TYPE="mse"  # Options: mse, bce
EXPERT_THRESHOLD=30.0  # Threshold for expert classification (only used for BCE loss)

# Weights & Biases configuration
WANDB_PROJECT="social-belief-modeling"
WANDB_ENTITY="juliomartinez-individual"

# Checkpoint configuration
CHECKPOINT_DIR="checkpoints"
SAVE_EVERY=2  # Save only best and final (set to N to save every N epochs)
ARTIFACT_NAME="sbi-transformer-checkpoint"

# Learning rate scheduler
USE_SCHEDULER=true  # Enable LR scheduler
WARMUP_EPOCHS=2     # Warmup epochs for cosine scheduler

# Scheduler type: "plateau" or "cosine"
SCHEDULER="plateau"
#SCHEDULER="cosine"  # Uncomment to use cosine scheduler instead

# Minimum LR (as ratio of initial LR, e.g., 1e-5 means min_lr = lr * 1e-5)
MIN_LR_RATIO=0.00001

# ReduceLROnPlateau parameters (only used when SCHEDULER="plateau")
PATIENCE=5          # Number of epochs with no improvement before reducing LR
FACTOR=0.5          # Factor by which to reduce LR (new_lr = lr * factor)

# Early stopping (set to 0 to disable)
EARLY_STOPPING_PATIENCE=10  # Stop training if no improvement for N epochs
# EARLY_STOPPING_PATIENCE=0   # Uncomment to disable early stopping

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
echo "  Weight decay: $WEIGHT_DECAY"
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
echo "  Experience loss weight: $EXPERIENCE_LOSS_WEIGHT"
echo "  Masking strategy: $MASKING_STRATEGY"
echo ""
echo "Profile Method & Experience Head:"
echo "  Profile method: $PROFILE_METHOD"
echo "  Experience head loss weight: $EXPERIENCE_HEAD_LOSS_WEIGHT"
echo "  Experience head loss type: $EXPERIENCE_HEAD_LOSS_TYPE"
echo "  Expert threshold (for BCE): $EXPERT_THRESHOLD"
echo ""
echo "Weights & Biases:"
echo "  Project: $WANDB_PROJECT"
echo "  Entity: $WANDB_ENTITY"
echo ""
echo "Checkpoints:"
echo "  Directory: $CHECKPOINT_DIR"
echo "  Save every: $SAVE_EVERY epochs"
echo "  Artifact name: $ARTIFACT_NAME"
echo ""
echo "Learning Rate Scheduler:"
echo "  Use scheduler: $USE_SCHEDULER"
echo "  Scheduler type: $SCHEDULER"
echo "  Warmup epochs: $WARMUP_EPOCHS (for cosine)"
echo "  Min LR ratio: $MIN_LR_RATIO"
if [ "$SCHEDULER" = "plateau" ]; then
    echo "  Patience: $PATIENCE epochs"
    echo "  Reduction factor: $FACTOR"
fi
echo ""
echo "Early Stopping:"
if [ "$EARLY_STOPPING_PATIENCE" -gt 0 ]; then
    echo "  Enabled: stop after $EARLY_STOPPING_PATIENCE epochs without improvement"
else
    echo "  Disabled"
fi
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

# Build scheduler flag
SCHEDULER_FLAG=""
if [ "$USE_SCHEDULER" = "true" ]; then
    SCHEDULER_FLAG="--use_scheduler"
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
    --weight_decay $WEIGHT_DECAY \
    --dropout $DROPOUT \
    --nhead $NHEAD \
    --num_layers $NUM_LAYERS \
    --dim_feedforward $DIM_FEEDFORWARD \
    --d_model $D_MODEL \
    --demo_loss_weight $DEMO_LOSS_WEIGHT \
    --demo_belief_loss $DEMO_BELIEF_LOSS \
    --experience_loss_weight $EXPERIENCE_LOSS_WEIGHT \
    --masking_strategy $MASKING_STRATEGY \
    --profile_method $PROFILE_METHOD \
    --experience_head_loss_weight $EXPERIENCE_HEAD_LOSS_WEIGHT \
    --experience_head_loss_type $EXPERIENCE_HEAD_LOSS_TYPE \
    --expert_threshold $EXPERT_THRESHOLD \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --save_every $SAVE_EVERY \
    --artifact_name "$ARTIFACT_NAME" \
    $SCHEDULER_FLAG \
    --scheduler $SCHEDULER \
    --warmup_epochs $WARMUP_EPOCHS \
    --min_lr_ratio $MIN_LR_RATIO \
    --patience $PATIENCE \
    --factor $FACTOR \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE

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

