#!/bin/bash
# Script to train the Social Bandit Guider model
# on generated demonstration data

# Load environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/../.env}"
source "$ENV_FILE"

# Activate conda environment
source "$CONDA_SETUP"
conda activate SocialTransformer

# Set parameters
# Override from the shell, e.g. `DATASET_NAME=demonstrations_data_03 bash scripts/run_training.sh`.
DATASET_NAME="${DATASET_NAME:-demonstrations_data_02}"

# DDP: number of GPUs on this node. Override per-invocation with
# `NUM_GPUS=4 bash scripts/run_training.sh`. NUM_GPUS=1 is the
# legacy single-GPU path (still goes through torchrun, but world_size=1).
NUM_GPUS="${NUM_GPUS:-1}"

# Model hyperparameters
# NOTE: BATCH_SIZE is per-rank under DDP. Effective global batch = BATCH_SIZE * NUM_GPUS.
BATCH_SIZE=128
# DataLoader worker processes. Override: `NUM_WORKERS=8 bash scripts/run_training.sh`.
NUM_WORKERS="${NUM_WORKERS:-4}"
# RNG seed for weight init / shuffling / segment-cutoff draws.
SEED="${SEED:-42}"
LEARNING_RATE=0.00005
EPOCHS=500
DEVICE="cuda"
WEIGHT_DECAY=0.0 # Weight decay for regularization

# Transformer hyperparameters
OPTIMIZER="adamw"
DROPOUT=0.2
NHEAD=4
NUM_LAYERS=8
DIM_FEEDFORWARD=1024
D_MODEL=256
ENV_THETA_LOSS="bce" # Options: bce, beta_nll
DEMO_LOSS_WEIGHT=1.0
DEMO_BELIEF_LOSS="cross_entropy"       # Options: log_mse, kl_divergence, expected_value_mse, cross_entropy
EXPERIENCE_LOSS_WEIGHT=1.0             # Weight for N_tot experience loss (0.0 = disabled, 1.0 = equal weight)
MASKING_STRATEGY="all_bandits" # Options: all_bandits, current_bandit_only

# Model variant + Experience Head Configuration.
# Override from the shell, e.g. `MODEL_TYPE=social bash scripts/run_training.sh`.
# Keep this at `social`: it is the type every published result used, and these
# defaults are that run's hyperparameters (see README, "Reproducing the model
# used in the analysis"). The social_profile_* types are deprecated exploratory
# paths: social, social_profile_learned, social_profile_agent_pooling, social_profile_cross_attention
MODEL_TYPE="${MODEL_TYPE:-social}"
EXPERIENCE_HEAD_LOSS_WEIGHT=0.0    # Weight for direct experience head loss (only used for profile model_types)
EXPERIENCE_HEAD_LOSS_TYPE="mse"    # Options: mse, bce
EXPERT_THRESHOLD=30.0              # Threshold for expert classification (only used for BCE loss)

# Head Input Configurations (Phase 3)
THETA_HEAD_INPUT="query_only"           # Options: query_only, query_and_profiles
DEMO_BELIEFS_HEAD_INPUT="profiles_only" # Options: query_only, profiles_only, query_and_profiles
EXPERIENCE_HEAD_INPUT="profiles_only"   # Options: query_only, profiles_only, query_and_profiles

# Checkpoint configuration
# Note: Checkpoint directory is now managed automatically by train.py
ARTIFACT_NAME="social-transformer"
SAVE_EVERY=1 # Save only best and final (set to N to save every N epochs)

# Learning rate scheduler
USE_SCHEDULER=true # Enable LR scheduler
WARMUP_EPOCHS=2    # Warmup epochs for cosine scheduler

# Scheduler type: "plateau" or "cosine"
SCHEDULER="plateau"
#SCHEDULER="cosine"  # Uncomment to use cosine scheduler instead

# Minimum LR (absolute value, not ratio)
MIN_LR=1e-8

# ReduceLROnPlateau parameters (only used when SCHEDULER="plateau")
PATIENCE=15 # Number of epochs with no improvement before reducing LR
FACTOR=0.7  # Factor by which to reduce LR (new_lr = lr * factor)

# Overfitting-based early stopping (set threshold to 0 to disable)
OVERFITTING_THRESHOLD=15.0 # Percentage above best val loss to detect overfitting (0=disabled)
OVERFITTING_PATIENCE=50    # Consecutive epochs exceeding threshold to trigger stop
OVERFITTING_WINDOW=15      # Window size for moving average trend detection

# Print configuration
echo "=================================="
echo "Training Configuration"
echo "=================================="
echo "Dataset: $DATASET_NAME"
echo "Dataset root: $TRAINING_DATA_ROOT_DIR"
echo ""
echo "DDP:"
echo "  NUM_GPUS: $NUM_GPUS (per-rank batch = $BATCH_SIZE, effective batch = $((BATCH_SIZE * NUM_GPUS)))"
echo ""
echo "Model Hyperparameters:"
echo "  Batch size (per rank): $BATCH_SIZE"
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
echo "  Env theta loss: $ENV_THETA_LOSS"
echo "  Demo loss weight: $DEMO_LOSS_WEIGHT"
echo "  Demo belief loss: $DEMO_BELIEF_LOSS"
echo "  Experience loss weight: $EXPERIENCE_LOSS_WEIGHT"
echo "  Masking strategy: $MASKING_STRATEGY"
echo ""
echo "Profile Method & Experience Head:"
echo "  Model type: $MODEL_TYPE"
echo "  Experience head loss weight: $EXPERIENCE_HEAD_LOSS_WEIGHT"
echo "  Experience head loss type: $EXPERIENCE_HEAD_LOSS_TYPE"
echo "  Expert threshold (for BCE): $EXPERT_THRESHOLD"
echo ""
echo "Head Input Configurations:"
echo "  Theta head input: $THETA_HEAD_INPUT"
echo "  Demo beliefs head input: $DEMO_BELIEFS_HEAD_INPUT"
echo "  Experience head input: $EXPERIENCE_HEAD_INPUT"
echo ""
echo "Weights & Biases:"
echo "  Project: $WANDB_PROJECT"
echo "  Entity: $WANDB_ENTITY"
echo ""
echo "Checkpoints:"
echo "  Directory: Managed automatically (\$OUTPUT_DATA_DIR/training_runs/<run_id>/checkpoints/)"
echo "  Save every: $SAVE_EVERY epochs"
echo "  Artifact name: $ARTIFACT_NAME"
echo ""
echo "Learning Rate Scheduler:"
echo "  Use scheduler: $USE_SCHEDULER"
echo "  Scheduler type: $SCHEDULER"
echo "  Warmup epochs: $WARMUP_EPOCHS (for cosine)"
echo "  Min LR: $MIN_LR"
if [ "$SCHEDULER" = "plateau" ]; then
  echo "  Patience: $PATIENCE epochs"
  echo "  Reduction factor: $FACTOR"
fi
echo ""
echo "Overfitting Detection:"
if [ "$OVERFITTING_THRESHOLD" == "0" ] || [ "$OVERFITTING_THRESHOLD" == "0.0" ]; then
  echo "  Disabled"
else
  echo "  Threshold: ${OVERFITTING_THRESHOLD}% above best val loss"
  echo "  Patience: $OVERFITTING_PATIENCE consecutive epochs"
  echo "  Trend window: $OVERFITTING_WINDOW epochs"
fi
echo "=================================="
echo ""

# Check if dataset exists (Arrow layout: root metadata.json + train/val Arrow dirs)
DATASET_PATH="${TRAINING_DATA_ROOT_DIR}/${DATASET_NAME}"
DATASET_METADATA="${DATASET_PATH}/metadata.json"

if [ ! -f "$DATASET_METADATA" ]; then
  echo "ERROR: Dataset metadata not found at $DATASET_METADATA"
  echo "Please generate data first using scripts/generate_simulation.sh"
  exit 1
fi

if [ ! -d "$DATASET_PATH/train" ]; then
  echo "ERROR: Training Arrow dataset not found at $DATASET_PATH/train"
  echo "Please generate data first using scripts/generate_simulation.sh"
  exit 1
fi

if [ ! -d "$DATASET_PATH/val" ]; then
  echo "ERROR: Validation Arrow dataset not found at $DATASET_PATH/val"
  echo "Please generate data first using scripts/generate_simulation.sh"
  exit 1
fi

# Build scheduler flag
SCHEDULER_FLAG=""
if [ "$USE_SCHEDULER" = "true" ]; then
  SCHEDULER_FLAG="--use_scheduler"
fi

# Run training (DDP via torchrun; --standalone picks a free rendezvous port).
echo "Starting training with torchrun (nproc_per_node=$NUM_GPUS)..."
torchrun --standalone --nproc_per_node=$NUM_GPUS src/train.py \
  --dataset_name "$DATASET_NAME" \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --seed $SEED \
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
  --env_theta_loss $ENV_THETA_LOSS \
  --demo_loss_weight $DEMO_LOSS_WEIGHT \
  --demo_belief_loss $DEMO_BELIEF_LOSS \
  --experience_loss_weight $EXPERIENCE_LOSS_WEIGHT \
  --masking_strategy $MASKING_STRATEGY \
  --model_type $MODEL_TYPE \
  --experience_head_loss_weight $EXPERIENCE_HEAD_LOSS_WEIGHT \
  --experience_head_loss_type $EXPERIENCE_HEAD_LOSS_TYPE \
  --expert_threshold $EXPERT_THRESHOLD \
  --theta_head_input $THETA_HEAD_INPUT \
  --demo_beliefs_head_input $DEMO_BELIEFS_HEAD_INPUT \
  --experience_head_input $EXPERIENCE_HEAD_INPUT \
  --use_wandb \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_entity "$WANDB_ENTITY" \
  --save_every $SAVE_EVERY \
  --artifact_name "$ARTIFACT_NAME" \
  $SCHEDULER_FLAG \
  --scheduler $SCHEDULER \
  --warmup_epochs $WARMUP_EPOCHS \
  --min_lr $MIN_LR \
  --patience $PATIENCE \
  --factor $FACTOR \
  --overfitting_threshold $OVERFITTING_THRESHOLD \
  --overfitting_patience $OVERFITTING_PATIENCE \
  --overfitting_window $OVERFITTING_WINDOW \
  "$@"

# Check if training was successful
if [ $? -eq 0 ]; then
  echo ""
  echo "=================================="
  echo "Training completed successfully!"
  echo "=================================="
  echo "Checkpoints saved to: $OUTPUT_DATA_DIR/training_runs/<run_id>/checkpoints/"
  echo "  - best_model.pt (best validation loss)"
  echo "  - final_model.pt (final epoch)"
  if [ "$SAVE_EVERY" -gt 0 ]; then
    echo "  - checkpoint_epoch_N.pt (periodic checkpoints)"
  fi
  echo ""
  echo "Note: All checkpoints and results organized in experiments directory"
  if [ "$USE_WANDB" = "true" ]; then
    echo "Wandb artifacts logged!"
    echo "View at: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT/artifacts"
  fi
  echo ""
else
  echo ""
  echo "=================================="
  echo "ERROR: Training failed!"
  echo "=================================="
  exit 1
fi
