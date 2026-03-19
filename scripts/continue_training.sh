#!/bin/bash
# Script to continue training from a checkpoint with optional hyperparameter overrides
# Supports loading from local checkpoints or wandb artifacts

# Activate conda environment
source /home/julio/miniforge3/etc/profile.d/conda.sh
conda activate SocialTransformer

# Load environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

# Default configuration
WANDB_PROJECT="social-belief-modeling"
WANDB_ENTITY="juliomartinez-individual"

# Parse arguments
CHECKPOINT=""
SOURCE=""
RESUME_RUN=false
OVERRIDES=()

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --source)
            SOURCE="$2"
            shift 2
            ;;
        --resume_run)
            RESUME_RUN=true
            shift
            ;;
        --epochs|--lr|--scheduler|--optimizer|--weight_decay|--batch_size|\
        --warmup_epochs|--min_lr_ratio|--patience|--factor|--early_stopping_patience|\
        --dropout|--nhead|--num_layers|--dim_feedforward|--d_model|\
        --demo_loss_weight|--demo_belief_loss|--experience_loss_weight|--dataset_path|--device)
            OVERRIDES+=("$1" "$2")
            shift 2
            ;;
        --use_scheduler)
            OVERRIDES+=("$1")
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: --checkpoint is required"
    echo ""
    echo "Usage: ./scripts/continue_training.sh --checkpoint <path_or_run_id> --source <local|wandb> [OPTIONS]"
    echo ""
    echo "Required arguments:"
    echo "  --checkpoint <value>    Checkpoint path (local) or run ID (wandb)"
    echo "  --source <local|wandb>  Source of checkpoint"
    echo ""
    echo "Optional arguments:"
    echo "  --resume_run            Resume same wandb run ID (vs creating new run)"
    echo ""
    echo "Optional overrides:"
    echo "  --epochs <int>          Total epochs to train to"
    echo "  --lr <float>            Learning rate"
    echo "  --scheduler <type>      LR scheduler type (plateau, cosine)"
    echo "  --optimizer <type>      Optimizer (adam, adamw, sgd)"
    echo "  --weight_decay <float>  Weight decay for regularization"
    echo "  --batch_size <int>      Batch size"
    echo ""
    echo "Examples:"
    echo "  # Resume from wandb artifact, add 50 more epochs"
    echo "  ./scripts/continue_training.sh --checkpoint abc123 --source wandb --epochs 150"
    echo ""
    echo "  # Resume same wandb run with new learning rate"
    echo "  ./scripts/continue_training.sh --checkpoint abc123 --source wandb --resume_run --lr 0.00001"
    echo ""
    echo "  # Resume from local checkpoint"
    echo "  ./scripts/continue_training.sh --checkpoint checkpoints/run_id/epoch_050_best.pt --source local --epochs 100"
    exit 1
fi

if [ -z "$SOURCE" ]; then
    echo "ERROR: --source is required (local or wandb)"
    exit 1
fi

if [ "$SOURCE" != "local" ] && [ "$SOURCE" != "wandb" ]; then
    echo "ERROR: --source must be 'local' or 'wandb'"
    exit 1
fi

# Print configuration
echo "=================================="
echo "Continue Training Configuration"
echo "=================================="
echo "Checkpoint: $CHECKPOINT"
echo "Source: $SOURCE"
echo "Resume run: $RESUME_RUN"
if [ ${#OVERRIDES[@]} -gt 0 ]; then
    echo "Overrides: ${OVERRIDES[*]}"
else
    echo "Overrides: None (using checkpoint config)"
fi
echo "=================================="
echo ""

# Build the train.py command
CMD=(
    python src/train.py
    --resume_from_checkpoint "$CHECKPOINT"
    --checkpoint_source "$SOURCE"
    --use_wandb
    --wandb_project "$WANDB_PROJECT"
    --wandb_entity "$WANDB_ENTITY"
)

# Add resume_run flag if set
if [ "$RESUME_RUN" = true ]; then
    CMD+=(--resume_run)
fi

# Add any overrides
CMD+=("${OVERRIDES[@]}")

# Print command for debugging
echo "Executing command:"
echo "${CMD[@]}"
echo ""

# Execute training
"${CMD[@]}"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "Training continuation completed successfully!"
    echo "=================================="
else
    echo ""
    echo "=================================="
    echo "ERROR: Training continuation failed!"
    echo "=================================="
    exit 1
fi
