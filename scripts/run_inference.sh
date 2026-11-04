#!/bin/bash
# Script to run inference on human experiment data
# using a trained Social Bandit Guider model

# Load environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/../.env}"
source "$ENV_FILE"

# Activate conda environment
source "$CONDA_SETUP"
conda activate SocialTransformer

# Set parameters
# Pilot selection — override from caller (e.g. run_eval.sh) via env; else defaults apply.
VERSION="${VERSION:-pilot_D}"
EXPERIMENT_DATA_PATH="${EXPERIMENT_DATA_DIR}/${VERSION}/preprocessed_participant_data.csv"
EVAL_DIR="${OUTPUT_DATA_DIR}/eval/${VERSION}"   # per-pilot outputs: eval/<version>/{data,metadata}
DEVICE="cuda"  # Options: cuda, cpu
NUM_ROLLOUTS="${NUM_ROLLOUTS:-5}"  # Inherited from run_eval.sh env if set; else default to 5
SEED=42  # Base random seed for reproducibility

# Optional: Override masking strategy from checkpoint (uncomment to use)
#MASKING_STRATEGY_OVERRIDE="all_bandits"  # Options: all_bandits, current_bandit_only


# ==============================================================================
# CHECKPOINT LOADING OPTIONS (choose ONE of the following three options)
# ==============================================================================

# Option 1: Load from local file path (pilot E checkpoint, run wandb_0i8yhqm7, epoch 119)
CHECKPOINT_PATH="$OUTPUT_DATA_DIR/training_runs/wandb_0i8yhqm7/checkpoints/epoch_119_best.pt"
CHECKPOINT_SOURCE="local"

# Option 2: Load from wandb artifact (uncomment to use)
# Format: "entity/project/artifact:alias"
#WANDB_ARTIFACT="juliomartinez-individual/social-belief-modeling/sbi-transformer-run-oy7ru2t3:v88"
#CHECKPOINT_SOURCE="wandb_artifact"

# Print configuration
echo "=================================="
echo "Inference Configuration"
echo "=================================="
echo "Pilot: $VERSION"
echo "Experiment data: $EXPERIMENT_DATA_PATH"
echo "Device: $DEVICE"
echo "Number of rollouts: $NUM_ROLLOUTS"
echo "Random seed: $SEED"
echo ""
echo "Checkpoint source: $CHECKPOINT_SOURCE"
if [ "$CHECKPOINT_SOURCE" = "local" ]; then
    echo "  Local path: $CHECKPOINT_PATH"
elif [ "$CHECKPOINT_SOURCE" = "wandb_artifact" ]; then
    echo "  Wandb artifact: $WANDB_ARTIFACT"
fi
echo ""
echo "Note: Model configuration (num_arms, num_agents, T_learner, etc.)"
echo "      will be loaded automatically from the checkpoint."
echo "=================================="
echo ""

# Check if experiment data exists
if [ ! -f "$EXPERIMENT_DATA_PATH" ]; then
    echo "ERROR: Experiment data not found at $EXPERIMENT_DATA_PATH"
    exit 1
fi

# Validate checkpoint source configuration
if [ "$CHECKPOINT_SOURCE" = "local" ]; then
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
        exit 1
    fi
elif [ "$CHECKPOINT_SOURCE" = "wandb_artifact" ]; then
    if [ -z "$WANDB_ARTIFACT" ]; then
        echo "ERROR: WANDB_ARTIFACT not set when CHECKPOINT_SOURCE=wandb_artifact"
        exit 1
    fi
else
    echo "ERROR: Invalid CHECKPOINT_SOURCE: $CHECKPOINT_SOURCE"
    echo "Must be one of: local, wandb_artifact"
    exit 1
fi

# Build masking_strategy_override flag
MASKING_STRATEGY_FLAG=""
if [ -n "$MASKING_STRATEGY_OVERRIDE" ]; then
    MASKING_STRATEGY_FLAG="--masking_strategy_override $MASKING_STRATEGY_OVERRIDE"
fi

# Build python command based on checkpoint source
echo "Starting inference..."
if [ "$CHECKPOINT_SOURCE" = "local" ]; then
    python src/run_inference.py \
        --experiment_data_path "$EXPERIMENT_DATA_PATH" \
        --eval_dir "$EVAL_DIR" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --device $DEVICE \
        --num_rollouts $NUM_ROLLOUTS \
        --seed $SEED \
        $MASKING_STRATEGY_FLAG
elif [ "$CHECKPOINT_SOURCE" = "wandb_artifact" ]; then
    python src/run_inference.py \
        --experiment_data_path "$EXPERIMENT_DATA_PATH" \
        --eval_dir "$EVAL_DIR" \
        --wandb_artifact "$WANDB_ARTIFACT" \
        --device $DEVICE \
        --num_rollouts $NUM_ROLLOUTS \
        --seed $SEED \
        $MASKING_STRATEGY_FLAG
fi

# Check if inference was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "Inference completed successfully!"
    echo "=================================="
    echo "Experiment predictions saved to: $OUTPUT_DATA_DIR/eval/data/"
    echo ""
    echo "The output contains:"
    echo "  - Original demonstration segments from human data (duplicated across rollouts)"
    echo "  - Model-simulated learner segments ($NUM_ROLLOUTS independent rollouts per session)"
    echo ""
else
    echo ""
    echo "=================================="
    echo "ERROR: Inference failed!"
    echo "=================================="
    exit 1
fi

