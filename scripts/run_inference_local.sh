#!/bin/bash
# Script to run inference on human experiment data
# using a trained Social Bandit Guider model

# Load environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

# Activate conda environment
source "$CONDA_SETUP"
conda activate SocialTransformer

# Set parameters
EXPERIMENT_DATA_PATH="${EXPERIMENT_DATA_PATH}"
OUTPUT_PATH="${TRAINING_DATA_ROOT_DIR}/model_predictions/hybrid_model_predictions.csv"
DEVICE="cpu"  # Options: cuda, cpu
NUM_ROLLOUTS=10  # Number of independent rollouts per session (default: 10 for variance reduction)
SEED=42  # Base random seed for reproducibility

# Optional: Override masking strategy from checkpoint (uncomment to use)
#MASKING_STRATEGY_OVERRIDE="all_bandits"  # Options: all_bandits, current_bandit_only

# Optional: Override profile method from checkpoint (uncomment to use)
#PROFILE_METHOD_OVERRIDE="no_profile"  # Options: no_profile, learned_profile, agent_pooling, cross_attention

# ==============================================================================
# CHECKPOINT LOADING OPTIONS (choose ONE of the following three options)
# ==============================================================================

# Option 1: Load from local file path
#CHECKPOINT_PATH="/Users/juliomartinez/Documents/PhD/SocialBanditGuider/checkpoints/1nb09uxg/epoch_003_best.pt"
#CHECKPOINT_SOURCE="local"

# Option 2: Load from wandb artifact (uncomment to use)
# Format: "entity/project/artifact:alias"
WANDB_ARTIFACT="juliomartinez-individual/social-belief-modeling/sbi-transformer-run-oy7ru2t3:v88"
CHECKPOINT_SOURCE="wandb_artifact"

# Option 3: Load best model from wandb sweep (uncomment to use)
# Format: "entity/project/sweep_id"
#WANDB_SWEEP="juliomartinez-individual/social-belief-modeling/74zpyuvn"
#CHECKPOINT_SOURCE="wandb_sweep"

# Print configuration
echo "=================================="
echo "Inference Configuration"
echo "=================================="
echo "Experiment data: $EXPERIMENT_DATA_PATH"
echo "Output path: $OUTPUT_PATH"
echo "Device: $DEVICE"
echo "Number of rollouts: $NUM_ROLLOUTS"
echo "Random seed: $SEED"
echo ""
echo "Checkpoint source: $CHECKPOINT_SOURCE"
if [ "$CHECKPOINT_SOURCE" = "local" ]; then
    echo "  Local path: $CHECKPOINT_PATH"
elif [ "$CHECKPOINT_SOURCE" = "wandb_artifact" ]; then
    echo "  Wandb artifact: $WANDB_ARTIFACT"
elif [ "$CHECKPOINT_SOURCE" = "wandb_sweep" ]; then
    echo "  Wandb sweep: $WANDB_SWEEP"
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
elif [ "$CHECKPOINT_SOURCE" = "wandb_sweep" ]; then
    if [ -z "$WANDB_SWEEP" ]; then
        echo "ERROR: WANDB_SWEEP not set when CHECKPOINT_SOURCE=wandb_sweep"
        exit 1
    fi
else
    echo "ERROR: Invalid CHECKPOINT_SOURCE: $CHECKPOINT_SOURCE"
    echo "Must be one of: local, wandb_artifact, wandb_sweep"
    exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
mkdir -p "$OUTPUT_DIR"

# Build masking_strategy_override flag
MASKING_STRATEGY_FLAG=""
if [ -n "$MASKING_STRATEGY_OVERRIDE" ]; then
    MASKING_STRATEGY_FLAG="--masking_strategy_override $MASKING_STRATEGY_OVERRIDE"
fi

# Build profile_method_override flag
PROFILE_METHOD_FLAG=""
if [ -n "$PROFILE_METHOD_OVERRIDE" ]; then
    PROFILE_METHOD_FLAG="--profile_method_override $PROFILE_METHOD_OVERRIDE"
fi

# Build python command based on checkpoint source
echo "Starting inference..."
if [ "$CHECKPOINT_SOURCE" = "local" ]; then
    python src/inference.py \
        --experiment_data_path "$EXPERIMENT_DATA_PATH" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --output_path "$OUTPUT_PATH" \
        --device $DEVICE \
        --num_rollouts $NUM_ROLLOUTS \
        --seed $SEED \
        $MASKING_STRATEGY_FLAG \
        $PROFILE_METHOD_FLAG
elif [ "$CHECKPOINT_SOURCE" = "wandb_artifact" ]; then
    python src/inference.py \
        --experiment_data_path "$EXPERIMENT_DATA_PATH" \
        --wandb_artifact "$WANDB_ARTIFACT" \
        --output_path "$OUTPUT_PATH" \
        --device $DEVICE \
        --num_rollouts $NUM_ROLLOUTS \
        --seed $SEED \
        $MASKING_STRATEGY_FLAG \
        $PROFILE_METHOD_FLAG
elif [ "$CHECKPOINT_SOURCE" = "wandb_sweep" ]; then
    python src/inference.py \
        --experiment_data_path "$EXPERIMENT_DATA_PATH" \
        --wandb_sweep "$WANDB_SWEEP" \
        --output_path "$OUTPUT_PATH" \
        --device $DEVICE \
        --num_rollouts $NUM_ROLLOUTS \
        --seed $SEED \
        $MASKING_STRATEGY_FLAG \
        $PROFILE_METHOD_FLAG
fi

# Check if inference was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "Inference completed successfully!"
    echo "=================================="
    echo "Hybrid predictions saved to: $OUTPUT_DATA_DIR/analysis/"
    echo ""
    echo "The output contains:"
    echo "  - Original demonstration segments from human data (duplicated across rollouts)"
    echo "  - Model-simulated learner segments (${NUM_ROLLOUTS} independent rollouts per session)"
    echo ""
else
    echo ""
    echo "=================================="
    echo "ERROR: Inference failed!"
    echo "=================================="
    exit 1
fi

