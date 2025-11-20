#!/bin/bash
# Script to run inference on human experiment data
# using a trained Social Bandit Guider model

# Activate conda environment
source ~/anaconda3/bin/activate
conda activate banditguider

# Set parameters
EXPERIMENT_DATA_PATH="data/experiment_data/transfer_bandits_choice_textbox/pilot_C/preprocessed_data.csv"
OUTPUT_PATH="data/model_predictions/hybrid_model_predictions.csv"
DEVICE="cpu"  # Options: cuda, cpu

# ==============================================================================
# CHECKPOINT LOADING OPTIONS (choose ONE of the following three options)
# ==============================================================================

# Option 1: Load from local file path
#CHECKPOINT_PATH="/Users/juliomartinez/Documents/PhD/SocialBanditGuider/checkpoints/1nb09uxg/epoch_003_best.pt"
#CHECKPOINT_SOURCE="local"

# Option 2: Load from wandb artifact (uncomment to use)
# Format: "entity/project/artifact:alias"
#WANDB_ARTIFACT="juliomartinez-individual/social-belief-modeling/sbi-transformer-checkpoint:v3"
#CHECKPOINT_SOURCE="wandb_artifact"

# Option 3: Load best model from wandb sweep (uncomment to use)
# Format: "entity/project/sweep_id"
WANDB_SWEEP="juliomartinez-individual/social-belief-modeling/74zpyuvn"
CHECKPOINT_SOURCE="wandb_sweep"

# Print configuration
echo "=================================="
echo "Inference Configuration"
echo "=================================="
echo "Experiment data: $EXPERIMENT_DATA_PATH"
echo "Output path: $OUTPUT_PATH"
echo "Device: $DEVICE"
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

# Build python command based on checkpoint source
echo "Starting inference..."
if [ "$CHECKPOINT_SOURCE" = "local" ]; then
    python src/inference.py \
        --experiment_data_path "$EXPERIMENT_DATA_PATH" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --output_path "$OUTPUT_PATH" \
        --device $DEVICE
elif [ "$CHECKPOINT_SOURCE" = "wandb_artifact" ]; then
    python src/inference.py \
        --experiment_data_path "$EXPERIMENT_DATA_PATH" \
        --wandb_artifact "$WANDB_ARTIFACT" \
        --output_path "$OUTPUT_PATH" \
        --device $DEVICE
elif [ "$CHECKPOINT_SOURCE" = "wandb_sweep" ]; then
    python src/inference.py \
        --experiment_data_path "$EXPERIMENT_DATA_PATH" \
        --wandb_sweep "$WANDB_SWEEP" \
        --output_path "$OUTPUT_PATH" \
        --device $DEVICE
fi

# Check if inference was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "Inference completed successfully!"
    echo "=================================="
    echo "Hybrid predictions saved to: $OUTPUT_PATH"
    echo ""
    echo "The output contains:"
    echo "  - Original demonstration segments from human data"
    echo "  - Model-simulated learner segments"
    echo ""
else
    echo ""
    echo "=================================="
    echo "ERROR: Inference failed!"
    echo "=================================="
    exit 1
fi

