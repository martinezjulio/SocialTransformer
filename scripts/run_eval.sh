#!/bin/bash
# Script to evaluate model on test set
# Measures theta and demonstrator prediction accuracy at different context lengths

# Activate conda environment
source /home/julio/miniforge3/etc/profile.d/conda.sh
conda activate SocialTransformer

# Set parameters
TEST_MANIFEST="data/demonstrations_data_02/test/manifest.json"
OUTPUT_PATH="data/evaluation/test_evaluation.csv"
DEVICE="cuda"  # Options: cuda, cpu
MAX_EXAMPLES=""  # Leave empty for all examples, or set number (e.g., 1000)

# ==============================================================================
# CHECKPOINT LOADING OPTIONS (choose ONE of the following three options)
# ==============================================================================

# Option 1: Load from local file path
#CHECKPOINT_PATH="checkpoints/best_model.pt"
#CHECKPOINT_SOURCE="local"

# Option 2: Load from wandb artifact (uncomment to use)
# Format: "entity/project/artifact:alias"
WANDB_ARTIFACT="juliomartinez-individual/social-belief-modeling/sbi-transformer-checkpoint:v1387"
CHECKPOINT_SOURCE="wandb_artifact"

# Option 3: Load best model from wandb sweep (uncomment to use)
# Format: "entity/project/sweep_id"
#WANDB_SWEEP="juliomartinez-individual/social-belief-modeling/74zpyuvn"
#CHECKPOINT_SOURCE="wandb_sweep"

# Print configuration
echo "=================================="
echo "Evaluation Configuration"
echo "=================================="
echo "Test manifest: $TEST_MANIFEST"
echo "Output path: $OUTPUT_PATH"
echo "Device: $DEVICE"
if [ -n "$MAX_EXAMPLES" ]; then
    echo "Max examples: $MAX_EXAMPLES"
else
    echo "Max examples: All"
fi
echo ""
echo "Checkpoint source: $CHECKPOINT_SOURCE"
if [ "$CHECKPOINT_SOURCE" = "local" ]; then
    echo "  Local path: $CHECKPOINT_PATH"
elif [ "$CHECKPOINT_SOURCE" = "wandb_artifact" ]; then
    echo "  Wandb artifact: $WANDB_ARTIFACT"
elif [ "$CHECKPOINT_SOURCE" = "wandb_sweep" ]; then
    echo "  Wandb sweep: $WANDB_SWEEP"
fi
echo "=================================="
echo ""

# Check if test manifest exists
if [ ! -f "$TEST_MANIFEST" ]; then
    echo "ERROR: Test manifest not found at $TEST_MANIFEST"
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
echo "Starting evaluation..."

# Build max_examples flag
MAX_EXAMPLES_FLAG=""
if [ -n "$MAX_EXAMPLES" ]; then
    MAX_EXAMPLES_FLAG="--max_examples $MAX_EXAMPLES"
fi

if [ "$CHECKPOINT_SOURCE" = "local" ]; then
    python src/eval.py \
        --test_manifest "$TEST_MANIFEST" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --output_path "$OUTPUT_PATH" \
        --device $DEVICE \
        $MAX_EXAMPLES_FLAG
elif [ "$CHECKPOINT_SOURCE" = "wandb_artifact" ]; then
    python src/eval.py \
        --test_manifest "$TEST_MANIFEST" \
        --wandb_artifact "$WANDB_ARTIFACT" \
        --output_path "$OUTPUT_PATH" \
        --device $DEVICE \
        $MAX_EXAMPLES_FLAG
elif [ "$CHECKPOINT_SOURCE" = "wandb_sweep" ]; then
    python src/eval.py \
        --test_manifest "$TEST_MANIFEST" \
        --wandb_sweep "$WANDB_SWEEP" \
        --output_path "$OUTPUT_PATH" \
        --device $DEVICE \
        $MAX_EXAMPLES_FLAG
fi

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "Evaluation completed successfully!"
    echo "=================================="
    echo "Results saved to: $OUTPUT_PATH"
    echo ""
    echo "The output contains per-cutoff metrics for:"
    echo "  - Theta (payoff) prediction accuracy"
    echo "  - Expert demonstrator belief accuracy"
    echo "  - Novice demonstrator belief accuracy"
    echo ""
    echo "Analyze with R:"
    echo "  eval_df <- read_csv('$OUTPUT_PATH')"
    echo "  # Plot theta correlation vs cutoff"
    echo "  # Compare expert vs novice prediction accuracy"
    echo ""
else
    echo ""
    echo "=================================="
    echo "ERROR: Evaluation failed!"
    echo "=================================="
    exit 1
fi


