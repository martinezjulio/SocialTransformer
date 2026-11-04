#!/bin/bash
# Full evaluation pipeline:
#   Step 1: Evaluate model on synthetic test set
#   Step 2: Run inference on human experiment data
# Set EVAL_ONLY=1 to stop after step 1.

# Load environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/../.env}"
source "$ENV_FILE"

# Activate conda environment
source "$CONDA_SETUP"
conda activate SocialTransformer

export PYTHONPATH="src:$PYTHONPATH"

# Per-run pilot selection — inherited by run_inference.sh. Change these to switch pilots.
EXPERIMENT_NAME="transfer_bandits_choice_textbox"
VERSION="pilot_E"
export EXPERIMENT_NAME VERSION

# Number of independent Thompson sampling rollouts per session. Inherited by
# run_inference.sh and flows through to every downstream consumer.
NUM_ROLLOUTS="${NUM_ROLLOUTS:-5}"
export NUM_ROLLOUTS

# Set parameters
# The split matching the checkpoint above (demonstrations_data_00 predates the
# Arrow migration and has no loadable test/ dataset).
DATASET_NAME="demonstrations_data_02"
TEST_DATA_DIR="${TRAINING_DATA_ROOT_DIR}/${DATASET_NAME}"
# Note: Output paths managed automatically by eval.py ($OUTPUT_DATA_DIR/eval/)
DEVICE="cuda"  # Options: cuda, cpu
MAX_EXAMPLES=""  # Leave empty for all examples, or set number (e.g., 1000)

# Optional: Override masking strategy from checkpoint (uncomment to use)
#MASKING_STRATEGY_OVERRIDE="all_bandits"  # Options: all_bandits, current_bandit_only


# ==============================================================================
# CHECKPOINT LOADING OPTIONS (choose ONE of the following three options)
# ==============================================================================

# Option 1: Load from local file path — the checkpoint behind every analysis
# result (wandb run 0i8yhqm7, model_type "social"). See the README,
# "Reproducing the model used in the analysis".
CHECKPOINT_PATH="$OUTPUT_DATA_DIR/training_runs/wandb_0i8yhqm7/checkpoints/epoch_119_best.pt"
CHECKPOINT_SOURCE="local"

# Option 2: Load from wandb artifact (uncomment to use)
# Format: "entity/project/artifact:alias"
# NOTE: sbi-transformer-run-oy7ru2t3:v88 was the old default; it is a profile
# model with a beta_nll theta head (2K outputs) and no longer loads here.
#WANDB_ARTIFACT="juliomartinez-individual/social-belief-modeling/sbi-transformer-run-oy7ru2t3:v88"
#CHECKPOINT_SOURCE="wandb_artifact"

# Print configuration
echo "=================================="
echo "Evaluation Configuration"
echo "=================================="
echo "Test data dir: $TEST_DATA_DIR (Arrow split at $TEST_DATA_DIR/test/)"
echo "Output path: \$OUTPUT_DATA_DIR/eval/{data,figures,metadata}/"
echo "Device: $DEVICE"
echo "Rollouts (N): $NUM_ROLLOUTS"
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
fi
echo "=================================="
echo ""

# Check that the Arrow test split exists
if [ ! -d "$TEST_DATA_DIR/test" ]; then
    echo "ERROR: Test Arrow dataset not found at $TEST_DATA_DIR/test"
    echo "(Re-generate the dataset with scripts/generate_simulation.sh)"
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

# Build python command based on checkpoint source (output dirs created by Python)
echo "Starting evaluation..."

# Build max_examples flag
MAX_EXAMPLES_FLAG=""
if [ -n "$MAX_EXAMPLES" ]; then
    MAX_EXAMPLES_FLAG="--max_examples $MAX_EXAMPLES"
fi

# Build masking_strategy_override flag
MASKING_STRATEGY_FLAG=""
if [ -n "$MASKING_STRATEGY_OVERRIDE" ]; then
    MASKING_STRATEGY_FLAG="--masking_strategy_override $MASKING_STRATEGY_OVERRIDE"
fi

if [ "$CHECKPOINT_SOURCE" = "local" ]; then
    python src/eval/eval.py \
        --test_data_dir "$TEST_DATA_DIR" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --device $DEVICE \
        $MAX_EXAMPLES_FLAG \
        $MASKING_STRATEGY_FLAG
elif [ "$CHECKPOINT_SOURCE" = "wandb_artifact" ]; then
    python src/eval/eval.py \
        --test_data_dir "$TEST_DATA_DIR" \
        --wandb_artifact "$WANDB_ARTIFACT" \
        --device $DEVICE \
        $MAX_EXAMPLES_FLAG \
        $MASKING_STRATEGY_FLAG
fi

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "Step 1 completed: Evaluation"
    echo "=================================="
    echo "Results saved to: $OUTPUT_DATA_DIR/eval/"
    echo ""
else
    echo ""
    echo "=================================="
    echo "ERROR: Evaluation failed!"
    echo "=================================="
    exit 1
fi

# Stop after step 1 when EVAL_ONLY is set: synthetic test-set metrics only,
# skipping human inference.
#   EVAL_ONLY=1 bash scripts/run_eval.sh
if [ -n "$EVAL_ONLY" ]; then
    echo "EVAL_ONLY set - stopping after step 1 (no inference)."
    exit 0
fi

# ==============================================================================
# Step 2: Run inference on human experiment data
# ==============================================================================
echo "=================================="
echo "Step 2: Running inference on human experiment data..."
echo "=================================="

bash "${SCRIPT_DIR}/run_inference.sh"

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "Step 2 completed: Inference"
    echo "=================================="
    echo ""
else
    echo ""
    echo "=================================="
    echo "ERROR: Inference failed!"
    echo "=================================="
    exit 1
fi

echo "=================================="
echo "All steps completed successfully!"
echo "=================================="
echo "Eval data:  $OUTPUT_DATA_DIR/eval/data/"
echo "Eval plots: $OUTPUT_DATA_DIR/eval/figures/"
