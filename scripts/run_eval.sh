#!/bin/bash
# Full evaluation pipeline:
#   Step 1: Evaluate model on synthetic test set
#   Step 2: Run inference on human experiment data
#   Step 3: Render eval report (2x2 N_total comparison figure)

# Load environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

# Activate conda environment
source "$CONDA_SETUP"
conda activate SocialTransformer

export PYTHONPATH="src:$PYTHONPATH"

# Set parameters
DATASET_NAME="demonstrations_data_02"
TEST_MANIFEST="${TRAINING_DATA_ROOT_DIR}/${DATASET_NAME}/test/manifest.json"
# Note: Output paths managed automatically by eval.py ($OUTPUT_DATA_DIR/eval/)
DEVICE="cuda"  # Options: cuda, cpu
MAX_EXAMPLES=""  # Leave empty for all examples, or set number (e.g., 1000)

# Optional: Override masking strategy from checkpoint (uncomment to use)
#MASKING_STRATEGY_OVERRIDE="all_bandits"  # Options: all_bandits, current_bandit_only

# Optional: Override profile method from checkpoint (uncomment to use)
#PROFILE_METHOD_OVERRIDE="no_profile"  # Options: no_profile, learned_profile, agent_pooling, cross_attention

# ==============================================================================
# CHECKPOINT LOADING OPTIONS (choose ONE of the following three options)
# ==============================================================================

# Option 1: Load from local file path
#CHECKPOINT_PATH="checkpoints/best_model.pt"
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
echo "Evaluation Configuration"
echo "=================================="
echo "Test manifest: $TEST_MANIFEST"
echo "Output path: \$OUTPUT_DATA_DIR/eval/{data,figures,metadata}/"
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

# Build profile_method_override flag
PROFILE_METHOD_FLAG=""
if [ -n "$PROFILE_METHOD_OVERRIDE" ]; then
    PROFILE_METHOD_FLAG="--profile_method_override $PROFILE_METHOD_OVERRIDE"
fi

if [ "$CHECKPOINT_SOURCE" = "local" ]; then
    python src/eval/eval.py \
        --test_manifest "$TEST_MANIFEST" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --device $DEVICE \
        $MAX_EXAMPLES_FLAG \
        $MASKING_STRATEGY_FLAG \
        $PROFILE_METHOD_FLAG
elif [ "$CHECKPOINT_SOURCE" = "wandb_artifact" ]; then
    python src/eval/eval.py \
        --test_manifest "$TEST_MANIFEST" \
        --wandb_artifact "$WANDB_ARTIFACT" \
        --device $DEVICE \
        $MAX_EXAMPLES_FLAG \
        $MASKING_STRATEGY_FLAG \
        $PROFILE_METHOD_FLAG
elif [ "$CHECKPOINT_SOURCE" = "wandb_sweep" ]; then
    python src/eval/eval.py \
        --test_manifest "$TEST_MANIFEST" \
        --wandb_sweep "$WANDB_SWEEP" \
        --device $DEVICE \
        $MAX_EXAMPLES_FLAG \
        $MASKING_STRATEGY_FLAG \
        $PROFILE_METHOD_FLAG
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

# ==============================================================================
# Step 3: Render eval report (2x2 N_total comparison figure)
# ==============================================================================
echo "=================================="
echo "Rendering eval report..."
echo "=================================="

cd "${SCRIPT_DIR}/.."
quarto render reports/eval_report.qmd --output-dir "$OUTPUT_DATA_DIR/eval"

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "All steps completed successfully!"
    echo "=================================="
    echo "Eval data:  $OUTPUT_DATA_DIR/eval/data/"
    echo "Eval plots: $OUTPUT_DATA_DIR/eval/figures/"
    echo "Report:     $OUTPUT_DATA_DIR/eval/eval_report.html"
    echo "Figure:     $OUTPUT_DATA_DIR/eval/figures/ntot_comparison.png"
else
    echo ""
    echo "=================================="
    echo "ERROR: Eval report rendering failed!"
    echo "=================================="
    exit 1
fi


