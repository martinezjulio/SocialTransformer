#!/bin/bash
# Script to generate training, validation, and test data
# for the Social Bandit Guider project
# 
# This version uses:
# - Permutation-based bandit generation (fixed payoffs permuted)
# - Expert demonstrators with exactly N=72 experience
# - Novice demonstrators with N=0 using random exclusion strategy

# Load environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

# Activate conda environment
source "$CONDA_SETUP"
conda activate SocialTransformer

# Auto-increment dataset name
BASE_DATA_DIR="${TRAINING_DATA_ROOT_DIR}"

# Find the highest existing dataset number (start at -1 so first dataset is 00)
MAX_NUM=-1
for dir in "$BASE_DATA_DIR"/demonstrations_data_*; do
    if [ -d "$dir" ]; then
        # Extract number from directory name (e.g., demonstrations_data_07 -> 07)
        NUM=$(basename "$dir" | sed 's/demonstrations_data_//')
        # Convert to integer and compare
        if [ "$NUM" -eq "$NUM" ] 2>/dev/null; then  # Check if it's a valid number
            if [ "$NUM" -gt "$MAX_NUM" ]; then
                MAX_NUM=$NUM
            fi
        fi
    fi
done

# Increment to get next dataset number
NEXT_NUM=$((MAX_NUM + 1))
DATASET_NAME=$(printf "demonstrations_data_%02d" $NEXT_NUM)
OUT_DIR="${BASE_DATA_DIR}/${DATASET_NAME}"

echo "=================================="
echo "Auto-detected dataset name: $DATASET_NAME"
echo "=================================="
echo ""

# Set parameters
K=12
NUM_DEMONSTRATORS=2
NUM_ROUNDS=3
CHANGE_BANDITS_WITH_ROUNDS=true
ORDER=2
NUM_TRAIN_RUNS=120000
NUM_VALID_RUNS=6000
NUM_TEST_RUNS=6000
T_DEMO=6
T_LEARNER=6
DEMO_STRATEGY="expert_novice_thompson"  # expert_novice_thompson, uniform_thompson, expert_thompson_novice_random
EXPERT_N_MIN=128  # Fixed expert experience
EXPERT_N_MAX=128  # Fixed expert experience
BANDIT_STRATEGY="permute"  # Use permutation-based bandit generation
BANDIT_PAYOFFS="0.0,0.1,0.2,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.75,0.8"  # Fixed payoffs to permute (must have exactly K=12 values)

# Print configuration
echo "=================================="
echo "Data Generation Configuration"
echo "=================================="
echo "Output directory: $OUT_DIR"
echo "Number of arms (K): $K"
echo "Number of demonstrators: $NUM_DEMONSTRATORS"
echo "Number of rounds: $NUM_ROUNDS"
echo "Change bandits with rounds: $CHANGE_BANDITS_WITH_ROUNDS"
echo "Order: $ORDER (0=demo-first, 1=learner-first, 2=both)"
echo "Training runs: $NUM_TRAIN_RUNS"
echo "Validation runs: $NUM_VALID_RUNS"
echo "Test runs: $NUM_TEST_RUNS"
echo "T_demo: $T_DEMO"
echo "T_learner: $T_LEARNER"
echo "Demo strategy: $DEMO_STRATEGY"
echo "Expert N_min: $EXPERT_N_MIN"
echo "Expert N_max: $EXPERT_N_MAX"
echo "Bandit strategy: $BANDIT_STRATEGY"
echo "Bandit payoffs: $BANDIT_PAYOFFS"
echo "Storage: mmap-backed HuggingFace Arrow (one dataset per split)"
echo "=================================="
echo ""

# Build command with flag handling
CHANGE_BANDITS_FLAG=""
if [ "$CHANGE_BANDITS_WITH_ROUNDS" = "true" ]; then
    CHANGE_BANDITS_FLAG="--change_bandits_with_rounds"
fi

# Run generation script
echo "Starting data generation..."
python src/generate_simulations.py \
    --out "$OUT_DIR" \
    --K $K \
    --num_demonstrators $NUM_DEMONSTRATORS \
    --num_rounds $NUM_ROUNDS \
    $CHANGE_BANDITS_FLAG \
    --order $ORDER \
    --num_train_runs $NUM_TRAIN_RUNS \
    --num_valid_runs $NUM_VALID_RUNS \
    --num_test_runs $NUM_TEST_RUNS \
    --T_demo $T_DEMO \
    --T_learner $T_LEARNER \
    --demo_strategy $DEMO_STRATEGY \
    --expert_N_min $EXPERT_N_MIN \
    --expert_N_max $EXPERT_N_MAX \
    --bandit_strategy $BANDIT_STRATEGY \
    --bandit_payoffs "$BANDIT_PAYOFFS"

# Check if generation was successful (the Python generator writes metadata.json
# at the dataset root — no shell-side metadata writing needed anymore).
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "Data generation completed successfully!"
    echo "=================================="
    echo "Generated data in: $OUT_DIR/"
    echo "  - $OUT_DIR/train/  (Arrow dataset)"
    echo "  - $OUT_DIR/val/    (Arrow dataset)"
    echo "  - $OUT_DIR/test/   (Arrow dataset)"
    echo "  - $OUT_DIR/metadata.json"
    ls -lh "$OUT_DIR"/train/*.arrow 2>/dev/null | head -3
    echo ""
else
    echo ""
    echo "=================================="
    echo "ERROR: Data generation failed!"
    echo "=================================="
    exit 1
fi
