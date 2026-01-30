#!/bin/bash
# Script to generate training, validation, and test data
# for the Social Bandit Guider project
# 
# This version uses:
# - Permutation-based bandit generation (fixed payoffs permuted)
# - Expert demonstrators with exactly N=72 experience
# - Novice demonstrators with N=0 using random exclusion strategy

# Activate conda environment
#source ~/anaconda3/bin/activate
source /home/julio/miniforge3/etc/profile.d/conda.sh
conda activate SocialTransformer

# Auto-increment dataset name
BASE_DATA_DIR="data"

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
N_MAX=72
NUM_DEMONSTRATORS=2
NUM_ROUNDS=3
CHANGE_BANDITS_WITH_ROUNDS=true
ORDER=2
NUM_TRAIN_RUNS=120000  
NUM_VALID_RUNS=6000  
NUM_TEST_RUNS=6000   
T_DEMO=6
T_LEARNER=6
SHARD_SIZE=2000
MAX_CONTEXT=144
DEMO_STRATEGY="expert_thompson_novice_random"  # Expert uses Thompson, novice uses random exclusion
EXPERT_N_MIN=72  # Fixed expert experience
EXPERT_N_MAX=72  # Fixed expert experience
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
echo "N_max: $N_MAX"
echo "Demo strategy: $DEMO_STRATEGY"
echo "Expert N_min: $EXPERT_N_MIN (FIXED at 72)"
echo "Expert N_max: $EXPERT_N_MAX (FIXED at 72)"
echo "Bandit strategy: $BANDIT_STRATEGY"
echo "Bandit payoffs: $BANDIT_PAYOFFS"
echo "Shard size: $SHARD_SIZE"
echo "Max context length: $MAX_CONTEXT"
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
    --N_max $N_MAX \
    --num_demonstrators $NUM_DEMONSTRATORS \
    --num_rounds $NUM_ROUNDS \
    $CHANGE_BANDITS_FLAG \
    --order $ORDER \
    --num_train_runs $NUM_TRAIN_RUNS \
    --num_valid_runs $NUM_VALID_RUNS \
    --num_test_runs $NUM_TEST_RUNS \
    --T_demo $T_DEMO \
    --T_learner $T_LEARNER \
    --shard_size $SHARD_SIZE \
    --demo_strategy $DEMO_STRATEGY \
    --expert_N_min $EXPERT_N_MIN \
    --expert_N_max $EXPERT_N_MAX \
    --bandit_strategy $BANDIT_STRATEGY \
    --bandit_payoffs "$BANDIT_PAYOFFS"

# Check if generation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "Data generation completed successfully!"
    echo "=================================="
    echo "Generated data in: $OUT_DIR/"
    echo "  - $OUT_DIR/train/"
    echo "  - $OUT_DIR/val/"
    echo "  - $OUT_DIR/test/"
    ls -lh "$OUT_DIR"/train/*.pt 2>/dev/null | head -3
    echo ""
    
    # Create metadata file with generation parameters
    METADATA_FILE="${OUT_DIR}/dataset_metadata.json"
    echo "Creating metadata file: $METADATA_FILE"
    cat > "$METADATA_FILE" << EOF
{
  "dataset_name": "$DATASET_NAME",
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "parameters": {
    "K": $K,
    "N_max": $N_MAX,
    "num_demonstrators": $NUM_DEMONSTRATORS,
    "num_rounds": $NUM_ROUNDS,
    "change_bandits_with_rounds": $CHANGE_BANDITS_WITH_ROUNDS,
    "order": $ORDER,
    "num_train_runs": $NUM_TRAIN_RUNS,
    "num_valid_runs": $NUM_VALID_RUNS,
    "num_test_runs": $NUM_TEST_RUNS,
    "T_demo": $T_DEMO,
    "T_learner": $T_LEARNER,
    "shard_size": $SHARD_SIZE,
    "max_context": $MAX_CONTEXT,
    "demo_strategy": "$DEMO_STRATEGY",
    "expert_N_min": $EXPERT_N_MIN,
    "expert_N_max": $EXPERT_N_MAX,
    "bandit_strategy": "$BANDIT_STRATEGY",
    "bandit_payoffs": "$BANDIT_PAYOFFS"
  },
  "splits": {
    "train": "$OUT_DIR/train/",
    "val": "$OUT_DIR/val/",
    "test": "$OUT_DIR/test/"
  }
}
EOF
    echo "✓ Metadata saved to $METADATA_FILE"
    echo ""
else
    echo ""
    echo "=================================="
    echo "ERROR: Data generation failed!"
    echo "=================================="
    exit 1
fi
