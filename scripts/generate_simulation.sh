#!/bin/bash
# Script to generate training, validation, and test data
# for the Social Bandit Guider project

# Activate conda environment
source ~/anaconda3/bin/activate
conda activate banditguider

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
NUM_BANDITS=3
MAX_NUM_DEMOS=1
MAX_CONTEXT_BANDITS=2
NUM_TRAIN_RUNS=6000
NUM_VALID_RUNS=400
NUM_TEST_RUNS=400
T_DEMO=6
T_LEARNER=6
SHARD_SIZE=12000
MAX_DEMOS_PER_BANDIT=1
MAX_EPISODES_PER_BANDIT=1

# Print configuration
echo "=================================="
echo "Data Generation Configuration"
echo "=================================="
echo "Output directory: $OUT_DIR"
echo "Number of arms (K): $K"
echo "Number of bandits: $NUM_BANDITS"
echo "Max demonstrators: $MAX_NUM_DEMOS"
echo "Max demos per bandit: $MAX_DEMOS_PER_BANDIT"
echo "Max episodes per bandit: $MAX_EPISODES_PER_BANDIT"
echo "Max context bandits: $MAX_CONTEXT_BANDITS"
echo "Training runs: $NUM_TRAIN_RUNS"
echo "Validation runs: $NUM_VALID_RUNS"
echo "Test runs: $NUM_TEST_RUNS"
echo "T_demo: $T_DEMO"
echo "T_learner: $T_LEARNER"
echo "N_max: $N_MAX"
echo "Shard size: $SHARD_SIZE"
echo "=================================="
echo ""

# Remove old data if exists (optional - comment out if you want to keep)
# echo "Removing old data directory..."
# rm -rf "$OUT_DIR"

# Run generation script
echo "Starting data generation..."
python src/generate_simulations.py \
    --out "$OUT_DIR" \
    --K $K \
    --N_max $N_MAX \
    --num_bandits $NUM_BANDITS \
    --max_num_demos $MAX_NUM_DEMOS \
    --max_context_bandits $MAX_CONTEXT_BANDITS \
    --num_train_runs $NUM_TRAIN_RUNS \
    --num_valid_runs $NUM_VALID_RUNS \
    --num_test_runs $NUM_TEST_RUNS \
    --T_demo $T_DEMO \
    --T_learner $T_LEARNER \
    --shard_size $SHARD_SIZE \
    --max_demos_per_bandit $MAX_DEMOS_PER_BANDIT \
    --max_episodes_per_bandit $MAX_EPISODES_PER_BANDIT

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
    "num_bandits": $NUM_BANDITS,
    "max_num_demos": $MAX_NUM_DEMOS,
    "max_context_bandits": $MAX_CONTEXT_BANDITS,
    "num_train_runs": $NUM_TRAIN_RUNS,
    "num_valid_runs": $NUM_VALID_RUNS,
    "num_test_runs": $NUM_TEST_RUNS,
    "T_demo": $T_DEMO,
    "T_learner": $T_LEARNER,
    "shard_size": $SHARD_SIZE,
    "max_demos_per_bandit": $MAX_DEMOS_PER_BANDIT,
    "max_episodes_per_bandit": $MAX_EPISODES_PER_BANDIT
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

