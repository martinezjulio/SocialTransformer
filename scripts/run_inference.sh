#!/bin/bash
# Script to run inference on human experiment data
# using a trained Social Bandit Guider model

# Activate conda environment
source ~/anaconda3/bin/activate
conda activate banditguider

# Set parameters
EXPERIMENT_DATA_PATH="data/experiment_data/transfer_bandits_choice_textbox/pilot_C/preprocessed_data.csv"
#CHECKPOINT_PATH="/Users/juliomartinez/Documents/PhD/SocialBanditGuider/checkpoints/clcmq5qx/epoch_014_best.pt" 
CHECKPOINT_PATH="/Users/juliomartinez/Documents/PhD/SocialBanditGuider/checkpoints/3t18jmf8/epoch_011_best.pt"  
OUTPUT_PATH="data/model_predictions/hybrid_model_predictions.csv"
DEVICE="cpu"  # Options: cuda, cpu

# Model configuration
NUM_ARMS=12
NUM_AGENTS=4
LEARNER_STEPS=6

# Print configuration
echo "=================================="
echo "Inference Configuration"
echo "=================================="
echo "Experiment data: $EXPERIMENT_DATA_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output path: $OUTPUT_PATH"
echo "Device: $DEVICE"
echo ""
echo "Model Configuration:"
echo "  Number of arms: $NUM_ARMS"
echo "  Max agents: $NUM_AGENTS"
echo "  Learner steps: $LEARNER_STEPS"
echo "=================================="
echo ""

# Check if experiment data exists
if [ ! -f "$EXPERIMENT_DATA_PATH" ]; then
    echo "ERROR: Experiment data not found at $EXPERIMENT_DATA_PATH"
    exit 1
fi

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Please update CHECKPOINT_PATH in this script to point to your trained model checkpoint"
    exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
mkdir -p "$OUTPUT_DIR"

# Run inference
echo "Starting inference..."
python src/inference.py \
    --experiment_data_path "$EXPERIMENT_DATA_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --device $DEVICE \
    --num_arms $NUM_ARMS \
    --num_agents $NUM_AGENTS \
    --learner_steps $LEARNER_STEPS

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

