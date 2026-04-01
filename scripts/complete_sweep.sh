#!/bin/bash
# Script to resume incomplete runs from a wandb sweep
# This is a wrapper around src/complete_sweep.py that sets up the environment

# Load environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

# Activate conda environment
source "$CONDA_SETUP"
conda activate SocialTransformer

# Check if sweep ID provided
if [ -z "$1" ]; then
    echo "=================================="
    echo "Sweep Recovery Tool"
    echo "=================================="
    echo ""
    echo "Usage: ./scripts/complete_sweep.sh <sweep_id>"
    echo ""
    echo "Description:"
    echo "  This script resumes incomplete runs from a wandb sweep."
    echo "  It will query the sweep for crashed/failed/killed runs,"
    echo "  display them interactively, and let you select which to resume."
    echo ""
    echo "Example:"
    echo "  ./scripts/complete_sweep.sh abc123xyz"
    echo ""
    echo "Configuration:"
    echo "  Entity: $WANDB_ENTITY"
    echo "  Project: $WANDB_PROJECT"
    echo "  (Edit this script to change defaults)"
    echo ""
    exit 1
fi

SWEEP_ID=$1

echo "=================================="
echo "Sweep Recovery Tool"
echo "=================================="
echo "Sweep ID: $SWEEP_ID"
echo "Entity: $WANDB_ENTITY"
echo "Project: $WANDB_PROJECT"
echo "=================================="
echo ""

# Run the Python script
python src/complete_sweep.py "$SWEEP_ID" \
    --entity "$WANDB_ENTITY" \
    --project "$WANDB_PROJECT"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "Sweep recovery completed!"
    echo "=================================="
else
    echo ""
    echo "=================================="
    echo "ERROR: Sweep recovery failed!"
    echo "=================================="
    exit 1
fi
