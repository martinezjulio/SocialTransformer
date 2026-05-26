#!/bin/bash
# Render the Quarto analysis document
# Requires: experiment_trial_predictions.csv in $OUTPUT_DATA_DIR/eval/data/

# Load environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/../.env}"
source "$ENV_FILE"

# Activate conda environment
source "$CONDA_SETUP"
conda activate SocialTransformer

# Per-run selection — change these to render a different pilot.
EXPERIMENT_NAME="transfer_bandits_choice_textbox"
VERSION="pilot_C"

echo "=================================="
echo "Rendering analysis document"
echo "=================================="
echo "Pilot: $EXPERIMENT_NAME / $VERSION"
echo "Data directory: $OUTPUT_DATA_DIR/eval/data/"
echo ""

cd "${SCRIPT_DIR}/.."
quarto render reports/analysis_report.qmd \
    --output-dir "$OUTPUT_DATA_DIR/analysis" \
    -P experiment_name="$EXPERIMENT_NAME" \
    -P version="$VERSION"

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "Analysis rendered successfully!"
    echo "=================================="
    echo "Output: $OUTPUT_DATA_DIR/analysis/analysis_report.html"
else
    echo ""
    echo "=================================="
    echo "ERROR: Analysis rendering failed!"
    echo "=================================="
    exit 1
fi
