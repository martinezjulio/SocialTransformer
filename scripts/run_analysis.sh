#!/bin/bash
# Render the Quarto analysis document
# Requires: inference data in $OUTPUT_DATA_DIR/inference/data/

# Load environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

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
echo "Data directory: $OUTPUT_DATA_DIR/inference/data/"
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
