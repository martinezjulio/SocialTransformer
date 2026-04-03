#!/bin/bash
# Render the Quarto analysis document
# Requires: inference data in $OUTPUT_DATA_DIR/inference/data/

# Load environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

# Activate conda environment
source "$CONDA_SETUP"
conda activate SocialTransformer

echo "=================================="
echo "Rendering analysis document"
echo "=================================="
echo "Data directory: $OUTPUT_DATA_DIR/inference/data/"
echo ""

cd "${SCRIPT_DIR}/.."
quarto render reports/analysis_report.qmd --output-dir "$OUTPUT_DATA_DIR/other"

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "Analysis rendered successfully!"
    echo "=================================="
    echo "Output: $OUTPUT_DATA_DIR/other/analysis_report.html"
else
    echo ""
    echo "=================================="
    echo "ERROR: Analysis rendering failed!"
    echo "=================================="
    exit 1
fi
