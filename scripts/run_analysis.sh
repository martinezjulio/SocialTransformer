#!/bin/bash
# Render the Quarto analysis document
# Requires: inference data in $MODELING_DATA_DIR/analysis/data/

# Load environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

# Activate conda environment
source "$CONDA_SETUP"
conda activate SocialTransformer

echo "=================================="
echo "Rendering analysis document"
echo "=================================="
echo "Data directory: $MODELING_DATA_DIR/analysis/data/"
echo ""

cd "${SCRIPT_DIR}/../src/analysis"
make render

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "Analysis rendered successfully!"
    echo "=================================="
    echo "Output: src/analysis/analysis.html"
else
    echo ""
    echo "=================================="
    echo "ERROR: Analysis rendering failed!"
    echo "=================================="
    exit 1
fi
