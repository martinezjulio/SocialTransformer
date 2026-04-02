#!/bin/bash
# One-time setup: install R packages via renv for the analysis environment

# Load environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

# Activate conda environment
source "$CONDA_SETUP"
conda activate SocialTransformer

echo "=================================="
echo "Setting up R environment (renv)"
echo "=================================="

cd "${SCRIPT_DIR}/../src/analysis"
Rscript -e 'renv::restore()'

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "R environment setup complete!"
    echo "=================================="
else
    echo ""
    echo "=================================="
    echo "ERROR: renv::restore() failed!"
    echo "=================================="
    exit 1
fi
