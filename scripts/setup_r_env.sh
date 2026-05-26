#!/bin/bash
# One-time setup: install R packages via renv for the analysis environment

# Load environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/../.env}"
source "$ENV_FILE"

# Activate conda environment
source "$CONDA_SETUP"
conda activate SocialTransformer

echo "=================================="
echo "Setting up R environment (renv)"
echo "=================================="

cd "${SCRIPT_DIR}/.."
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
