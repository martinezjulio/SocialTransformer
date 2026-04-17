#!/bin/bash
# Render the eval report Quarto document (2x2 N_total comparison figure)
# Requires: eval data in $OUTPUT_DATA_DIR/eval/data/ (test_* and experiment_* CSVs)

# Load environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

# Activate conda environment
source "$CONDA_SETUP"
conda activate SocialTransformer

echo "=================================="
echo "Re-running inference (recent_betas + true_payoffs columns)"
echo "=================================="
bash "${SCRIPT_DIR}/run_inference.sh" || { echo "Inference failed"; exit 1; }

echo "=================================="
echo "Rendering eval report"
echo "=================================="
echo "Eval data: $OUTPUT_DATA_DIR/eval/data/"
echo ""

cd "${SCRIPT_DIR}/.."
quarto render reports/eval_report.qmd --output-dir "$OUTPUT_DATA_DIR/eval"

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "Eval report rendered successfully!"
    echo "=================================="
    echo "Output: $OUTPUT_DATA_DIR/eval/eval_report.html"
    echo "Figure: $OUTPUT_DATA_DIR/eval/figures/ntot_comparison.png"
    echo "Figure: $OUTPUT_DATA_DIR/eval/figures/ntot_model_beta_methods_comparison.png"
    echo "Figure: $OUTPUT_DATA_DIR/eval/figures/mean_reward_comparison.png"
    echo "Figure: $OUTPUT_DATA_DIR/eval/figures/beta_prior_comparison_expert_first_sigma_avg.png"
    echo "Figure: $OUTPUT_DATA_DIR/eval/figures/beta_prior_comparison_expert_first_sigma_pooled.png"
    echo "Figure: $OUTPUT_DATA_DIR/eval/figures/beta_prior_comparison_novice_first_sigma_avg.png"
    echo "Figure: $OUTPUT_DATA_DIR/eval/figures/beta_prior_comparison_novice_first_sigma_pooled.png"
    echo "Figures: $OUTPUT_DATA_DIR/eval/figures/per_session_beta_priors/ (one per participant)"
else
    echo ""
    echo "=================================="
    echo "ERROR: Eval report rendering failed!"
    echo "=================================="
    exit 1
fi
