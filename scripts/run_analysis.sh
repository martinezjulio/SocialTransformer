#!/bin/bash
# Driver of the analysis pipeline (src/analysis): the Python stages through src/run_analysis.py and the R stages
# in src/r/, in pipeline order, on one version of the experiment data. Every heavy stage is cached (tables and fits
# by content hash, the checks by unit), so a rerun recomputes only what changed.
#   bash scripts/run_analysis.sh --version pilot_E [--stages games,descriptive,...] [--dev]
# Default stages: games descriptive partner_choice inference-check choice checks unseen figures report.
# `power` runs only when named: it resamples the pilot (input to the registration, not a result).
# `--dev` allows the reduced-run sizes (N_DATASETS, N_MODEL_RECOVERY, N_PPC_DRAWS, N_SIMS); without it they are refused,
# since the stages overwrite their outputs unconditionally. Preprocessing (src/r/preprocess_participant_data.R) is a
# separate step on the raw data.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/../.env}"
set -a; source "$ENV_FILE"; set +a
source "$CONDA_SETUP"
conda activate SocialTransformer
cd "${SCRIPT_DIR}/.."

VERSION="${VERSION:-}"; STAGES="games,descriptive,partner_choice,inference-check,choice,checks,unseen,figures,report"; DEV=0
while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="$2"; shift 2 ;;
    --stages) STAGES="$2"; shift 2 ;;
    --dev) DEV=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$VERSION" ] || { echo "--version is required (e.g. pilot_E)" >&2; exit 2; }
for v in N_DATASETS N_MODEL_RECOVERY N_PPC_DRAWS N_SIMS; do
  if [ -n "${!v:-}" ] && [ "$DEV" -eq 0 ]; then echo "$v is set: a reduced run overwrites the full outputs; pass --dev if that is intended" >&2; exit 2; fi
done
[ "$DEV" -eq 0 ] || echo "DEV RUN: reduced sizes allowed; outputs under analysis/$VERSION are overwritten"

export VERSION
for stage in ${STAGES//,/ }; do
  echo "== $stage ($(date +%H:%M:%S))"
  case "$stage" in
    descriptive|partner_choice) Rscript "src/r/$stage.R" ;;
    power) echo "power resamples the pilot; it is an input to the registration, not a result"; Rscript src/r/power_analysis.R ;;
    games|inference-check|choice|checks|unseen|figures|report) python src/run_analysis.py --version "$VERSION" --stages "$stage" ;;
    *) echo "unknown stage: $stage" >&2; exit 2 ;;
  esac
done
echo "== done ($(date +%H:%M:%S))"
