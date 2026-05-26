#!/bin/bash
# Slurm wrapper for `wandb agent` — one job = one long-running sweep worker.
#
# A wandb sweep is two phases. Register the sweep once (no GPU needed):
#   wandb sweep src/Sweeps/Sweep_Version_EnvThetas.yaml
# That prints a sweep_id like "juliomartinez-individual/social-belief-modeling/abc123".
# Then submit one or more agents on the cluster:
#   sbatch scripts/run_sweep_agent.sbatch juliomartinez-individual/social-belief-modeling/abc123
# Submit it N times for N parallel workers — wandb dispatches trials independently.
#
# Optional: forward extra args to `wandb agent`. For one-trial-per-job array runs:
#   sbatch --array=1-50 scripts/run_sweep_agent.sbatch <sweep_id> --count=1

#SBATCH --job-name=social-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/PLACEHOLDER/slurm-logs/sweep-%j.out
#SBATCH --error=/PLACEHOLDER/slurm-logs/sweep-%j.err
# #SBATCH --account=PLACEHOLDER
# #SBATCH --partition=PLACEHOLDER

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source cluster env directly. The sweep's `command:` block invokes torchrun → train.py,
# bypassing run_training.sh, so the ENV_FILE indirection used by run_training.sbatch
# doesn't apply here.
source "${SCRIPT_DIR}/../.env.cluster"
source "$CONDA_SETUP"
conda activate SocialTransformer

SWEEP_ID="${1:?usage: sbatch run_sweep_agent.sbatch <entity/project/sweep_id> [agent args...]}"
shift  # remaining "$@" forwards to wandb agent (e.g., --count=1 for array-job mode)

# NOTE: --gpus-per-node above MUST stay matched to the YAML's `--nproc_per_node`
# (currently 2 in src/Sweeps/Sweep_Version_EnvThetas.yaml). If you change one,
# change the other.

exec wandb agent "$@" "$SWEEP_ID"
