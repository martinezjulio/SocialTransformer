#!/bin/bash
# Slurm wrapper around scripts/run_training.sh for single-node multi-GPU DDP training.
#
# Hyperparameters and the torchrun invocation live in run_training.sh. This file owns
# ONLY Slurm scheduling + selection of the cluster .env. NUM_GPUS is derived from the
# Slurm allocation so `--nproc_per_node` matches the requested GPU count automatically.
#
# Submit:
#   sbatch scripts/run_training.sbatch
# Ad-hoc override:
#   sbatch --gpus-per-node=2 --time=4:00:00 scripts/run_training.sbatch

#SBATCH --job-name=social-transformer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/PLACEHOLDER/slurm-logs/slurm-%j.out
#SBATCH --error=/PLACEHOLDER/slurm-logs/slurm-%j.err
# #SBATCH --account=PLACEHOLDER
# #SBATCH --partition=PLACEHOLDER

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Cluster env (paths, conda, wandb). Real file is gitignored; template at .env.cluster.example.
export ENV_FILE="${SCRIPT_DIR}/../.env.cluster"

# Derive per-node rank count from Slurm. Falls back to 1 if neither var is set.
export NUM_GPUS="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}"

exec bash "${SCRIPT_DIR}/run_training.sh"
