#!/bin/bash
# Runs a command in the SocialTransformer environment: .env sourced, the conda env active, and, when .env sets CONTAINER
# (Sherlock), inside that Apptainer image, whose glibc the pinned wheels need (the nodes run CentOS 7, glibc 2.17; jax,
# torch, numpy and scipy wheels want >= 2.27). With no command, opens a shell in the environment.
#   bash scripts/with_env.sh python src/run_analysis.py --version pilot_E --list-units
#   bash scripts/with_env.sh pytest tests/test_numbers.py -q
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; source "${ENV_FILE:-.env}"; set +a
if [ -n "${CONTAINER:-}" ] && [ -z "${APPTAINER_CONTAINER:-}" ]; then exec apptainer exec --nv "$CONTAINER" bash scripts/with_env.sh "$@"; fi
[ -z "${APPTAINER_CONTAINER:-}" ] || export LC_ALL=C.UTF-8 LANG=C.UTF-8   # the host's locale does not exist in the image
if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then   # inside a Slurm task: every thread pool stays within the allocated cores (BLAS, OpenMP, torch, XLA on CPU)
  export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
  export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$SLURM_CPUS_PER_TASK ${XLA_FLAGS:-}"
fi
source "$CONDA_SETUP"
conda activate SocialTransformer
if [ $# -gt 0 ]; then exec "$@"; else exec bash; fi
