#!/bin/bash
# One array task = one cluster unit of the analysis pipeline (src/run_analysis.py --unit). Task i runs line i of the
# unit list file UNIT_LIST; scripts/slurm/submit.sh writes the lists and submits one array per level with the
# resources of that level (partition, GPU, CPUs, time), which override the defaults below. Every unit writes its
# cache piece atomically and is skipped when the piece exists, so a preempted task is simply requeued (--requeue)
# and resumes from the cache. Run one task by hand with SLURM_ARRAY_TASK_ID=<i> UNIT_LIST=<file> VERSION=<v>.
#SBATCH --job-name=analysis-units
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/../../.env}"
set -a; source "$ENV_FILE"; set +a
source "$CONDA_SETUP"
conda activate SocialTransformer
cd "${SCRIPT_DIR}/../.."
: "${VERSION:?VERSION is not set}" "${UNIT_LIST:?UNIT_LIST is not set}" "${SLURM_ARRAY_TASK_ID:?not an array task}"
UNIT="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$UNIT_LIST")"
[ -n "$UNIT" ] || { echo "no unit on line $SLURM_ARRAY_TASK_ID of $UNIT_LIST" >&2; exit 2; }
echo "== $UNIT on ${SLURMD_NODENAME:-$(hostname)} ($(date +%H:%M:%S))"
python src/run_analysis.py --version "$VERSION" --unit "$UNIT"
echo "== done ($(date +%H:%M:%S))"
