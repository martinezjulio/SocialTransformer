#!/bin/bash
# One array task = one cluster unit of the analysis pipeline (src/run_analysis.py --unit). Task i runs line i of the
# unit list file UNIT_LIST; scripts/slurm/submit.sh writes the lists and submits one array per level with the
# resources of that level (partition, GPU, CPUs, time), which override the defaults below. Every unit writes its
# cache piece atomically and is skipped when the piece exists, so a preempted task is simply requeued (--requeue)
# and resumes from the cache. Run one task by hand with SLURM_ARRAY_TASK_ID=<i> UNIT_LIST=<file> VERSION=<v>.
# sbatch runs a copy of this file from its spool directory, so the repo is REPO_DIR (exported by submit.sh), falling back
# to this file's location for runs by hand; with CONTAINER set in .env the task re-executes inside the image.
#SBATCH --job-name=analysis-units
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
set -euo pipefail
# The body is one function so bash parses the whole file before executing it: the container re-exec runs this file from the repo, and bash otherwise reads a script incrementally, so an edit during a long task would be executed as garbage (seen 4 Nov, exit 127).
main() {
  REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
  ENV_FILE="${ENV_FILE:-${REPO_DIR}/.env}"
  set -a; source "$ENV_FILE"; set +a
  if [ -n "${CONTAINER:-}" ] && [ -z "${APPTAINER_CONTAINER:-}" ]; then   # Sherlock: re-run inside the image; a container that fails to start (exit 255, e.g. a transient squashfuse mount error) is retried
    for attempt in 1 2 3; do
      bash "$REPO_DIR/scripts/with_env.sh" bash "$REPO_DIR/scripts/slurm/units.sbatch" && exit 0
      rc=$?; [ "$rc" -eq 255 ] || exit "$rc"
      echo "== container start failed (attempt $attempt, exit 255); retrying in 60 s"; sleep 60
    done
    exit 255
  fi
  source "$CONDA_SETUP"
  conda activate SocialTransformer
  cd "$REPO_DIR"
  export XDG_CACHE_HOME="/tmp/$USER/job-$SLURM_JOB_ID"; mkdir -p "$XDG_CACHE_HOME"   # per-task cache dir: Sherlock sets it to the node-shared /tmp, and arviz writes a stamp file there at import, which tasks starting together on one node raced on (exit 1)
  : "${VERSION:?VERSION is not set}" "${UNIT_LIST:?UNIT_LIST is not set}" "${SLURM_ARRAY_TASK_ID:?not an array task}"
  UNIT="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$UNIT_LIST")"
  [ -n "$UNIT" ] || { echo "no unit on line $SLURM_ARRAY_TASK_ID of $UNIT_LIST" >&2; exit 2; }
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1   # array tasks: the table builder's own pool uses the allocated cores; BLAS pools would only add idle threads beyond the allocation
  echo "== $UNIT on ${SLURMD_NODENAME:-$(hostname)} ($(date +%H:%M:%S)), threads: $(nproc) cores allocated"
  python src/run_analysis.py --version "$VERSION" --unit "$UNIT"
  echo "== done ($(date +%H:%M:%S))"
}
main "$@"; exit $?   # parsed in full before anything runs; nothing is read past this line
