#!/bin/bash
# Submits the analysis pipeline for one version to SLURM as dependent job arrays, one per unit level of
# `run_analysis.py --list-units` (0 table blocks and 1 assembly on the CPU partition, 2 fits, 3 datasets/PPC draws and
# 4 split-half draws on one GPU each), then the final assembly run of every stage (scripts/run_analysis.sh) on the
# long CPU partition, which finds every cache piece filled. Unit lists and logs go to $OUTPUT_DATA_DIR/analysis/<version>/slurm/.
#   bash scripts/slurm/submit.sh --version pilot_E [--dev] [--dry-run]
# .env: SLURM_CPU_PARTITION (default normal), SLURM_GPU_PARTITION (owners, preemptible: the tasks requeue),
# SLURM_FINAL_PARTITION (hns), SLURM_ACCOUNT (optional), N_TABLE_BLOCKS (8; rows per table block unit), SLURM_TABLE_CPUS (16) and
# SLURM_TABLE_MEM (64G) and SLURM_TABLE_TIME (4:00:00) per table block task, SLURM_FINAL_TIME (12:00:00) and SLURM_FINAL_MEM (64G; the pilot's final run peaked at 64 GB) for the final run, SLURM_GPU_OPTS (extra sbatch options
# for the GPU levels; Sherlock needs --gpu_cmode=shared, since its GPUs default to one CUDA context per job and the fits run in a
# spawned worker process beside the parent), SLURM_GPU_MEM (32G) and SLURM_GPU_TIME (2:00:00) per GPU unit (pilot recovery datasets:
# 16-120 min and 22 GB peak on owners' mixed GPUs; 4:00:00 and 48G there), SLURM_ARRAY_CHUNK (96): a level's units are submitted as
# arrays of at most that many tasks, so a partition list such as owners,gpu stays under the gpu QOS's per-job size limit (the
# next level depends on every chunk). Each submission appends its jobs to $DIR/jobs.tsv (stamp, level, job id, tasks) for
# scripts/slurm/status.sh.
# `--dev` allows the reduced sizes (N_DATASETS, N_MODEL_RECOVERY, N_PPC_DRAWS), which the tasks inherit from this shell.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/../../.env}"
set -a; source "$ENV_FILE"; set +a
source "$CONDA_SETUP"
conda activate SocialTransformer
cd "${SCRIPT_DIR}/../.."

VERSION="${VERSION:-}"; DEV=""; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="$2"; shift 2 ;;
    --dev) DEV="--dev"; shift ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$VERSION" ] || { echo "--version is required (e.g. pilot_E)" >&2; exit 2; }
for v in N_DATASETS N_MODEL_RECOVERY N_PPC_DRAWS; do
  if [ -n "${!v:-}" ] && [ -z "$DEV" ]; then echo "$v is set: a reduced run overwrites the full outputs; pass --dev if that is intended" >&2; exit 2; fi
done
export VERSION N_TABLE_BLOCKS="${N_TABLE_BLOCKS:-8}" REPO_DIR="$PWD"
CPU="--partition=${SLURM_CPU_PARTITION:-normal}"; GPU="--partition=${SLURM_GPU_PARTITION:-owners} --gpus=1 ${SLURM_GPU_OPTS:-} --cpus-per-task=4 --mem=${SLURM_GPU_MEM:-32G} --time=${SLURM_GPU_TIME:-2:00:00}"
ACCOUNT="${SLURM_ACCOUNT:+--account=$SLURM_ACCOUNT}"
LEVELS=("$CPU --cpus-per-task=${SLURM_TABLE_CPUS:-16} --mem=${SLURM_TABLE_MEM:-64G} --time=${SLURM_TABLE_TIME:-4:00:00}" "$CPU --cpus-per-task=8 --mem=64G --time=1:00:00" "$GPU" "$GPU" "$GPU")   # assembly: 8 cores for 64 GB under normal's 8 GB/core cap
DIR="$OUTPUT_DATA_DIR/analysis/$VERSION/slurm"; mkdir -p "$DIR"
bash scripts/with_env.sh python src/run_analysis.py --version "$VERSION" --list-units --missing > "$DIR/units.tsv"   # inside the image on Sherlock; cached units are not queued, a level with none is skipped
submit() { local name="$1"; shift; if [ "$DRY" -eq 1 ]; then echo "sbatch $*" >&2; echo "dry-$name"; else sbatch --parsable "$@"; fi; }
STAMP=$(date +%Y-%m-%dT%H:%M:%S); CHUNK="${SLURM_ARRAY_CHUNK:-96}"; DEP=""
record() { [ "$DRY" -eq 1 ] || printf "%s\t%s\t%s\t%s\n" "$STAMP" "$1" "$2" "$3" >> "$DIR/jobs.tsv"; }
for level in 0 1 2 3 4; do
  LIST="$DIR/level$level.txt"; awk -F'\t' -v l="$level" '$1 == l {print $2}' "$DIR/units.tsv" > "$LIST"
  n=$(wc -l < "$LIST"); [ "$n" -gt 0 ] || continue
  rm -f "$DIR/level$level-"[0-9][0-9].txt; split -l "$CHUNK" -d -a 2 "$LIST" "$DIR/level$level-"; JOBS=""
  for PART in "$DIR/level$level-"[0-9][0-9]; do
    mv "$PART" "$PART.txt"; m=$(wc -l < "$PART.txt")
    JOB=$(submit "level$level" $ACCOUNT ${LEVELS[$level]} ${DEP:+--dependency=afterok:$DEP} --array="1-$m" --export=ALL,UNIT_LIST="$PART.txt" --output="$DIR/level$level-%A_%a.out" scripts/slurm/units.sbatch)
    record "$level" "$JOB" "$m"; JOBS="${JOBS:+$JOBS:}$JOB"
  done
  echo "level $level: $n units, job(s) $JOBS"; DEP="$JOBS"
done
JOB=$(submit final $ACCOUNT --partition="${SLURM_FINAL_PARTITION:-hns}" --cpus-per-task=16 --mem="${SLURM_FINAL_MEM:-64G}" --time="${SLURM_FINAL_TIME:-12:00:00}" ${DEP:+--dependency=afterok:$DEP} --job-name=analysis-final --output="$DIR/final-%j.out" --wrap "bash scripts/run_analysis.sh --version $VERSION $DEV")
record final "$JOB" 1
echo "final run: job $JOB (lists and logs under $DIR)"
