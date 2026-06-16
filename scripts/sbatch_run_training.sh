#!/usr/bin/env bash
#SBATCH --job-name=social-transformer
#SBATCH -A marlowe-m000102
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append
#SBATCH --mail-user=juliomz@stanford.edu
#SBATCH --mail-type=ALL,TIME_LIMIT
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Marlowe preempt training with automatic resume into a SINGLE wandb run.
#
# Restart model: --requeue auto-requeues on preemption (same job id); the USR1
# trap (fired by --signal ~2 min before the 4 h wall) requeues on walltime.
# Either way SLURM keeps the same $SLURM_JOB_ID, so --run_id episode-$SLURM_JOB_ID
# stays stable and train.py auto-resumes from the latest epoch_*_latest.pt checkpoint.
# You'll get an email on end/fail/preemption(REQUEUE)/time-limit; expect a
# REQUEUE+BEGIN pair on every restart of a long multi-slot run.
#
# wandb runs OFFLINE on compute nodes (set WANDB_MODE=offline in .env.cluster;
# compute nodes have no outbound internet). All slots share the run id under
# run_dir/wandb/. After training, sync ONCE from a LOGIN node (has internet):
#   scripts/wandb_sync.sh episode-<jobid>
#
# Submit:        sbatch scripts/sbatch_run_training.sh
# Ad-hoc GPUs:   sbatch --gpus-per-node=4 scripts/sbatch_run_training.sh

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# --- Marlowe paths (override via env if your layout differs) ---
IMAGE="${IMAGE:-/scratch/m000102/juliomz/apptainerfiles/sifs/marlowe-social.sif}"
REPO="${REPO:-/projects/m000102/juliomz/SocialBanditsProject/SocialTransformer}"

# Tell the in-container launcher to source .env.cluster, and forward GPU count.
# run_training.sh reads NUM_GPUS for torchrun --nproc_per_node.
export ENV_FILE=.env.cluster
export NUM_GPUS="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}"

# Requeue on imminent walltime (USR1 from --signal). Runs on the HOST because
# scontrol is not available inside the container.
trap 'echo "[sbatch] USR1 received: requeuing $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"' USR1

# Run the existing conda launcher INSIDE the container (it sources .env.cluster,
# activates the SocialTransformer env, and runs torchrun). --run_id is forwarded
# into train.py via run_training.sh's trailing "$@".
apptainer exec --nv --cleanenv \
  --pwd "$REPO" \
  --env ENV_FILE="$ENV_FILE",NUM_GPUS="$NUM_GPUS" \
  -B /projects/m000102/juliomz -B /scratch/m000102/juliomz \
  "$IMAGE" \
  scripts/run_training.sh --run_id "episode-$SLURM_JOB_ID" "$@" &

# Wait, re-entering if a trapped signal (USR1) interrupts wait while the child
# is still alive; preserve the child's real exit code for the FAIL email.
child=$!
while true; do
  wait "$child"; rc=$?
  [ "$rc" -lt 128 ] && break             # child exited on its own (0 or crash)
  kill -0 "$child" 2>/dev/null || break  # interrupted by signal and child is gone
done
exit "$rc"
