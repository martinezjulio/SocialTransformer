#!/usr/bin/env bash
# Push an OFFLINE Marlowe training run to wandb. Run from a LOGIN node (needs
# outbound internet + wandb auth via `wandb login` or WANDB_API_KEY).
#
# It does two things:
#   1. `wandb sync` the offline shards  -> metrics / history / config (one run,
#      because every requeue shared --run_id episode-$SLURM_JOB_ID).
#   2. Reopen the run and attach the dataset-metadata + best-checkpoint artifacts
#      (deferred from training; see train.py's `offline` gating). The best
#      checkpoint then downloads for eval/inference via load_checkpoint with
#      wandb_artifact="<entity>/<project>/social-transformer-run-<run_id>:best".
#
# Usage:
#   scripts/wandb_sync.sh <run_id>                 # e.g. episode-123456
#   ENV_FILE=.env.cluster scripts/wandb_sync.sh episode-123456
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 1 ]; then
  echo "Usage: $0 <run_id>   (e.g. episode-\$SLURM_JOB_ID)" >&2
  exit 1
fi
RUN_ID="$1"

ENV_FILE="${ENV_FILE:-.env.cluster}"
if [ -f "$ENV_FILE" ]; then
  set -a; source "$ENV_FILE"; set +a
else
  echo "ERROR: $ENV_FILE not found. Copy .env.cluster.example to $ENV_FILE." >&2
  exit 1
fi

source "$CONDA_SETUP"
conda activate SocialTransformer

# Syncing/relogging must reach wandb — override the offline default from .env.cluster.
export WANDB_MODE=online
export PYTHONPATH="src:${PYTHONPATH:-}"
# Artifact base name must match train.py save_checkpoint (ARTIFACT_NAME in run_training.sh).
export ARTIFACT_NAME="${ARTIFACT_NAME:-social-transformer}"

RUN_DIR="${OUTPUT_DATA_DIR}/training_runs/wandb_${RUN_ID}"
[ -d "$RUN_DIR" ] || { echo "ERROR: run dir not found: $RUN_DIR" >&2; exit 1; }

echo "== 1/2 syncing offline history from $RUN_DIR/wandb =="
shopt -s nullglob
shards=("$RUN_DIR"/wandb/offline-run-*)
if [ ${#shards[@]} -eq 0 ]; then
  echo "WARN: no offline-run-* shards under $RUN_DIR/wandb" >&2
else
  wandb sync "${shards[@]}" || echo "WARN: wandb sync reported errors (continuing)"
fi

echo "== 2/2 attaching dataset + best-checkpoint artifacts to run $RUN_ID =="
RUN_ID="$RUN_ID" RUN_DIR="$RUN_DIR" python - <<'PY'
import os, glob, json
from pathlib import Path
import wandb
from train import log_dataset_artifact

run_id  = os.environ["RUN_ID"]
run_dir = Path(os.environ["RUN_DIR"])

run = wandb.init(
    id=run_id, dir=str(run_dir), resume="must",
    entity=os.environ.get("WANDB_ENTITY") or None,
    project=os.environ.get("WANDB_PROJECT") or None,
)
cfg = dict(run.config)

# Dataset metadata artifact. cfg["dataset_name"] is the resolved absolute path
# (train.py joins it against TRAINING_DATA_ROOT_DIR before logging config).
dataset_path = cfg.get("dataset_name")
meta_file = os.path.join(dataset_path, "metadata.json") if dataset_path else None
if meta_file and os.path.isfile(meta_file):
    with open(meta_file) as f:
        dataset_meta = json.load(f)
    log_dataset_artifact(run, dataset_path, dataset_meta)
    print(f"  + dataset artifact from {dataset_path}")
else:
    print(f"  ! dataset metadata missing ({meta_file}); skipping dataset artifact")

# Best checkpoint — name/aliases mirror save_checkpoint so load_checkpoint(:best) resolves.
artifact_base = os.environ["ARTIFACT_NAME"]
best = sorted(glob.glob(str(run_dir / "checkpoints" / "epoch_*_best.pt")))
if best:
    fp = best[-1]
    epoch = int(Path(fp).stem.split("_")[1])      # epoch_NNN_best -> NNN
    art = wandb.Artifact(name=f"{artifact_base}-run-{run.id}", type="model")
    art.add_file(fp, name=f"epoch_{epoch:03d}.pt")
    run.log_artifact(art, aliases=[f"epoch_{epoch}", "best", "latest"])
    print(f"  + best checkpoint artifact from {fp}")
else:
    print("  ! no epoch_*_best.pt found; skipping checkpoint artifact")

run.finish()
print("DONE")
PY

echo "== done: ${WANDB_ENTITY:-<entity>}/${WANDB_PROJECT:-<project>}/${ARTIFACT_NAME}-run-${RUN_ID}:best =="
