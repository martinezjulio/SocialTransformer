# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SocialTransformer is a transformer-based model for learning demonstrator beliefs and bandit payoffs from sequential observations. It models social learning where a learner observes multiple demonstrators interacting with multi-armed bandits across multiple rounds, predicting both bandit arm payoffs and individual demonstrator belief states as Beta distributions.

## Environment Setup

```bash
source /home/julio/miniforge3/etc/profile.d/conda.sh
conda activate SocialTransformer
```

Python 3.11, PyTorch 2.9.1+cu130, wandb for experiment tracking.

## Directory Configuration

Out-of-repo directories (data, experiments) are configured via a `.env` file in the repo root. Copy `.env.example` to `.env` and set absolute paths for your machine:

```bash
cp .env.example .env
# Edit .env with your paths:
#   TRAINING_DATA_ROOT_DIR=/path/to/data
#   OUTPUT_DATA_DIR=/path/to/experiments/dev
#   EXPERIMENT_DATA_PATH=/path/to/data/experiment_data/.../preprocessed_data.csv
```

All shell scripts source `.env` automatically. Dataset names (e.g., `demonstrations_data_04`) are specified per-script/sweep and resolved against `TRAINING_DATA_ROOT_DIR` by `train.py`.

## Common Commands

```bash
# Generate dataset (auto-increments dataset name)
bash scripts/generate_simulation.sh

# Train model
bash scripts/run_training.sh

# Evaluate on test set
bash scripts/run_eval.sh

# Run inference on human experiment data
bash scripts/run_inference.sh

# Continue training from checkpoint
./scripts/continue_training.sh --checkpoint <run_id> --source wandb --epochs 150

# Hyperparameter sweep
wandb sweep src/Sweeps/Sweep_Version_*.yaml
wandb agent <sweep-id>

# Resume incomplete sweep runs after crash/preemption
./scripts/complete_sweep.sh <sweep_id>
```

There is no test suite; validation is done via `eval.py` against held-out test sets.

## Architecture

### Pipeline: Data Generation → Training → Evaluation → Inference

1. **`src/generate_simulations.py`** — Creates synthetic multi-round bandit scenarios with expert/novice demonstrators. Outputs sharded `.pt` files with `manifest.json` per split.

2. **`src/dataloader.py`** — Loads shards via `ShardedExamplesDataset`, tokenizes events into 5 channels (`bandit_ids`, `agent_ids`, `action_ids`, `reward_ids`, `special_ids`) via `BanditSequenceDataset`. Agent IDs are remapped: 0=learner, 1+=demonstrators. A QUERY token is appended for prediction readout.

3. **`src/models/BetaModel.py`** — `SBIPosteriorTransformer` with three stages:
   - `EventEmbedder`: sums per-field embeddings + positional encoding
   - `CausalTransformer`: standard causal attention with optional custom masks
   - Prediction heads outputting Beta distribution parameters for: bandit payoffs (θ), per-demonstrator beliefs (α,β), and optionally demonstrator experience counts

4. **`src/train.py`** — Training loop with **segment-level masking**: each example randomly selects 1..N visible segments, masks future segments in attention, and predicts for the bandit corresponding to the last visible segment's round. This teaches the model to predict from variable context lengths.

5. **`src/eval/eval.py`** / **`src/eval/eval2.py`** — Evaluates model at different context cutoffs (number of visible segments), measuring theta prediction and demonstrator belief accuracy.

6. **`src/inference.py`** — Loads a trained model and runs Thompson sampling on human experiment data, producing hybrid predictions (real demonstrator data + model-simulated learner actions).

### Key Design Decisions

- **Beta distribution outputs**: The model predicts Beta(α,β) parameters, not point estimates. Theta loss uses Beta NLL; belief loss uses log-MSE, KL, or cross-entropy against true pre-episode counts.
- **Per-demonstrator predictions**: Beliefs are predicted separately per demonstrator (not averaged), capturing individual expertise differences.
- **Profile methods**: `no_profile`, `learned_profile`, `agent_pooling`, `cross_attention` — different ways to incorporate demonstrator identity into predictions.
- **Head input configurations**: Each prediction head (`theta`, `demo_beliefs`, `experience`) can use `query_only`, `profiles_only`, or `query_and_profiles`.
- **Masking strategies**: `all_bandits` vs `current_bandit_only` — whether to predict for all bandits or only the current round's bandit.

### Run Management

Experiment outputs are organized under `$OUTPUT_DATA_DIR/` (set in `.env`):

```
$OUTPUT_DATA_DIR/
  training_runs/              # Per-run directories keyed by wandb ID
    wandb_<id>/
      checkpoints/            # .pt files (local or downloaded from wandb)
      wandb/                  # wandb logs
  eval/                       # Evaluation outputs (overwrites on re-run)
    data/                     # evaluation CSVs
    figures/                  # evaluation plots
    metadata/                 # config.yaml, run.yaml
  inference/                  # Inference outputs (overwrites on re-run)
    data/                     # inference CSVs
    metadata/                 # config.yaml, run.yaml
  analysis/                   # Analysis outputs (overwrites on re-run)
    data/                     # analysis CSVs
    figures/                  # analysis plots
    metadata/                 # config.yaml, run.yaml
```

Training runs get per-run directories (keyed by wandb ID) under `training_runs/`. Eval, inference, and analysis use flat directories that overwrite on re-run.

### Checkpoint Loading

Three sources supported across eval/inference scripts:
- `local` — file path to `.pt` checkpoint
- `wandb_artifact` — `entity/project/artifact:alias`
- `wandb_sweep` — `entity/project/sweep_id` (loads best model)

Downloaded wandb checkpoints are cached under `training_runs/wandb_<id>/checkpoints/` for reuse across eval/inference runs. Subsequent loads skip the download if cached files exist.

Checkpoints store model weights, optimizer/scheduler state, training config, and epoch info. Model architecture params are auto-detected from checkpoint config.

## Data Format

Events are tuples `(bandit_id, agent_id, action, reward)`:
- `agent_id`: 0-N for demonstrators, 99 for learner, -1 for BANDIT_SEP tokens
- Datasets stored as sharded `.pt` files under `$TRAINING_DATA_ROOT_DIR/demonstrations_data_NN/` with train/val/test splits
- Dataset parameters (K, num_demonstrators, num_rounds) are auto-detected from `manifest.json`

## R/Quarto Analysis Environment

The R/Quarto analysis environment uses `renv` (at repo root) for R package management and the conda environment's R runtime. Quarto reports live in `reports/`, R scripts in `src/analysis/`.

### Setup

```bash
# One-time: install R packages
bash scripts/setup_r_env.sh

# Render analysis (requires inference data in $OUTPUT_DATA_DIR/inference/data/)
bash scripts/run_analysis.sh
```

### Structure

```
repo root/
├── .Rprofile                          # Activates renv on R startup
├── _quarto.yml                        # Anchors Quarto project root
├── renv.lock                          # Locked R dependencies
├── renv/                              # renv bootstrap files
├── reports/
│   └── analysis_report.qmd           # Main Quarto analysis document
└── src/
    └── analysis/
        ├── reward_plots.R             # R helper scripts (plots, preprocessing)
        ├── entropy_plots.R
        ├── choice_frequencies.R
        ├── correlations.R
        ├── individual_behavior.R
        ├── aggregate_behavior_distribution.R
        ├── average_behavior_distribution.R
        ├── individual_behavior_distribution.R
        ├── experience_plots.R
        ├── behavior_distribution_average.R
        ├── plots.R
        └── preprocess.R
```

### Key Details

- **R runtime**: Installed via conda (`r-base`) in the `SocialTransformer` environment.
- **R packages**: Managed by `renv` (at repo root, independent of conda). Run `renv::restore()` to install from `renv.lock`.
- **Data paths**: `analysis_report.qmd` reads `OUTPUT_DATA_DIR` and `EXPERIMENT_DATA_PATH` from the root `.env` file via the `dotenv` R package.
- **`here::here()`** resolves to the repo root (anchored by `.git/`). No `.here` sentinel needed.
- **Figures**: All plots save to `$OUTPUT_DATA_DIR/analysis/figures/` when called from `analysis_report.qmd` (via `fig_dir` parameter). R scripts fall back to `src/analysis/figures/` (gitignored) when run standalone.
- **Preprocessing**: `src/analysis/preprocess.R` reads/writes experiment data from `$EXPERIMENT_DATA_DIR`.

## W&B Integration

- Project: `social-belief-modeling`, Entity: `juliomartinez-individual`
- Sweep configs in `src/Sweeps/Sweep_Version_*.yaml` (Bayesian optimization)
- `src/complete_sweep.py` recovers crashed/killed/failed sweep runs
