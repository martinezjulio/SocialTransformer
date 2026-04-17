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
#   EXPERIMENT_DATA_DIR=/path/to/experiment_data
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

Lightweight pytest suite under `tests/`. Run with `pytest tests/` inside the
`SocialTransformer` conda env. Tests validate invariants of generated
artifacts under `$OUTPUT_DATA_DIR` and auto-skip when required CSVs are
missing. Broader validation still happens via `eval.py` against held-out
test sets.

## Architecture

### Pipeline: Data Generation → Training → Evaluation → Inference

1. **`src/generate_simulations.py`** — Creates synthetic multi-round bandit scenarios with expert/novice demonstrators. Outputs sharded `.pt` files with `manifest.json` per split.

2. **`src/data/dataloader.py`** — Loads shards via `ShardedExamplesDataset`, tokenizes events into 5 channels (`bandit_ids`, `agent_ids`, `action_ids`, `reward_ids`, `special_ids`) via `BanditSequenceDataset`. Agent IDs are remapped: 0=learner, 1+=demonstrators. A QUERY token is appended for prediction readout. **`src/data/transforms.py`** provides `BeliefNormalizer` for target normalization.

3. **`src/models/BetaModel.py`** — `SBIPosteriorTransformer` with three stages:
   - `EventEmbedder`: sums per-field embeddings + positional encoding
   - `CausalTransformer`: standard causal attention with optional custom masks
   - Prediction heads outputting Beta distribution parameters for: bandit payoffs (θ), per-demonstrator beliefs (α,β), and optionally demonstrator experience counts
   - **`src/models/losses.py`** — Loss functions (Beta NLL, belief losses, experience loss). **`src/models/load_checkpoint.py`** — Checkpoint loading from local/wandb sources.

4. **`src/train.py`** — Training loop with **segment-level masking**: each example randomly selects 1..N visible segments, masks future segments in attention, and predicts for the bandit corresponding to the last visible segment's round. This teaches the model to predict from variable context lengths.

5. **`src/eval/eval.py`** — Evaluates model at different context cutoffs (number of visible segments), measuring theta prediction and demonstrator belief accuracy.

6. **`src/run_inference.py`** — Loads a trained model and runs inference on human experiment data via a unified cutoff sweep over the full tokenized session. Human mode uses the real learner actions; model mode runs a pre-segment forward pass per learner segment, feeds the resulting belief into Thompson sampling, and splices the sampled actions back into the shared tensors so later cutoffs condition on the simulated play. Emits two output schemas per run: `experiment_trial_predictions.csv` (per-trial behavior rows containing both human and model learner segments) and `experiment_cutoff_{human,model}_predictions.csv` (per-cutoff posterior rows for the N_total experience plots).

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
  eval/                       # All eval + inference outputs (flat layout, overwrites on re-run)
    data/                     # CSVs — naming: {source}_{shape}[_{mode}]_predictions.csv
                              #   test_trial_model_predictions.csv        (from eval.py)
                              #   test_cutoff_model_predictions.csv       (from eval.py)
                              #   experiment_trial_predictions.csv        (from run_inference.py — both human & model rows)
                              #   experiment_cutoff_human_predictions.csv (from run_inference.py)
                              #   experiment_cutoff_model_predictions.csv (from run_inference.py)
    figures/                  # eval_report + hybrid_model_pipeline_eval figures
                              #   hybrid_model_pipeline_sessions/ — per-session PNGs
    metadata/                 # eval_run.yaml, eval_config.yaml (from eval.py)
                              # inference_run.yaml, inference_config.yaml (from run_inference.py)
  analysis/                   # Analysis outputs (overwrites on re-run)
    data/                     # analysis CSVs
    figures/                  # analysis plots
    metadata/                 # config.yaml, run.yaml
```

Training runs get per-run directories (keyed by wandb ID) under `training_runs/`. Eval, inference, and analysis use flat directories that overwrite on re-run. Both `eval.py` and `run_inference.py` now write to `eval/` (one shared flat layout). File naming encodes source (`test` = synthetic test set, `experiment` = human data) and shape (`trial` = per-trial rows, `cutoff` = per-cutoff posterior rows). The `experiment_trial_predictions.csv` file contains both `learner_segment_input ∈ {human, model}` rows in the same file and therefore carries no mode suffix.

### Checkpoint Loading

Three sources supported via `src/models/load_checkpoint.py` across eval/inference scripts:
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

The R/Quarto analysis environment uses `renv` (at repo root) for R package management and the conda environment's R runtime. Quarto reports live in `reports/`, R scripts in `reports/r/`.

### Setup

```bash
# One-time: install R packages
bash scripts/setup_r_env.sh

# Render analysis (requires experiment_trial_predictions.csv in $OUTPUT_DATA_DIR/eval/data/)
bash scripts/run_analysis.sh
```

### Structure

```
repo root/
├── .Rprofile                          # Activates renv on R startup
├── _quarto.yml                        # Anchors Quarto project root
├── renv.lock                          # Locked R dependencies
├── renv/                              # renv bootstrap files
└── reports/
    ├── analysis_report.qmd                # Behavior analysis (reward, choice, entropy, correlations) — reads experiment_trial_predictions.csv
    ├── eval_report.qmd                     # N_total experience comparison (3x2 plot) — reads the per-cutoff CSVs in eval/data/
    ├── hybrid_model_pipeline_eval.qmd      # Per-session + cross-session rank-space diagnostics (issue #11) — reads experiment_trial_predictions.csv + experiment_cutoff_model_predictions.csv (or the test_* pair when data_source=test)
    └── r/
        ├── reward_plots.R                  # R helper scripts (plots, preprocessing)
        ├── entropy_plots.R
        ├── choice_frequencies.R
        ├── correlations.R
        ├── individual_behavior.R
        ├── aggregate_behavior_distribution.R
        ├── average_behavior_distribution.R
        ├── individual_behavior_distribution.R
        ├── experience_plots.R
        ├── behavior_distribution_average.R
        ├── hybrid_pipeline_plots.R         # Hybrid pipeline rank-space plotting helpers (issue #11)
        └── preprocess.R
```

### Key Details

- **R runtime**: Installed via conda (`r-base`) in the `SocialTransformer` environment.
- **R packages**: Managed by `renv` (at repo root, independent of conda). Run `renv::restore()` to install from `renv.lock`.
- **Data paths**: `analysis_report.qmd` reads `OUTPUT_DATA_DIR` and `EXPERIMENT_DATA_DIR` from the root `.env` file via the `dotenv` R package; the preprocessed CSV path is built from `EXPERIMENT_DATA_DIR` + report params (`experiment_name`, `version`).
- **`here::here()`** resolves to the repo root (anchored by `.git/`). No `.here` sentinel needed.
- **Figures**: All plots save to `$OUTPUT_DATA_DIR/analysis/figures/` when called from `analysis_report.qmd` (via `fig_dir` parameter). R scripts fall back to `reports/r/figures/` (gitignored) when run standalone.
- **Preprocessing**: `reports/r/preprocess.R` reads/writes experiment data from `$EXPERIMENT_DATA_DIR`.

## W&B Integration

- Project: `social-belief-modeling`, Entity: `juliomartinez-individual`
- Sweep configs in `src/Sweeps/Sweep_Version_*.yaml` (Bayesian optimization)
- `src/complete_sweep.py` recovers crashed/killed/failed sweep runs
