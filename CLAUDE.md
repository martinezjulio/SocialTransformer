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
#   DATA_DIR=/path/to/data
#   TRAINING_DATA_DIR=/path/to/data/demonstrations_data_04
#   EXPERIMENTS_DIR=/path/to/experiments/dev
```

All shell scripts source `.env` automatically. Python scripts use `EXPERIMENTS_DIR` (via `utils.py`) and fall back to `TRAINING_DATA_DIR` for dataset paths.

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

5. **`src/eval.py`** / **`src/eval2.py`** — Evaluates model at different context cutoffs (number of visible segments), measuring theta prediction and demonstrator belief accuracy.

6. **`src/inference.py`** — Loads a trained model and runs Thompson sampling on human experiment data, producing hybrid predictions (real demonstrator data + model-simulated learner actions).

### Key Design Decisions

- **Beta distribution outputs**: The model predicts Beta(α,β) parameters, not point estimates. Theta loss uses Beta NLL; belief loss uses log-MSE, KL, or cross-entropy against true pre-episode counts.
- **Per-demonstrator predictions**: Beliefs are predicted separately per demonstrator (not averaged), capturing individual expertise differences.
- **Profile methods**: `no_profile`, `learned_profile`, `agent_pooling`, `cross_attention` — different ways to incorporate demonstrator identity into predictions.
- **Head input configurations**: Each prediction head (`theta`, `demo_beliefs`, `experience`) can use `query_only`, `profiles_only`, or `query_and_profiles`.
- **Masking strategies**: `all_bandits` vs `current_bandit_only` — whether to predict for all bandits or only the current round's bandit.

### Run Management

All scripts write outputs to `$EXPERIMENTS_DIR/runs/<run_id>/` where `EXPERIMENTS_DIR` is set in `.env`.

Run IDs: `wandb_<id>` (with wandb) or `uuid_<10char>` (without). Each run directory contains `checkpoints/`, `logs/`, `results/`, `config.yaml`, `run.yaml`.

### Checkpoint Loading

Three sources supported across eval/inference scripts:
- `local` — file path to `.pt` checkpoint
- `wandb_artifact` — `entity/project/artifact:alias`
- `wandb_sweep` — `entity/project/sweep_id` (loads best model)

Checkpoints store model weights, optimizer/scheduler state, training config, and epoch info. Model architecture params are auto-detected from checkpoint config.

## Data Format

Events are tuples `(bandit_id, agent_id, action, reward)`:
- `agent_id`: 0-N for demonstrators, 99 for learner, -1 for BANDIT_SEP tokens
- Datasets stored as sharded `.pt` files under `$DATA_DIR/demonstrations_data_NN/` with train/val/test splits
- Dataset parameters (K, num_demonstrators, num_rounds) are auto-detected from `manifest.json`

## W&B Integration

- Project: `social-belief-modeling`, Entity: `juliomartinez-individual`
- Sweep configs in `src/Sweeps/Sweep_Version_*.yaml` (Bayesian optimization)
- `src/complete_sweep.py` recovers crashed/killed/failed sweep runs
