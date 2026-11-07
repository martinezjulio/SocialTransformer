# SocialTransformer

A transformer that watches demonstrators play multi-armed bandits over several
rounds and predicts, from a QUERY token, both the arm payoffs and **each
demonstrator's belief state** as Beta distributions. Training examples mask all
but the first `k` segments of an episode, so one episode supervises predictions
at every level of evidence.

The second half of the project is an analysis pipeline for the human experiment:
it fits a registry of choice models to participants' play, runs simulation-based
checks on those fits, and emits every number the paper cites as a LaTeX macro —
no figure or statistic is typed by hand.

**Preregistration.** The full study is preregistered on OSF at
<https://osf.io/etxw9> (registered 5 November 2026, before data collection;
embargoed while data are collected). The registered state of this repository is
tag `prereg-2026-11-05`; the document is `reports/preregistration/` and the
checkpoint it names is `epoch_119_best.pt`, sha256
`9f3586bd30704526da7042c87f5492a6109867b5eb50dc8b4ddf211f538decac`.

Internals — architecture, data format, losses, masking, per-stage analysis
details and cluster operations — live in [`ARCHITECTURE.md`](ARCHITECTURE.md).
Design decisions not to revisit are in
[`CLAUDE.md`](CLAUDE.md).

---

## Setup

```bash
# 1. Python environment (the name SocialTransformer is what every script activates)
conda create -n SocialTransformer python=3.11
conda activate SocialTransformer
pip install -r requirements.txt           # a CUDA 13 freeze; repin torch/jax for your GPU
                                          # on Sherlock: requirements-sherlock.txt

# 2. R packages for the R analysis stages (one-time)
bash scripts/setup_r_env.sh               # renv::restore() from renv.lock

# 3. Paths
cp .env.example .env                      # then edit the absolute paths
```

Afterwards, `source "$CONDA_SETUP" && conda activate SocialTransformer` in any
shell where you call the Python or R entry points directly. The `scripts/*.sh`
wrappers do it for you.

### The `.env` contract

Data and outputs live **outside** the repo. Every script in `scripts/` sources
`.env`; the Python and R entry points load it themselves when the variables are
unset. Set these to absolute paths:

| variable | what it points at |
|---|---|
| `CONDA_SETUP` | your `conda.sh`, e.g. `~/miniforge3/etc/profile.d/conda.sh` |
| `TRAINING_DATA_ROOT_DIR` | where synthetic datasets are generated and read from |
| `OUTPUT_DATA_DIR` | checkpoints, eval output, analysis output |
| `EXPERIMENT_DATA_DIR` | human experiment data, one folder per version |
| `WANDB_ENTITY`, `WANDB_PROJECT` | training/sweep logging |

Optional: `SLURM_*` and `N_TABLE_BLOCKS` (cluster submission), `CONTAINER`
(Apptainer image the pipeline re-executes inside), `RENV_PATHS_CACHE`. See
[`.env.example`](.env.example) for the annotated list, and
`.env.cluster.example` for the offline-wandb cluster variant
(`ENV_FILE=.env.cluster bash scripts/<script>.sh`).

Human experiment data and analysis outputs are keyed by **version only** —
`$EXPERIMENT_DATA_DIR/<version>/` and `$OUTPUT_DATA_DIR/analysis/<version>/`.
Synthetic dataset names (`demonstrations_data_04`) are given per script and
resolved against `TRAINING_DATA_ROOT_DIR`.

> **There is no `configs/` directory.** Each `scripts/*.sh` holds its step's
> configuration as a block of shell constants near the top, and expands them
> into the command line. To change hyperparameters or dataset knobs, edit the
> script (or override the few env vars it reads). The only YAML configs are the
> wandb sweep specs in `src/Sweeps/`.

---

## Pipeline

End to end:

```bash
bash scripts/generate_simulation.sh                      # synthetic training data
bash scripts/run_training.sh                             # train
bash scripts/run_eval.sh                                 # eval on synthetic test + human inference
VERSION=pilot_E Rscript src/r/preprocess_participant_data.R   # human data -> preprocessed CSV
bash scripts/run_analysis.sh --version pilot_E           # analysis stages, Python + R
pytest tests/                                            # invariants on what was produced
```

Data flows `TRAINING_DATA_ROOT_DIR/<dataset>` → `OUTPUT_DATA_DIR/training_runs/<run_id>/checkpoints/`
→ `OUTPUT_DATA_DIR/eval/[<version>/]` → `OUTPUT_DATA_DIR/analysis/<version>/` →
`analysis/<version>/paper/`.

### Generate data

```bash
bash scripts/generate_simulation.sh
LEARNER_POLICY=naive-learner bash scripts/generate_simulation.sh   # single policy instead of mixed
```

Writes `$TRAINING_DATA_ROOT_DIR/demonstrations_data_NN/` — three HuggingFace
Arrow splits (`train`, `val`, `test`) plus a root `metadata.json`. `NN`
auto-increments from the highest existing dataset, and an existing split
directory is never overwritten. Dataset shape (`K`, number of demonstrators,
priors) is read from `metadata.json` downstream, so no training flag repeats it.

### Train

```bash
bash scripts/run_training.sh                                          # wraps torchrun src/train.py
DATASET_NAME=demonstrations_data_04 NUM_GPUS=4 bash scripts/run_training.sh
MODEL_TYPE=social_profile_cross_attention bash scripts/run_training.sh
bash scripts/run_training.sh --run_id episode-123                     # extra flags pass through
```

Model types are built through `build_model()` in `src/models/__init__.py`.
`social` is the one in use; the three `social_profile_*` variants are
exploratory and were not used for any published result. `python src/train.py
--help` lists every flag.

#### Reproducing the model used in the analysis

Every inference run feeding the analysis pipeline used one checkpoint —
`training_runs/wandb_0i8yhqm7/checkpoints/epoch_119_best.pt` (wandb run
`0i8yhqm7`, single GPU). **The committed defaults in `scripts/run_training.sh`
are that run's hyperparameters**, so retraining the same architecture is just:

```bash
bash scripts/run_training.sh      # keep NUM_GPUS=1: batch size is per rank
```

The published configuration, recorded here so it survives later edits to that
script:

| | |
|---|---|
| model | `social` — plain `SocialTransformer`, no demonstrator profiles, no experience head |
| architecture | `d_model` 256, `nhead` 4, `num_layers` 8, `dim_feedforward` 1024, `dropout` 0.2 |
| data | `demonstrations_data_02`, `seed` 42 |
| optimisation | `adamw`, `lr` 5e-5, `batch_size` 128 (world size 1), `weight_decay` 0.0, `epochs` 500 |
| scheduler | `plateau`, `patience` 15, `factor` 0.7, `min_lr` 1e-8, `warmup_epochs` 2 |
| losses | `env_theta_loss` `bce`, `demo_belief_loss` `cross_entropy`, `demo_loss_weight` 1.0, `experience_loss_weight` 1.0 |
| masking | `all_bandits` |
| early stopping | overfitting threshold 15%, patience 50, window 15 — best epoch was **119** |

The `report` stage also records that checkpoint's `sha256` in the paper
provenance block, so a rebuilt model can be checked against the one the results
came from.

To continue a run:

```bash
bash scripts/continue_training.sh --checkpoint <run_id> --source wandb --epochs 150
```

Sweeps:

```bash
wandb sweep src/Sweeps/Sweep_Version_EnvThetas.yaml   # prints entity/project/sweep_id
wandb agent <sweep-id>                                 # repeat for parallel workers
python src/analyze_sweep.py --sweep_id <sweep-id>      # best config per objective
bash scripts/complete_sweep.sh <sweep-id>              # resume crashed/killed runs
```

### Evaluate and run inference on human data

```bash
EVAL_ONLY=1 bash scripts/run_eval.sh          # synthetic test set only
bash scripts/run_eval.sh                      # ... then calls run_inference.sh
VERSION=pilot_E bash scripts/run_inference.sh # human experiment data only
```

Set the checkpoint (`CHECKPOINT_PATH` or `WANDB_ARTIFACT`) inside the script.
Both write to `$OUTPUT_DATA_DIR/eval/` and **overwrite on re-run**. CSVs in
`eval/data/` are named `{source}_{shape}[_{mode}]_predictions.csv`: source is
`test` (synthetic) or `experiment` (human), shape is `trial` (per-trial rows) or
`cutoff` (per-cutoff posterior rows), mode is `human` or `model`.

### Preprocess the human data

```bash
VERSION=pilot_E Rscript src/r/preprocess_participant_data.R
```

A **separate step, not an analysis stage.** Reads the anonymized export from
`$EXPERIMENT_DATA_DIR/<version>/` and writes
`preprocessed_participant_data.csv`, `exclusions.csv`,
`demographics_summary.csv` and a timestamp-free public CSV. Every exclusion rule
lives in this script; before the rules it drops exact duplicate rows, which a
retried upload of one event leaves behind (counted on the console, not in the
ledger).

### Analyse

```bash
bash scripts/run_analysis.sh --version pilot_E
bash scripts/run_analysis.sh --version pilot_E --stages games,choice
bash scripts/run_analysis.sh --version pilot_E --stages power      # only when named
```

Takes two inputs per version: `preprocessed_participant_data.csv` and the
inference CSVs in `$OUTPUT_DATA_DIR/eval/<version>/`. Stages run in a fixed
order — `games`, `descriptive`, `partner_choice`, `inference-check`, `choice`,
`checks`, `unseen`, `figures`, `report` — mixing Python
([`src/run_analysis.py`](src/run_analysis.py)) and R (`src/r/`). Each writes one
folder under `$OUTPUT_DATA_DIR/analysis/<version>/`, and `report` emits `paper/`
(`results.tex` macros, `results.md`, tables, figures) or fails naming any
missing number.

Belief tables and fits are cached by content hash, so reruns are cheap. But
stages **overwrite their outputs unconditionally**, so a reduced run needs
`--dev`; without it the script refuses to start when `N_DATASETS`,
`N_MODEL_RECOVERY`, `N_PPC_DRAWS` or `N_SIMS` is set.

Running at full size on a cluster, and the Sherlock/Apptainer setup, are covered
in `ARCHITECTURE.md` §7. The short version:

```bash
bash scripts/setup_sherlock.sh                     # one-time, inside an sh_dev session
bash scripts/slurm/submit.sh --version pilot_E     # dependent job arrays
bash scripts/with_env.sh <command>                 # anything else, inside the container
```

---

## Tests

```bash
pytest tests/
```

The suite is **not self-contained**: it checks invariants of whatever inference
artifacts currently exist under `$OUTPUT_DATA_DIR` and *skips* rather than fails
on a fresh clone. Set `EXPERIMENT_VERSION` to pick the version (default
`pilot_D`). See [`tests/README.md`](tests/README.md).

---

## Layout

```
src/
├── generate_simulations.py   # synthetic data generator
├── train.py                  # training loop, segment masking, loss assembly
├── run_inference.py          # trained model over human sessions
├── run_analysis.py           # analysis driver (Python stages)
├── data/                     # Arrow encoding, dataloader, belief normalization
├── models/                   # build_model factory, variants, heads, losses, masking
├── eval/                     # metrics on the synthetic test split
├── inference/                # readout, Thompson sampling, session simulation
├── analysis/                 # stages, policy registry, fitting, numbers manifest, figures
├── r/                        # preprocessing + the R analysis stages
└── Sweeps/                   # wandb sweep configs

scripts/                      # one shell entry point per step; holds the config constants
tests/                        # invariants on generated outputs
reports/                      # Quarto reports and the preregistration
```
