# CLAUDE.md

SocialTransformer: a transformer that observes demonstrators playing multi-armed
bandits over several rounds and predicts arm payoffs plus each demonstrator's
belief state as Beta distributions. Human experiment data is analysed by the
pipeline in `src/analysis` (see below).

## Environment

```bash
source /home/julio/miniforge3/etc/profile.d/conda.sh
conda activate SocialTransformer
```

## Out-of-repo directories (.env contract)

Data and experiment outputs live outside the repo. Copy `.env.example` to `.env`
and set absolute paths; every script in `scripts/` sources it. Human experiment data
and analysis outputs are keyed by version only: `$EXPERIMENT_DATA_DIR/<version>/` and
`$OUTPUT_DATA_DIR/analysis/<version>/` (no experiment-name level). Dataset names
(e.g. `demonstrations_data_04`) are given per script or sweep and resolved by
`train.py` against `TRAINING_DATA_ROOT_DIR`. See `.env.example` for the
variable list.

## Running things

```bash
bash scripts/run_training.sh      # train
bash scripts/run_eval.sh          # evaluate on synthetic test set
bash scripts/run_inference.sh     # run model on human experiment data
bash scripts/run_analysis.sh --version pilot_E   # analysis pipeline; see its header for stages and --dev
bash scripts/setup_sherlock.sh && bash scripts/slurm/submit.sh --version pilot_E   # full size on Sherlock; README §6
pytest tests/                     # invariants on generated outputs; see tests/README.md
wandb sweep src/Sweeps/Sweep_Version_*.yaml && wandb agent <sweep-id>
bash scripts/setup_r_env.sh       # one-time R packages via renv
```

## Analysis pipeline (`src/analysis`, driver `src/run_analysis.py`)

Two inputs per version: `$EXPERIMENT_DATA_DIR/<version>/preprocessed_participant_data.csv`
(from `src/r/preprocess_participant_data.R`) and the inference CSVs in `$OUTPUT_DATA_DIR/eval/<version>/`.
Stages run in a fixed order (Python in `run_analysis.py`, R in `src/r/`) and each writes one folder under
`$OUTPUT_DATA_DIR/analysis/<version>/`. Belief tables and fits are cached by content hash and the
simulation checks by unit, so reruns are cheap; stages overwrite their outputs unconditionally, so a
reduced run needs `--dev`. The paper's numbers are never typed by hand: stages write `numbers.json`,
`numbers.py` holds the manifest of every cited number, and the `report` stage emits `paper/`
(`results.tex` macros, `results.md`, tables) or fails naming any missing number. Every figure
(paper, slides, webpage) comes from the `figures` stage into the one `figures/` folder; ΔELPD is
drawn as model minus ROS there, while the numbers keep the registered ROS-minus-rival direction. Model names
follow the ROS convention (R repeat, O own experience, S social belief) in `policies.py`'s registry.

## Design decisions (do not silently revisit)

- **Segment-level masking**: each training example picks 1..N visible segments,
  masks later segments in attention, and predicts for the last visible
  segment's bandit. Evaluation sweeps this cutoff. See `src/train.py`.
- **Beta outputs, not point estimates**: the belief heads emit Beta(alpha,
  beta), scored against true pre-episode counts by log-MSE, KL, expected-value
  MSE or cross-entropy (`--demo_belief_loss`). Theta takes `--env_theta_loss`:
  `bce` on logits (the default, and what the analysis model used) or `beta_nll`
  on a predicted Beta. See `src/models/losses.py`.
- **Per-demonstrator beliefs**: predicted separately per demonstrator, never
  averaged.
- **Model factory boundary**: `build_model(model_type, config)` in
  `src/models/__init__.py` is the only constructor used by train, eval, and
  inference; it accepts the pre-rename type strings as aliases so old
  checkpoints still load.
- **Untracked working material**: `notebooks/`, `.claude/` and everything in
  `reports/` except the preregistration are gitignored on purpose. They stay on
  disk for local use only; do not add them back.

## Checkpoints

`BaseSocialTransformer.load_checkpoint` (in `src/models/base.py`) takes exactly
one of `local_path` or `wandb_artifact` (`entity/project/artifact:alias`).
wandb downloads are cached under
`$OUTPUT_DATA_DIR/training_runs/wandb_<id>/checkpoints/` and reused.

## Eval output naming

`eval.py` and `run_inference.py` both write to `$OUTPUT_DATA_DIR/eval/` and
overwrite on re-run. CSVs in `eval/data/` are named
`{source}_{shape}[_{mode}]_predictions.csv`: source is `test` (synthetic) or
`experiment` (human), shape is `trial` (per-trial rows) or `cutoff` (per-cutoff
posterior rows), mode is `human` or `model`. `experiment_trial_predictions.csv`
holds both modes in one file, distinguished by `learner_segment_input`.
