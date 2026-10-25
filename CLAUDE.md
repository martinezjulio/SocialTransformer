# CLAUDE.md

SocialTransformer: a transformer that observes demonstrators playing multi-armed
bandits over several rounds and predicts arm payoffs plus each demonstrator's
belief state as Beta distributions. Human experiment data is analyzed by the
pipeline in `src/analysis` (new, under construction) and the frozen legacy notebook.

## Environment

```bash
source /home/julio/miniforge3/etc/profile.d/conda.sh
conda activate SocialTransformer
```

## Out-of-repo directories (.env contract)

Data and experiment outputs live outside the repo. Copy `.env.example` to `.env`
and set absolute paths; every script in `scripts/` sources it. Dataset names
(e.g. `demonstrations_data_04`) are given per script or sweep and resolved by
`train.py` against `TRAINING_DATA_ROOT_DIR`. See `.env.example` for the
variable list.

## Running things

```bash
bash scripts/run_training.sh      # train
bash scripts/run_eval.sh          # evaluate on synthetic test set
bash scripts/run_inference.sh     # run model on human experiment data
pytest tests/                     # invariants on generated outputs; see tests/README.md
wandb sweep src/Sweeps/Sweep_Version_*.yaml && wandb agent <sweep-id>
bash scripts/setup_r_env.sh       # one-time R packages via renv
python scripts/run_notebook.py --version pilot_E   # legacy notebook, batch; see its docstring
bash scripts/run_quarto_analysis_report.sh         # legacy Quarto report in reports/
```

## Design decisions (do not silently revisit)

- **Segment-level masking**: each training example picks 1..N visible segments,
  masks later segments in attention, and predicts for the last visible
  segment's bandit. Evaluation sweeps this cutoff. See `src/train.py`.
- **Beta outputs, not point estimates**: heads emit Beta(alpha, beta). Theta
  uses Beta NLL; beliefs use log-MSE, KL, or cross-entropy against true
  pre-episode counts. See `src/models/losses.py`.
- **Per-demonstrator beliefs**: predicted separately per demonstrator, never
  averaged.
- **Model factory boundary**: `build_model(model_type, config)` in
  `src/models/__init__.py` is the only constructor used by train, eval, and
  inference; it accepts the pre-rename type strings as aliases so old
  checkpoints still load.
- **Legacy analysis is frozen and isolated**: `notebooks/analysis.py` with its
  package `notebooks/legacy_analysis/` writes to `$OUTPUT_DATA_DIR/legacy_analysis/`;
  the new pipeline (`src/analysis`) writes to `$OUTPUT_DATA_DIR/analysis/`. Neither
  imports the other (`tests/test_no_cross_imports.py`).

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
