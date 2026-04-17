# Tests

Lightweight pytest suite. Tests run against whatever pilot's inference outputs
currently live under `$OUTPUT_DATA_DIR` — they're not self-contained; they
validate invariants of the generated artifacts.

## Running

```bash
conda activate SocialTransformer
pytest tests/
```

Tests that require CSVs under `$OUTPUT_DATA_DIR` are auto-skipped when those
files are missing, so a fresh clone with no inference run yet will report skips
rather than failures.

## What's here

- `test_analysis_eval_parity.py` — ensures the analysis pipeline (reading
  `hybrid_model_predictions.csv` and aggregating per-game) produces the same
  per-game mean reward as the eval pipeline (reading
  `model_mode_inference_predictions.csv` and aggregating per-cutoff). A failure
  here means issue #9 has regressed: something is dropping sessions or
  aggregating differently between the two reports.
