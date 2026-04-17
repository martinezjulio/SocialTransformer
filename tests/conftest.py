"""Shared pytest fixtures.

Paths are resolved from the .env file at the repo root so tests agree with
the rest of the pipeline on where outputs live.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")


@pytest.fixture(scope="session")
def output_data_dir() -> Path:
    val = os.environ.get("OUTPUT_DATA_DIR")
    if not val:
        pytest.skip("OUTPUT_DATA_DIR not set in .env")
    return Path(val)


@pytest.fixture(scope="session")
def hybrid_csv(output_data_dir: Path) -> Path:
    p = output_data_dir / "eval" / "data" / "experiment_trial_predictions.csv"
    if not p.exists():
        pytest.skip(f"{p} not present — run scripts/run_inference.sh first")
    return p


@pytest.fixture(scope="session")
def model_mode_csv(output_data_dir: Path) -> Path:
    p = output_data_dir / "eval" / "data" / "experiment_cutoff_model_predictions.csv"
    if not p.exists():
        pytest.skip(f"{p} not present — run scripts/run_inference.sh first")
    return p
