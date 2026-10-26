"""Driver of the analysis pipeline: run registered stages for one experiment version.

    python src/run_analysis.py --version pilot_E [--stages games,inference-check]

Inputs:  $EXPERIMENT_DATA_DIR/<version>/preprocessed_participant_data.csv and $OUTPUT_DATA_DIR/eval/<version>/
         (run_inference.py). Outputs: $OUTPUT_DATA_DIR/analysis/<version>/<stage>/. The R stages
         (descriptive, partner_choice, power) run from scripts/run_analysis.sh.
Stages import what they need when they run, so the driver itself never loads torch or JAX.
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

STAGES = {}   # name -> function(Paths); insertion order is pipeline order


def stage(name):
    def register(fn):
        STAGES[name] = fn
        return fn
    return register


@dataclass(frozen=True)
class Paths:
    version: str
    preprocessed: Path   # all sessions, post-exclusion (src/r/preprocess_participant_data.R)
    eval_dir: Path       # inference CSVs + metadata for this version
    analysis_dir: Path   # every stage writes under here


@stage("games")
def run_games(p: Paths) -> None:
    from analysis.games import write_play_games
    print("games:", write_play_games(p.preprocessed, p.analysis_dir))


@stage("inference-check")
def run_inference_check(p: Paths) -> None:
    """The inference CSVs were produced from this version's preprocessed file and cover all its social sessions."""
    import pandas as pd
    import yaml
    cfg = yaml.safe_load((p.eval_dir / "metadata" / "inference_config.yaml").read_text())
    assert Path(cfg["experiment_data_path"]).resolve() == p.preprocessed.resolve(), \
        f"inference ran on {cfg['experiment_data_path']}, not {p.preprocessed}: rerun scripts/run_inference.sh"
    shots = pd.read_csv(p.preprocessed, usecols=["anonymous_id", "session_id", "game_sequence"], low_memory=False)
    social = shots[shots["game_sequence"] != "play-play-play-play"]
    trial = pd.read_csv(p.eval_dir / "data" / "experiment_trial_predictions.csv", usecols=["anonymous_id", "session_id"])
    assert set(trial["session_id"]) == set(social["session_id"]), "inference sessions differ from the social sessions in the preprocessed file"
    assert set(trial["anonymous_id"]) == set(social["anonymous_id"])
    print(f"inference-check: {trial['anonymous_id'].nunique()} participants, {trial['session_id'].nunique()} social sessions, "
          f"checkpoint {Path(cfg['checkpoint_path']).name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", required=True, help="e.g. pilot_E")
    ap.add_argument("--stages", default=",".join(STAGES), help=f"comma-separated subset of: {', '.join(STAGES)}")
    args = ap.parse_args()
    if "OUTPUT_DATA_DIR" not in os.environ:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    names = args.stages.split(",")
    unknown = [n for n in names if n not in STAGES]
    if unknown:
        ap.error(f"unknown stage(s) {unknown}; known: {', '.join(STAGES)}")
    p = Paths(args.version, Path(os.environ["EXPERIMENT_DATA_DIR"]) / args.version / "preprocessed_participant_data.csv",
              Path(os.environ["OUTPUT_DATA_DIR"]) / "eval" / args.version, Path(os.environ["OUTPUT_DATA_DIR"]) / "analysis" / args.version)
    p.analysis_dir.mkdir(parents=True, exist_ok=True)
    for name in [n for n in STAGES if n in names]:   # pipeline order regardless of how they were listed
        STAGES[name](p)


if __name__ == "__main__":
    main()
