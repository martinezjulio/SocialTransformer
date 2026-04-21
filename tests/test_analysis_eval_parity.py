"""Regression test for issue #9 — analysis and eval pipelines must agree.

The analysis report aggregates per-step `reward_generated` values from
`experiment_trial_predictions.csv` into per-(game_sequence, game_index) means.
The eval report aggregates per-segment `segment_total_reward` values from
`experiment_cutoff_model_predictions.csv` into per-(game_sequence, cutoff) means.

These are two paths to the same numbers. If they ever diverge, a filter
or aggregation bug has been introduced somewhere (see issue #9 for history).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Round-structure knowledge: in a watch-first session with 3 rounds of
# [bandit_sep, D1, L1, D2, L2, query] (6 segments/round), the 12 game_index
# values map 1:1 to cutoffs as:
#   round r: game_index 4r   -> cutoff 6r+2  (D1)
#            game_index 4r+1 -> cutoff 6r+3  (L1)
#            game_index 4r+2 -> cutoff 6r+4  (D2)
#            game_index 4r+3 -> cutoff 6r+5  (L2)
GAME_INDEX_TO_CUTOFF = {
    4 * r + slot: 6 * r + 1 + (slot + 1)
    for r in range(3)
    for slot in range(4)
}


def _analysis_means(hybrid_csv: Path) -> pd.DataFrame:
    """Analysis-side: group by session/game, sum step rewards, mean over sessions.

    Mirrors `reports/r/reward_plots.R` for the model-mode play segment.
    """
    df = pd.read_csv(hybrid_csv)
    d = df[
        (df["learner_segment_input"] == "model")
        & (df["beta_method"] == "theta_betas")
    ].copy()
    per_game = (
        d.groupby(
            [
                "anonymous_id",
                "session_id",
                "session_index",
                "game_sequence",
                "game_index",
                "game",
                "round_index",
                "rollout_id",
            ],
            dropna=False,
            as_index=False,
        )["reward_generated"]
        .sum()
        .rename(columns={"reward_generated": "game_total"})
    )
    return per_game.groupby(
        ["game_sequence", "game_index", "game", "round_index"],
        dropna=False,
        as_index=False,
    ).agg(mean_reward=("game_total", "mean"), n=("game_total", "size"))


def _eval_means(model_mode_csv: Path) -> pd.DataFrame:
    """Eval-side: per-session-cutoff mean across rollouts, then mean across sessions.

    Mirrors `reports/eval_report.qmd`'s `prepare_source` + reward summary.
    """
    df = pd.read_csv(model_mode_csv)
    d = df[
        (df["learner_segment_input"] == "model")
        & (df["beta_method"] == "theta_betas")
        & (df["segment_order"] == "watch-first")
        & (df["segment_type"].isin(["demo", "learner"]))
        & (df["segment_total_reward"].notna())
    ].copy()
    per_session = d.groupby(
        [
            "session_id",
            "beta_method",
            "cutoff",
            "segment_type",
            "segment_order",
            "game_sequence",
            "round_index",
        ],
        dropna=False,
        as_index=False,
    )["segment_total_reward"].mean()
    return per_session.groupby(
        ["game_sequence", "cutoff", "segment_type", "round_index"],
        dropna=False,
        as_index=False,
    ).agg(mean_reward=("segment_total_reward", "mean"), n=("segment_total_reward", "size"))


def test_analysis_and_eval_agree_on_every_game(hybrid_csv, model_mode_csv):
    """Per-(game_sequence, game_index) means match between analysis and eval."""
    analysis = _analysis_means(hybrid_csv)
    eval_side = _eval_means(model_mode_csv)

    analysis = analysis.assign(
        cutoff=analysis["game_index"].astype(int).map(GAME_INDEX_TO_CUTOFF)
    )
    merged = analysis.merge(
        eval_side[["game_sequence", "cutoff", "mean_reward", "n"]],
        on=["game_sequence", "cutoff"],
        suffixes=("_analysis", "_eval"),
    )

    # Every game_index has a matching cutoff.
    assert len(merged) == len(analysis), (
        f"{len(analysis) - len(merged)} game_index rows lost their cutoff match"
    )

    merged["diff"] = (merged["mean_reward_analysis"] - merged["mean_reward_eval"]).abs()
    worst = merged.sort_values("diff", ascending=False).head(3)
    max_diff = float(merged["diff"].max())

    # Allow a tiny float tolerance for CSV round-tripping; the two paths
    # should be mathematically identical, not just close.
    assert max_diff < 1e-9, (
        f"Analysis and eval means diverge (max abs diff = {max_diff}).\n"
        f"Top 3 diffs:\n{worst.to_string(index=False)}"
    )


def test_sample_counts_match(hybrid_csv, model_mode_csv):
    """Same sessions should populate both pipelines — n must match per (seq, game)."""
    analysis = _analysis_means(hybrid_csv)
    eval_side = _eval_means(model_mode_csv)

    # Analysis groups by (session, rollout) so its count is N× the eval count
    # (which collapses rollouts first). Derive N from the CSV so the test
    # adapts to whatever NUM_ROLLOUTS the run used.
    n_rollouts = pd.read_csv(hybrid_csv, usecols=["rollout_id"])["rollout_id"].nunique()
    assert n_rollouts >= 1, "hybrid_csv has no rollout_id values"

    analysis = analysis.assign(
        cutoff=analysis["game_index"].astype(int).map(GAME_INDEX_TO_CUTOFF)
    )
    merged = analysis.merge(
        eval_side[["game_sequence", "cutoff", "n"]],
        on=["game_sequence", "cutoff"],
        suffixes=("_analysis", "_eval"),
    )
    mismatches = merged[merged["n_analysis"] != merged["n_eval"] * n_rollouts]
    assert mismatches.empty, (
        f"Row counts diverge between analysis (per session-rollout) and "
        f"eval (per session × {n_rollouts} rollouts):\n{mismatches.to_string(index=False)}"
    )
