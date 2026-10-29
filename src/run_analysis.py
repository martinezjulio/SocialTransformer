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
    checkpoint: str = None   # transformer checkpoint for the simulation checks; None = the one the inference CSVs came from


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


@stage("choice")
def run_choice(p: Paths) -> None:
    """Fit the choice-model registry on this version's social trials: tables per (source, κ, own) built once and
    cached, fits cached by table hash, PSIS-LOO ladder vs HOS, verdicts for S1a–S3b (2 SE each), the claims S1–S3 (all parts
    required) and S4 (δ_N 95% ETI), numbers.json."""
    import json
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    import pandas as pd
    from analysis.compare import by_cell, ladder, loo, paired
    from analysis.data import load_sessions, trial_data
    from analysis.fitting import run_many
    from analysis.policies import REGISTRY, belief, grids
    from analysis.thompson import table, table_path
    T, out = trial_data(load_sessions(p.eval_dir, p.version)), p.analysis_dir / "choice"
    out.mkdir(exist_ok=True)

    builder, pending = ThreadPoolExecutor(1), {}   # builds queue on the CPU threads, in registry order, while the GPU fits

    def tab(source, spec):   # queue the build on first use; return the cache path
        k, w, c = grids(spec)
        args = (f"{source}_{int(spec.kappa)}_{int(spec.own)}", *belief(T, source), T.own_seg, T.own_prev, p.analysis_dir / "tables")
        path = table_path(*args, kappa=k, lam_w=w, lam_c=c)
        if path not in pending:
            pending[path] = builder.submit(lambda *a, **kw: table(*a, **kw).shape, *args, kappa=k, lam_w=w, lam_c=c)
        return path
    paths = {n: (tab(m.source, m), tab("novice", m) if m.mixture != "none" else None) for n, m in REGISTRY.items()}
    fits = {}
    for n, (a, b) in paths.items():   # fit cache keyed by the model's own table hashes; wait only for this model's tables
        pending[a].result(), b and pending[b].result()
        fits.update(run_many([n], p.eval_dir, p.version, paths, a.stem.split("_")[-1] + (f"-{b.stem.split('_')[-1]}" if b else ""), out / "fits"))
    builder.shutdown()
    loos = {n: loo(f) for n, f in fits.items() if f.passed}
    lad = ladder(loos, "HOS", T.part).join(pd.DataFrame({n: f.diagnostics for n, f in fits.items()}).T)
    lad.to_csv(out / "ladder.csv")
    rounds = {"round 1": T.round_pm < 0, "round 2": T.round_pm > 0}   # the ladder per round (Fig 3b, 3d): ΔELPD vs HOS, paired SE, share
    pd.concat({n: by_cell(l, loos["HOS"], T, rounds) for n, l in loos.items()}, names=["model", "round"]).to_csv(out / "ladder_by_round.csv")
    pd.DataFrame({n: paired(l, loos["HOS"], T.part)["per_participant"] for n, l in loos.items()}).rename_axis("participant").to_csv(out / "elpd_per_participant.csv")   # Fig S: dots behind every paired SE
    d = fits["Switch-by-round"].posterior["delta_N"].reshape(-1) if "Switch-by-round" in loos else np.array([np.nan])
    rows = {"S4": dict(estimate=d.mean(), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), rule="95% ETI > 0")}
    for name, a, b in [("S1a", "HOS", "recent"), ("S1b", "HOS", "novice"), ("S2a", "HOS", "naive-recent"), ("S2b", "HOS", "naive-expert"),
                       ("S2c", "HOS", "naive"), ("S3a", "HS", "HO"), ("S3b", "HOS", "HO"),
                       ("switch_round_term", "Switch-by-round", "Switch"), ("own_on_top", "HOS", "HS")]:
        q = paired(loos[a], loos[b], T.part) if a in loos and b in loos else dict(delta=np.nan, se=np.nan)
        rows[name] = dict(estimate=q["delta"], lo=q["delta"] - 2 * q["se"], hi=q["delta"] + 2 * q["se"], rule="ΔELPD > 2 SE" if name[0] == "S" else "reported")
    for r in rows.values():
        r["verdict"] = "" if r["rule"] == "reported" else "not evaluable" if np.isnan(r["estimate"]) else "supported" if r["lo"] > 0 else "not supported"
    for claim in ("S1", "S2", "S3"):   # a claim holds when every part does; per-part verdicts stay reported
        parts = [r["verdict"] for k, r in rows.items() if k.startswith(claim) and len(k) == 3]
        rows[claim] = dict(estimate=np.nan, lo=np.nan, hi=np.nan, rule="all parts > 2 SE", verdict="supported" if all(v == "supported" for v in parts) else "not evaluable" if "not evaluable" in parts else "not supported")
    pd.DataFrame(rows).T.to_csv(out / "verdicts.csv")
    json.dump({f"{k}_{f}": float(v) for k, r in rows.items() for f, v in r.items() if f != "verdict" and f != "rule"} | {f"elpd_{n}": l.elpd for n, l in loos.items()},
              open(out / "numbers.json", "w"), indent=1)
    print(lad[["elpd", "delta", "se", "n_bad_k", "rung", "seconds"]].round(1).to_string(), "\n", pd.DataFrame(rows).T.to_string())


def human_tables(p: Paths, T, name: str) -> list:
    """Paths of the human tables a registry model reads, in the choice stage's cache (its fit is keyed by their hashes)."""
    from analysis.policies import REGISTRY, belief, grids
    from analysis.thompson import table_path
    m = REGISTRY[name]
    k, w, c = grids(m)
    return [table_path(f"{s}_{int(m.kappa)}_{int(m.own)}", *belief(T, s), T.own_seg, T.own_prev, p.analysis_dir / "tables", k, w, c)
            for s in ((m.source, "novice") if m.mixture != "none" else (m.source,))]


def choice_fit(p: Paths, T, name: str):
    """The choice stage's cached fit of a registry model."""
    import pickle
    from analysis.fitting import cache_path
    return pickle.load(open(cache_path(name, "-".join(t.stem.split("_")[-1] for t in human_tables(p, T, name)), p.analysis_dir / "choice" / "fits"), "rb"))


@stage("checks")
def run_checks(p: Paths) -> None:
    """Simulation checks of the fitted models: δ_N recovery, model recovery for S1a, S2a and S4, free-running PPC. The transformer
    is the one the inference CSVs came from (eval metadata) unless ``checkpoint`` overrides it. Every unit is cached
    under checks/, so a rerun only fills what is missing. Outputs: recovery.csv, model_recovery.csv, confusion.csv,
    ppc.csv, numbers.json."""
    import json
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from analysis.checks import COMPARISONS, confusion, model_recovery, ppc, recover_delta, recovery_verdict, split_half, templates
    from analysis.data import load_sessions, trial_data
    from analysis.policies import REGISTRY
    from inference.readout import load_model
    T, out = trial_data(load_sessions(p.eval_dir, p.version)), p.analysis_dir / "checks"
    out.mkdir(exist_ok=True)

    tables = lambda name: human_tables(p, T, name)
    posterior = lambda name: choice_fit(p, T, name).posterior
    cfg = yaml.safe_load(open(p.eval_dir / "metadata" / "inference_config.yaml"))
    model, _, norm = load_model(p.checkpoint or cfg["checkpoint_path"], None if p.checkpoint else cfg["wandb_artifact"], None, "cuda" if torch.cuda.is_available() else "cpu")
    pools = templates(p.preprocessed)
    frames = {f["session_id"].iloc[0]: f for pool in pools.values() for f in pool}
    rec = recover_delta(posterior("Switch-by-round"), pools, model, norm, out / "recovery")
    rec.to_csv(out / "recovery.csv", index=False)
    mr = model_recovery({n: posterior(n) for pair in COMPARISONS.values() for n in pair}, pools, model, norm, out / "model_recovery")
    mr.to_csv(out / "model_recovery.csv", index=False)
    confusion(mr).to_csv(out / "confusion.csv")
    bands = pd.concat({n: ppc(REGISTRY[n], posterior(n), T, frames, model, norm, out / "ppc") for n in ("HOS", "Switch-by-round")}, names=["model", "target"])
    halves = pd.concat({f"{n} split_half": split_half(REGISTRY[n], T, tuple(np.load(t) for t in tables(n)) + (None,) * (2 - len(tables(n))), frames, model, norm, out / "split_half")
                        for n in ("HOS", "Switch-by-round")}, names=["model", "target"])
    bands = pd.concat([bands, halves])
    bands.to_csv(out / "ppc.csv")
    numbers = {**{f"recovery_{k}": v for k, v in recovery_verdict(rec).items()}, "model_recovery_correct": int(mr["correct"].sum()), "model_recovery_n": len(mr),
               **{f"ppc_{m}_{t}_inside": bool(r["inside"]) for (m, t), r in bands.iterrows()}}
    json.dump(numbers, open(out / "numbers.json", "w"), indent=1)
    print(rec.round(3).to_string(), "\n", confusion(mr).to_string(), "\n", bands.round(4).to_string(), "\n", numbers)


@stage("unseen")
def run_unseen(p: Paths) -> None:
    """Unseen-expert section (exploratory): the readout-entering-play table, the copying belief against the unseen-expert
    readout, the ELPD split (expert readout vs recent readout, i.e. the novice's on these trials: S1a's models, cut into cells) and the expert-vs-copying ELPD on the after-novice cells from the
    choice stage's fits, the exploit statistic; numbers.json with the headline cells."""
    import json
    from analysis.compare import loo
    from analysis.data import load_sessions, trial_data
    from analysis.unseen import copying_table, elpd_by_cell, exploit_table, readout_table
    T, out = trial_data(load_sessions(p.eval_dir, p.version)), p.analysis_dir / "unseen"
    out.mkdir(exist_ok=True)
    tabs = dict(readout_table=readout_table(T), copying_table=copying_table(T), exploit=exploit_table(T))
    loos = {n: loo(choice_fit(p, T, n)) for n in ("HOS", "recent", "naive-recent")}
    tabs["elpd_split"], tabs["copying_elpd"] = elpd_by_cell(loos["HOS"], loos["recent"], T), elpd_by_cell(loos["HOS"], loos["naive-recent"], T)
    for name, t in tabs.items():
        t.to_csv(out / f"{name}.csv")
    r, x = tabs["readout_table"], tabs["exploit"]
    numbers = {f"unseen_expert_round{k}_{stat}": float(r.loc[("expert", False, k), (stat, "mean")]) for k in (1, 2) for stat in ("confidence", "rank_corr", "favourite_payoff")}
    numbers |= {f"exploit_unseen_round{k}": float(x.loc[(False, k), ("exploit", "mean")]) for k in (1, 2)} | {"exploit_chance": float(x[("chance", "mean")].mean())}
    numbers |= {f"split_{n.replace(' ', '_').replace(',', '')}_{f}": float(v) for n, row in tabs["elpd_split"].iterrows() for f, v in row.items() if f != "share"}
    json.dump(numbers, open(out / "numbers.json", "w"), indent=1)
    print(*(f"{n}\n{t.round(3).to_string()}" for n, t in tabs.items()), numbers, sep="\n\n")


@stage("figures")
def run_figures(p: Paths) -> None:
    """Every figure module on the stage outputs and the cached fits: Fig 1 inset, Figs 2–4, the supplement; PDF + PNG under
    figures/. Headless: the Agg backend unless MPLBACKEND says otherwise."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    from analysis.figures.fig1_inset import fig1_inset
    from analysis.figures.fig2 import fig2
    from analysis.figures.fig3 import fig3
    from analysis.figures.fig4 import fig4
    from analysis.figures.supp import supp
    paths = (fig1_inset(p.eval_dir, p.version, p.analysis_dir) + fig2(p.analysis_dir, p.preprocessed) + fig3(p.analysis_dir)
             + fig4(p.analysis_dir, p.eval_dir, p.version) + supp(p.analysis_dir, p.preprocessed.parent))
    print("figures:", len(paths), "files under", p.analysis_dir / "figures")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", required=True, help="e.g. pilot_E")
    ap.add_argument("--stages", default=",".join(STAGES), help=f"comma-separated subset of: {', '.join(STAGES)}")
    ap.add_argument("--checkpoint", default=None, help="transformer checkpoint for the checks stage (default: the inference run's)")
    args = ap.parse_args()
    if "OUTPUT_DATA_DIR" not in os.environ:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    names = args.stages.split(",")
    unknown = [n for n in names if n not in STAGES]
    if unknown:
        ap.error(f"unknown stage(s) {unknown}; known: {', '.join(STAGES)}")
    p = Paths(args.version, Path(os.environ["EXPERIMENT_DATA_DIR"]) / args.version / "preprocessed_participant_data.csv",
              Path(os.environ["OUTPUT_DATA_DIR"]) / "eval" / args.version, Path(os.environ["OUTPUT_DATA_DIR"]) / "analysis" / args.version, args.checkpoint)
    p.analysis_dir.mkdir(parents=True, exist_ok=True)
    for name in [n for n in STAGES if n in names]:   # pipeline order regardless of how they were listed
        STAGES[name](p)


if __name__ == "__main__":
    main()
