"""Driver of the analysis pipeline: run registered stages for one experiment version.

    python src/run_analysis.py --version pilot_E [--stages games,inference-check]
    python src/run_analysis.py --version pilot_E --list-units | --unit fit:HOS    # one cluster unit (scripts/slurm)

Inputs:  $EXPERIMENT_DATA_DIR/<version>/preprocessed_participant_data.csv and $OUTPUT_DATA_DIR/eval/<version>/
         (run_inference.py). Outputs: $OUTPUT_DATA_DIR/analysis/<version>/<stage>/. The R stages
         (descriptive, partner_choice, power) run from scripts/run_analysis.sh.
Stages import what they need when they run, so the driver itself never loads torch or JAX.
"""

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

STAGES = {}   # name -> function(Paths); insertion order is pipeline order
N_TABLE_BLOCKS = int(os.environ.get("N_TABLE_BLOCKS", 1))   # row blocks per human table in the cluster units (scripts/slurm)


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
    recorded = Path(cfg["experiment_data_path"])   # the metadata records the inference machine's paths: compare file and version folder
    assert (recorded.parent.name, recorded.name) == (p.preprocessed.parent.name, p.preprocessed.name), \
        f"inference ran on {recorded}, not {p.preprocessed}: rerun scripts/run_inference.sh"
    shots = pd.read_csv(p.preprocessed, usecols=["anonymous_id", "session_id", "game_sequence"], low_memory=False)
    social = shots[shots["game_sequence"] != "play-play-play-play"]
    trial = pd.read_csv(p.eval_dir / "data" / "experiment_trial_predictions.csv", usecols=["anonymous_id", "session_id"])
    assert set(trial["session_id"]) == set(social["session_id"]), "inference sessions differ from the social sessions in the preprocessed file"
    assert set(trial["anonymous_id"]) == set(social["anonymous_id"])
    print(f"inference-check: {trial['anonymous_id'].nunique()} participants, {trial['session_id'].nunique()} social sessions, "
          f"checkpoint {Path(cfg['checkpoint_path']).name}")


def verdict(estimate: float, lo: float, hi: float) -> str:
    """The registration's four outcomes (prereg 4.3) of one interval: supported when it lies above zero, contradicted
    below, not supported when it covers zero, not evaluable when there is no estimate (a fit failed the gate)."""
    return "not evaluable" if math.isnan(estimate) else "supported" if lo > 0 else "contradicted" if hi < 0 else "not supported"


def claim_verdict(parts: list) -> str:
    """A claim from its parts: supported when every part is, else contradicted when any part is (never masked by a
    not-evaluable part), else not evaluable when any part is, else not supported."""
    return ("supported" if all(v == "supported" for v in parts) else "contradicted" if "contradicted" in parts
            else "not evaluable" if "not evaluable" in parts else "not supported")


@stage("choice")
def run_choice(p: Paths) -> None:
    """Fit the choice-model registry on this version's social trials: tables per (source, κ, own) built once and
    cached, fits cached by table hash, PSIS-LOO ladder vs HOS, the registration's four-outcome verdicts for S1a–S3b (2 SE each),
    the claims S1–S3 (from their parts) and S4 (δ_N 95% ETI), numbers.json."""
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    import pandas as pd
    from analysis.numbers import write_numbers
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
        r["verdict"] = "" if r["rule"] == "reported" else verdict(r["estimate"], r["lo"], r["hi"])
    for claim in ("S1", "S2", "S3"):   # per-part verdicts stay reported beside the claim's
        rows[claim] = dict(estimate=np.nan, lo=np.nan, hi=np.nan, rule="all parts > 2 SE", verdict=claim_verdict([r["verdict"] for k, r in rows.items() if k.startswith(claim) and len(k) == 3]))
    pd.DataFrame(rows).T.to_csv(out / "verdicts.csv")
    write_numbers(out, {f"{k}_{f}": float(v) for k, r in rows.items() for f, v in r.items() if f != "verdict" and f != "rule"}
                  | {f"{k}_verdict": r["verdict"] for k, r in rows.items() if r["verdict"]} | {f"elpd_{n}": l.elpd for n, l in loos.items()})
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


def table_arrays(p: Paths, T, name: str) -> tuple:
    """The model's human tables loaded, padded to (table, second table or None) as ``split_half`` takes them."""
    import numpy as np
    tabs = [np.load(t) for t in human_tables(p, T, name)]
    return (tabs[0], tabs[1] if len(tabs) > 1 else None)


def checks_setup(p: Paths) -> tuple:
    """What the checks stage and its units share: trials, the checks dir, the choice stage's posterior by model name, the
    transformer (the inference run's unless ``checkpoint`` overrides it) and its normaliser, the session template pools
    by condition and the human shot frames by session id."""
    import torch
    import yaml
    from analysis.checks import templates
    from analysis.data import checkpoint_path, load_sessions, trial_data
    from inference.readout import load_model
    T, out = trial_data(load_sessions(p.eval_dir, p.version)), p.analysis_dir / "checks"
    out.mkdir(exist_ok=True)
    cfg = yaml.safe_load(open(p.eval_dir / "metadata" / "inference_config.yaml"))
    model, _, norm = load_model(p.checkpoint or str(checkpoint_path(p.eval_dir)), None if p.checkpoint else cfg["wandb_artifact"], None, "cuda" if torch.cuda.is_available() else "cpu")
    pools = templates(p.preprocessed)
    return T, out, lambda name: choice_fit(p, T, name).posterior, model, norm, pools, {f["session_id"].iloc[0]: f for pool in pools.values() for f in pool}


@stage("checks")
def run_checks(p: Paths) -> None:
    """Simulation checks of the fitted models: δ_N recovery, model recovery for S1a, S2a and S4, free-running PPC. The transformer
    is the one the inference CSVs came from (eval metadata) unless ``checkpoint`` overrides it. Every unit is cached
    under checks/, so a rerun only fills what is missing. Outputs: recovery.csv, model_recovery.csv, confusion.csv,
    ppc.csv, numbers.json."""
    import pandas as pd
    from analysis.checks import COMPARISONS, PPC_MODELS, confusion, model_recovery, ppc, recover_delta, recovery_verdict, split_half
    from analysis.numbers import write_numbers
    from analysis.policies import REGISTRY
    T, out, posterior, model, norm, pools, frames = checks_setup(p)
    rec = recover_delta(posterior("Switch-by-round"), pools, model, norm, out / "recovery")
    rec.to_csv(out / "recovery.csv", index=False)
    mr = model_recovery({n: posterior(n) for pair in COMPARISONS.values() for n in pair}, pools, model, norm, out / "model_recovery")
    mr.to_csv(out / "model_recovery.csv", index=False)
    confusion(mr).to_csv(out / "confusion.csv")
    bands = pd.concat({n: ppc(REGISTRY[n], posterior(n), T, frames, model, norm, out / "ppc") for n in PPC_MODELS}, names=["model", "target"])
    halves = pd.concat({f"{n} split_half": split_half(REGISTRY[n], T, table_arrays(p, T, n), frames, model, norm, out / "split_half") for n in PPC_MODELS}, names=["model", "target"])
    bands = pd.concat([bands, halves])
    bands.to_csv(out / "ppc.csv")
    numbers = {**{f"recovery_{k}": v for k, v in recovery_verdict(rec).items()}, "model_recovery_correct": int(mr["correct"].sum()), "model_recovery_n": len(mr),
               **{f"ppc_{m}_{t}_inside": bool(r["inside"]) for (m, t), r in bands.iterrows()}}
    write_numbers(out, numbers)
    print(rec.round(3).to_string(), "\n", confusion(mr).to_string(), "\n", bands.round(4).to_string(), "\n", numbers)


@stage("unseen")
def run_unseen(p: Paths) -> None:
    """Unseen-expert section (exploratory): the readout-entering-play table, the copying belief against the unseen-expert
    readout, the ELPD split (expert readout vs recent readout, i.e. the novice's on these trials: S1a's models, cut into cells) and the expert-vs-copying ELPD on the after-novice cells from the
    choice stage's fits, the exploit statistic; numbers.json with the headline cells."""
    from analysis.compare import loo
    from analysis.data import load_sessions, trial_data
    from analysis.numbers import write_numbers
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
    write_numbers(out, numbers)
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


@stage("report")
def run_report(p: Paths) -> None:
    """Last: every stage's numbers merged with the provenance block and checked against the manifest (stops naming any
    missing or extra number), then paper/: results.tex, results.md, tables/, and a copy of figures/."""
    import shutil
    from analysis.data import load_sessions, trial_data
    from analysis.report import merge_numbers, provenance, write_results, write_tables
    T, out = trial_data(load_sessions(p.eval_dir, p.version)), p.analysis_dir / "paper"
    numbers = merge_numbers(p.analysis_dir, provenance(p.preprocessed, p.eval_dir, int(T.part.max() + 1), len(set(T.session)), len(T.act)))
    paths = write_results(numbers, out) + write_tables(p.analysis_dir, out)
    (out / "figures").mkdir(exist_ok=True)
    figures = [shutil.copy2(f, out / "figures" / f.name) for f in sorted((p.analysis_dir / "figures").glob("fig*"))]
    print(f"report: {len(numbers)} numbers, {[q.name for q in paths]}, {len(figures)} figure files under {out}")


def units(p: Paths) -> dict:
    """Every cluster unit (scripts/slurm) as name -> (level, thunk), in dependency order: 0 table blocks (CPU), 1 table
    assembly, 2 human fits, 3 recovery and model-recovery datasets, split-half fits and PPC draws (GPU), 4 split-half
    draws. Each fills one cache piece (one writer per file); the stages then run on the full caches."""
    from analysis.checks import COMPARISONS, N_DATASETS, N_MODEL_RECOVERY, N_PPC_DRAWS, PPC_MODELS, model_recovery, ppc, recover_delta, split_half
    from analysis.data import load_sessions, trial_data
    from analysis.fitting import run_many
    from analysis.policies import REGISTRY, belief, grids
    from analysis.thompson import table
    T, u = trial_data(load_sessions(p.eval_dir, p.version)), {}
    tab = lambda s, m, **kw: table(f"{s}_{int(m.kappa)}_{int(m.own)}", *belief(T, s), T.own_seg, T.own_prev, p.analysis_dir / "tables", *grids(m), n_blocks=N_TABLE_BLOCKS, **kw).shape
    check = lambda call: call(*checks_setup(p))   # a checks unit: the stage's setup, then one call restricted to the unit
    for n, m in REGISTRY.items():
        for s in (m.source, "novice") if m.mixture != "none" else (m.source,):
            for b in range(N_TABLE_BLOCKS):
                u[f"table:{s}_{int(m.kappa)}_{int(m.own)}:{b}/{N_TABLE_BLOCKS}"] = (0, lambda s=s, m=m, b=b: tab(s, m, block=b))
            u[f"assemble:{s}_{int(m.kappa)}_{int(m.own)}"] = (1, lambda s=s, m=m: tab(s, m))
        t = human_tables(p, T, n)
        u[f"fit:{n}"] = (2, lambda n=n, t=t: run_many([n], p.eval_dir, p.version, {n: (t[0], t[1] if len(t) > 1 else None)}, "-".join(x.stem.split("_")[-1] for x in t), p.analysis_dir / "choice" / "fits")[n].diagnostics)
    for d in range(N_DATASETS):
        u[f"recovery:{d}"] = (3, lambda d=d: check(lambda T, out, post, model, norm, pools, frames: recover_delta(post("Switch-by-round"), pools, model, norm, out / "recovery", datasets=[d])))
    for c, pair in COMPARISONS.items():
        for side in pair:
            for d in range(N_MODEL_RECOVERY):
                u[f"model_recovery:{c}:{side}:{d}"] = (3, lambda c=c, side=side, d=d, pair=pair: check(lambda T, out, post, model, norm, pools, frames: model_recovery(
                    {n: post(n) for n in pair}, pools, model, norm, out / "model_recovery", units=[(c, side, d)])))
    for n in PPC_MODELS:
        u[f"split_fit:{n}"] = (3, lambda n=n: check(lambda T, out, post, model, norm, pools, frames: split_half(REGISTRY[n], T, table_arrays(p, T, n), frames, model, norm, out / "split_half", draws=[])))
        for i in range(N_PPC_DRAWS):
            u[f"ppc:{n}:{i}"] = (3, lambda n=n, i=i: check(lambda T, out, post, model, norm, pools, frames: ppc(REGISTRY[n], post(n), T, frames, model, norm, out / "ppc", draws=[i])))
            u[f"split_half:{n}:{i}"] = (4, lambda n=n, i=i: check(lambda T, out, post, model, norm, pools, frames: split_half(REGISTRY[n], T, table_arrays(p, T, n), frames, model, norm, out / "split_half", draws=[i])))
    return u


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", required=True, help="e.g. pilot_E")
    ap.add_argument("--stages", default=",".join(STAGES), help=f"comma-separated subset of: {', '.join(STAGES)}")
    ap.add_argument("--checkpoint", default=None, help="transformer checkpoint for the checks stage (default: the inference run's)")
    ap.add_argument("--list-units", action="store_true", help="print the cluster units as level<TAB>name and exit (scripts/slurm)")
    ap.add_argument("--unit", default=None, help="run one cluster unit by name (see --list-units) and exit")
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
    if args.list_units or args.unit:
        u = units(p)
        if args.unit is not None and args.unit not in u:
            ap.error(f"unknown unit {args.unit!r}; see --list-units")
        print(*(f"{level}\t{name}" for name, (level, _) in u.items()), sep="\n") if args.list_units else print(args.unit, u[args.unit][1]())
        return
    for name in [n for n in STAGES if n in names]:   # pipeline order regardless of how they were listed
        STAGES[name](p)


if __name__ == "__main__":
    main()
