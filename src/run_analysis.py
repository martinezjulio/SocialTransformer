"""Driver of the analysis pipeline: run registered stages for one experiment version.

    python src/run_analysis.py --version pilot_E [--stages games,inference-check]
    python src/run_analysis.py --version pilot_E --list-units [--missing] | --unit fit:ROS    # cluster units (scripts/slurm)

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
N_TABLE_BLOCKS = int(os.environ.get("N_TABLE_BLOCKS", 1))
SKIP_MODELS = frozenset(filter(None, os.environ.get("SKIP_MODELS", "").split(",")))   # registry models not fitted in this run (an exploratory fit that did not converge within the compute budget); their numbers read "not evaluable"   # row blocks per human table in the cluster units (scripts/slurm)


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
    cached, fits cached by table hash, PSIS-LOO with the Pareto-k fallback on the S1/S2 models (``loso_loo``), ladder vs ROS,
    the registration's four-outcome verdicts for S1a–S3b (2 SE each), the claims S1–S3 (from their parts) and S4 (δ_N 95% ETI), numbers.json."""
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    import pandas as pd
    from analysis.numbers import write_numbers
    from analysis.compare import K_BAD, by_cell, ladder, loo, paired
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
    paths = {n: (tab(m.source, m), tab("novice", m) if m.mixture != "none" else None) for n, m in REGISTRY.items() if n not in SKIP_MODELS}
    fits = {}
    for n, (a, b) in paths.items():   # fit cache keyed by the model's own table hashes; wait only for this model's tables
        pending[a].result(), b and pending[b].result()
        fits.update(run_many([n], p.eval_dir, p.version, paths, a.stem.split("_")[-1] + (f"-{b.stem.split('_')[-1]}" if b else ""), out / "fits"))
    builder.shutdown()
    loos = {n: loo(f) for n, f in fits.items() if f.passed}
    bad = {n: l.n_bad for n, l in loos.items()}   # the ladder reports the pre-fallback count and the sessions the fallback refits
    refit = {n: len(set(T.session[l.pareto_k > K_BAD])) if n in FALLBACK_MODELS else 0 for n, l in loos.items()}
    loos = {n: loso_loo(p, T, n, l) if n in FALLBACK_MODELS else l for n, l in loos.items()}
    lad = ladder(loos, "ROS", T.part).assign(n_bad_k=pd.Series(bad), sessions_refit=pd.Series(refit)).join(pd.DataFrame({n: f.diagnostics for n, f in fits.items()}).T)
    lad.to_csv(out / "ladder.csv")
    rounds = {"round 1": T.round_pm < 0, "round 2": T.round_pm > 0}   # the ladder per round (Fig 3b, 3d): ΔELPD vs ROS, paired SE, share
    pd.concat({n: by_cell(l, loos["ROS"], T, rounds) for n, l in loos.items()}, names=["model", "round"]).to_csv(out / "ladder_by_round.csv")
    pd.DataFrame({n: paired(l, loos["ROS"], T.part)["per_participant"] for n, l in loos.items()}).rename_axis("participant").to_csv(out / "elpd_per_participant.csv")   # Fig S: dots behind every paired SE
    def post(name, site, f=lambda x: x):   # posterior mean and 95% ETI of a site (or a function of it); NaN when the fit failed the gate
        x = f(fits[name].posterior[site].reshape(-1)) if name in loos else np.array([np.nan])
        return dict(estimate=x.mean(), lo=np.percentile(x, 2.5), hi=np.percentile(x, 97.5), rule="posterior")
    rows = {"S4": post("Switch-by-round", "delta_N") | dict(rule="95% ETI > 0")}
    for name, a, b in [("S1a", "ROS", "recent"), ("S1b", "ROS", "novice"), ("S2a", "ROS", "naive-recent"), ("S2b", "ROS", "naive-expert"),
                       ("S2c", "ROS", "naive"), ("S3a", "RS", "RO"), ("S3b", "ROS", "RO"),
                       ("switch_round_term", "Switch-by-round", "Switch"), ("own_on_top", "ROS", "RS"), ("social_vs_own", "S", "O")]:
        q = paired(loos[a], loos[b], T.part) if a in loos and b in loos else dict(delta=np.nan, se=np.nan)
        rows[name] = dict(estimate=q["delta"], lo=q["delta"] - 2 * q["se"], hi=q["delta"] + 2 * q["se"], rule="ΔELPD > 2 SE" if name[0] == "S" else "reported")
    rows |= {"delta_E": post("Switch-by-round", "delta_E"), "ktrend_delta_kappa": post("Switch-by-round+ktrend", "delta_kappa"),   # E8, E10 (numbers.POSTERIORS)
             "ktrend_delta_N": post("Switch-by-round+ktrend", "delta_N"), "eps_ROS_eps": post("ROS+eps", "eps"), "eps_ROS_kappa": post("ROS+eps", "mu_log_kappa", np.exp),
             "eps_switch_eps": post("Switch-by-round+eps", "eps"), "eps_switch_delta_N": post("Switch-by-round+eps", "delta_N")}
    for r in rows.values():
        r["verdict"] = "" if r["rule"] in ("reported", "posterior") else verdict(r["estimate"], r["lo"], r["hi"])
    for claim in ("S1", "S2", "S3"):   # per-part verdicts stay reported beside the claim's
        rows[claim] = dict(estimate=np.nan, lo=np.nan, hi=np.nan, rule="all parts > 2 SE", verdict=claim_verdict([r["verdict"] for k, r in rows.items() if k.startswith(claim) and len(k) == 3]))
    pd.DataFrame(rows).T.to_csv(out / "verdicts.csv")
    write_numbers(out, {f"{k}_{f}": float(v) for k, r in rows.items() for f, v in r.items() if f != "verdict" and f != "rule"}
                  | {f"{k}_verdict": r["verdict"] for k, r in rows.items() if r["verdict"]} | {f"elpd_{n}": (loos[n].elpd if n in loos else float("nan")) for n in REGISTRY}
                  | {"models_not_evaluable": ", ".join(n for n in REGISTRY if n not in loos) or "none"})   # failed the gate or skipped (SKIP_MODELS): their numbers print as not evaluable
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


FALLBACK_MODELS = ("ROS", "recent", "novice", "naive-recent", "naive-expert", "naive")   # the registered S1/S2 comparisons (prereg 4.7.3)


def loso_loo(p: Paths, T, name: str, l):
    """The Pareto-k fallback of one model's ``Loo``: exact leave-one-session-out refits for the sessions holding k > 0.7
    trials (``compare.loso``), cached under ``choice/loso/`` by the fit's key; ``l`` itself when no trial is bad."""
    import pickle
    from analysis.compare import loso
    from analysis.data import atomic_write
    from analysis.fitting import cache_path
    from analysis.policies import REGISTRY
    if l.n_bad == 0:
        return l
    path = cache_path(name, "-".join(t.stem.split("_")[-1] for t in human_tables(p, T, name)), p.analysis_dir / "choice" / "loso")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tabs = [str(x) for x in human_tables(p, T, name)]   # paths: each session's refit runs in a spawned process that memory-maps them
        atomic_write(path, lambda t: pickle.dump(loso(l, REGISTRY[name], T, tabs[0], tabs[1] if len(tabs) > 1 else None), open(t, "wb")))
    return pickle.load(open(path, "rb"))


def loso_unit(p: Paths, T, name: str) -> dict:
    """The level-3 fallback unit: the model's LOO, its fallback refits if any session has k > 0.7, and a `.nobad` marker beside
    the would-be cache when none has (so `--list-units --missing` can tell without recomputing the LOO)."""
    from analysis.compare import loo
    from analysis.fitting import cache_path
    l = loo(choice_fit(p, T, name))
    path = cache_path(name, "-".join(t.stem.split("_")[-1] for t in human_tables(p, T, name)), p.analysis_dir / "choice" / "loso")
    if l.n_bad == 0:
        path.parent.mkdir(parents=True, exist_ok=True); path.with_suffix(".nobad").touch()
    return dict(n_bad=l.n_bad, elpd=l.elpd, elpd_after=loso_loo(p, T, name, l).elpd)


def recovery_pairs(p: Paths, T) -> dict:
    """The model-recovery pairs (``checks.comparisons``): from the choice stage's ladder, or, in a cluster unit that runs before
    that stage, from the same cached fits and fallback refits (every candidate is in FALLBACK_MODELS)."""
    import pandas as pd
    from analysis.checks import RIVALS, comparisons
    from analysis.compare import loo
    path = p.analysis_dir / "choice" / "ladder.csv"
    if path.exists():
        return comparisons(pd.read_csv(path, index_col="model"))
    elpd = {n: loso_loo(p, T, n, loo(choice_fit(p, T, n))).elpd for n in ("ROS", *sum(RIVALS.values(), ()))}
    return comparisons(pd.DataFrame({"delta": {n: e - elpd["ROS"] for n, e in elpd.items()}}))


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
    """Simulation checks of the fitted models: δ_N recovery, model recovery for S1 and S2 (ROS against its nearest rival in the
    ladder, ``recovery_pairs``) and S4, free-running PPC. The transformer
    is the one the inference CSVs came from (eval metadata) unless ``checkpoint`` overrides it. Every unit is cached
    under checks/, so a rerun only fills what is missing. Outputs: recovery.csv, model_recovery.csv, confusion.csv,
    ppc.csv, numbers.json."""
    import pandas as pd
    from analysis.checks import PPC_MODELS, confusion, model_recovery, model_recovery_summary, ppc, recover_delta, recovery_summary, split_half
    from analysis.numbers import write_numbers
    from analysis.policies import REGISTRY
    T, out, posterior, model, norm, pools, frames = checks_setup(p)
    rec = recover_delta(posterior("Switch-by-round"), pools, model, norm, out / "recovery")
    rec.to_csv(out / "recovery.csv", index=False)
    comps = recovery_pairs(p, T)
    mr = model_recovery({n: posterior(n) for pair in comps.values() for n in pair}, pools, model, norm, out / "model_recovery", comps)
    mr.to_csv(out / "model_recovery.csv", index=False)
    confusion(mr).to_csv(out / "confusion.csv")
    bands = pd.concat({n: ppc(REGISTRY[n], posterior(n), T, frames, model, norm, out / "ppc") for n in PPC_MODELS}, names=["model", "target"])
    halves = pd.concat({f"{n} split_half": split_half(REGISTRY[n], T, table_arrays(p, T, n), frames, model, norm, out / "split_half") for n in PPC_MODELS}, names=["model", "target"])
    bands = pd.concat([bands, halves])
    bands.to_csv(out / "ppc.csv")
    mrs = model_recovery_summary(mr, comps)   # per-comparison win counts, reported beside the claims (no gate, 4 Nov)
    pd.DataFrame(mrs).T.to_csv(out / "model_recovery_summary.csv")
    numbers = {**{f"recovery_{k}": v for k, v in recovery_summary(rec).items()}, "model_recovery_correct": int(mr["correct"].sum()), "model_recovery_n": len(mr),
               **{f"model_recovery_{c}_{k}": v[f] for c, v in mrs.items() for k, f in (("own_count", "a_wins_on_a"), ("rival_count", "a_wins_on_b"), ("side_n", "n_a"))},
               **{f"model_recovery_{c}_rival": comps[c][1] for c in ("S1", "S2")},
               **{f"ppc_{m}_{t}_{f}": (bool(r[f]) if f == "inside" else float(r[f])) for (m, t), r in bands.iterrows() for f in ("human", "lo", "hi", "inside")}}
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
    loos = {n: loso_loo(p, T, n, loo(choice_fit(p, T, n))) for n in ("ROS", "recent", "naive-recent")}   # the choice stage's fallback-corrected LOO (cache)
    tabs["elpd_split"], tabs["copying_elpd"] = elpd_by_cell(loos["ROS"], loos["recent"], T), elpd_by_cell(loos["ROS"], loos["naive-recent"], T)
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
    """Every cluster unit (scripts/slurm) as name -> (level, thunk, done), in dependency order: 0 table blocks (CPU), 1 table
    assembly, 2 human fits, 3 recovery datasets, split-half fits and PPC draws (GPU), 4 split-half draws and model-recovery
    datasets, with the loso refits of the S1/S2 models at level 3 (the model-recovery pair is read from them). Each fills one
    cache piece (one writer per file); the stages then run on the full caches. ``done()`` says whether the piece is already
    cached (``--list-units --missing`` skips it); a test whose own inputs are not cached yet counts as not done."""
    import pickle

    import numpy as np
    from analysis.checks import N_DATASETS, N_MODEL_RECOVERY, N_PPC_DRAWS, PPC_MODELS, RECOVERY_CLAIMS, model_recovery, ppc, recover_delta, split_half
    from analysis.compare import loo
    from analysis.data import load_sessions, trial_data
    from analysis.fitting import cache_path, run_many
    from analysis.policies import REGISTRY, belief, grids
    from analysis.thompson import block_path, table, table_path
    T, u = trial_data(load_sessions(p.eval_dir, p.version)), {}
    tab = lambda s, m, **kw: table(f"{s}_{int(m.kappa)}_{int(m.own)}", *belief(T, s), T.own_seg, T.own_prev, p.analysis_dir / "tables", *grids(m), n_blocks=N_TABLE_BLOCKS, **kw).shape
    tpath = lambda s, m: table_path(f"{s}_{int(m.kappa)}_{int(m.own)}", *belief(T, s), T.own_seg, T.own_prev, p.analysis_dir / "tables", *grids(m))
    check = lambda call: call(*checks_setup(p))   # a checks unit: the stage's setup, then one call restricted to the unit
    fits, checks = p.analysis_dir / "choice" / "fits", p.analysis_dir / "checks"
    fit_key = lambda n: "-".join(x.stem.split("_")[-1] for x in human_tables(p, T, n))
    pairs, sizes = {}, {}   # recovery_pairs and the fits' posterior sizes, computed once on first use

    def quiet(f):
        try:
            return bool(f())
        except Exception:
            return False

    def draw_path(n, i, cache_dir, key, sub, tag):   # the json of draw i of the evenly spaced grid over a cached fit's posterior
        fp = cache_path(n, key, cache_dir)
        if fp not in sizes:
            v = next(iter(pickle.load(open(fp, "rb")).posterior.values())); sizes[fp] = v.shape[0] * v.shape[1]
        size = sizes[fp]
        return checks / sub / f"{REGISTRY[n].name.replace(':', '-')}{tag}_draw{int(np.linspace(0, size - 1, N_PPC_DRAWS).astype(int)[i]):05d}.json"

    def mr_done(c, i, d):   # both fits of the pair exist for dataset d of side i; the pair is read from the file names (no LOO)
        from collections import Counter
        groups = Counter(f.name.split(f"_{c}_", 1)[1] for f in (checks / "model_recovery" / "fits").glob(f"*_{c}_*_vs_*_{d:02d}.pkl"))
        for suffix, n in groups.items():   # suffix = "{a}_vs_{b}_{side}_{dd}.pkl"
            a, rest = suffix.split("_vs_", 1); b_side = rest.rsplit("_", 1)[0]   # "{b}_{side}": the side is a (i = 0) when it ends with a
            if n >= 2 and ((b_side.endswith("_" + a) and i == 0) or (not b_side.endswith("_" + a) and i == 1)):
                return True
        return False

    for n, m in REGISTRY.items():
        if n in SKIP_MODELS:
            continue
        for s in (m.source, "novice") if m.mixture != "none" else (m.source,):
            tp = tpath(s, m)
            for b in range(N_TABLE_BLOCKS):
                u[f"table:{s}_{int(m.kappa)}_{int(m.own)}:{b}/{N_TABLE_BLOCKS}"] = (0, lambda s=s, m=m, b=b: tab(s, m, block=b),
                                                                                  lambda tp=tp, b=b: tp.exists() or block_path(tp, b, N_TABLE_BLOCKS).exists())
            u[f"assemble:{s}_{int(m.kappa)}_{int(m.own)}"] = (1, lambda s=s, m=m: tab(s, m), lambda tp=tp: tp.exists())
        t = human_tables(p, T, n)
        u[f"fit:{n}"] = (2, lambda n=n, t=t: run_many([n], p.eval_dir, p.version, {n: (t[0], t[1] if len(t) > 1 else None)}, "-".join(x.stem.split("_")[-1] for x in t), fits)[n].diagnostics,
                         lambda n=n: cache_path(n, fit_key(n), fits).exists())
    for n in FALLBACK_MODELS:   # the Pareto-k fallback refits (loso_loo) after the fits, on a GPU; the choice stage then reads the cache
        u[f"loso:{n}"] = (3, lambda n=n: loso_unit(p, T, n),
                          lambda n=n: cache_path(n, fit_key(n), p.analysis_dir / "choice" / "loso").exists() or cache_path(n, fit_key(n), p.analysis_dir / "choice" / "loso").with_suffix(".nobad").exists())
    for d in range(N_DATASETS):
        u[f"recovery:{d}"] = (3, lambda d=d: check(lambda T, out, post, model, norm, pools, frames: recover_delta(post("Switch-by-round"), pools, model, norm, out / "recovery", datasets=[d])),
                              lambda d=d: cache_path("Switch-by-round", f"dataset_{d:02d}", checks / "recovery" / "fits").exists())
    for c in RECOVERY_CLAIMS:   # side i of the pair, resolved at run time by recovery_pairs (after the level-3 fallback refits)
        for i in range(2):
            for d in range(N_MODEL_RECOVERY):
                u[f"model_recovery:{c}:{i}:{d}"] = (4, lambda c=c, i=i, d=d: check(lambda T, out, post, model, norm, pools, frames: (lambda pair: model_recovery(
                    {n: post(n) for n in pair}, pools, model, norm, out / "model_recovery", {c: pair}, units=[(c, pair[i], d)]))(recovery_pairs(p, T)[c])),
                    lambda c=c, i=i, d=d: mr_done(c, i, d))
    for n in PPC_MODELS:
        u[f"split_fit:{n}"] = (3, lambda n=n: check(lambda T, out, post, model, norm, pools, frames: split_half(REGISTRY[n], T, table_arrays(p, T, n), frames, model, norm, out / "split_half", draws=[])),
                               lambda n=n: cache_path(REGISTRY[n].name, "split0", checks / "split_half" / "fits").exists())
        for i in range(N_PPC_DRAWS):
            u[f"ppc:{n}:{i}"] = (3, lambda n=n, i=i: check(lambda T, out, post, model, norm, pools, frames: ppc(REGISTRY[n], post(n), T, frames, model, norm, out / "ppc", draws=[i])),
                                 lambda n=n, i=i: quiet(lambda: draw_path(n, i, fits, fit_key(n), "ppc", "").exists()))
            u[f"split_half:{n}:{i}"] = (4, lambda n=n, i=i: check(lambda T, out, post, model, norm, pools, frames: split_half(REGISTRY[n], T, table_arrays(p, T, n), frames, model, norm, out / "split_half", draws=[i])),
                                        lambda n=n, i=i: quiet(lambda: draw_path(REGISTRY[n].name, i, checks / "split_half" / "fits", "split0", "split_half", "_split0").exists()))
    return u


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", required=True, help="e.g. pilot_E")
    ap.add_argument("--stages", default=",".join(STAGES), help=f"comma-separated subset of: {', '.join(STAGES)}")
    ap.add_argument("--checkpoint", default=None, help="transformer checkpoint for the checks stage (default: the inference run's)")
    ap.add_argument("--list-units", action="store_true", help="print the cluster units as level<TAB>name and exit (scripts/slurm)")
    ap.add_argument("--unit", default=None, help="run one cluster unit by name (see --list-units) and exit")
    ap.add_argument("--missing", action="store_true", help="with --list-units: only the units whose cache piece does not exist yet")
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
        if args.list_units:
            print(*(f"{level}\t{name}" for name, (level, _, done) in u.items() if not (args.missing and done())), sep="\n")
        else:
            print(args.unit, u[args.unit][1]())
        return
    for name in [n for n in STAGES if n in names]:   # pipeline order regardless of how they were listed
        STAGES[name](p)


if __name__ == "__main__":
    main()
