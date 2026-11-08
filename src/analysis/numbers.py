# %% numbers
"""The numbers the paper cites: each stage writes a ``numbers.json`` through ``write_numbers``; the report stage turns
every key into a LaTeX macro (``macro_name``) formatted by its quantity type (``kind_of``, ``fmt``; meta-plan 9.2), after
``check_manifest`` has found every key the paper cites (``MANIFEST``, one block per figure or text section) and nothing else.

Keys are free text (``S1a_estimate``, ``elpd_ROS+eps``, ``ppc_ROS split_half_zigzag_inside``); the macro name keeps
letters only: words split at every non-letter, digits spelled out, each word capitalised, prefixed ``res``
(``resSOneAEstimate``). Two keys may not share a macro name. Values are plain numbers, booleans, or short text (verdicts, provenance).
"""

import json
import math
import re
from pathlib import Path

import numpy as np

from analysis.policies import REGISTRY

DIGITS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
# Quantity type by key pattern, first match wins: p-values; logit-scale posteriors (S4 is δ_N); ELPD and SE as integers;
# yes/no flags; counts; readout confidence (a Beta concentration); everything else (payoff, correlation, share) two decimals.
KINDS = [(r"^(commit|checkpoint_sha256|run_date|models_not_evaluable)$|_verdict$|_spec$|_rival$", "text"), (r"(_p|_p_holm)$", "pvalue"), (r"^S4_", "two"), (r"_eps_(estimate|lo|hi)$", "three"),
         (r"^S\d|^own_on_top|^switch_round_term|^social_vs_own|^elpd_|_delta$|_se$|_elpd$|^age_(min|max)$", "integer"), (r"_inside$|_passed$", "flag"),
         (r"_n$|_correct$|_sign$|_covered$|_count$|_kept$|^excluded_|^(participants|sessions|trials)$", "count"), (r"confidence$", "integer")]


def kind_of(key: str) -> str:
    return next((k for pat, k in KINDS if re.search(pat, key)), "two")


def fmt(value, kind: str) -> str:
    """One value in its type's format; NaN prints as an em dash (a not-evaluable quantity)."""
    if kind == "text":
        return str(value)
    if kind == "flag":
        return "yes" if value else "no"
    if isinstance(value, float) and math.isnan(value):
        return "—"
    if kind == "pvalue":
        return "< .001" if value < 0.001 else f"{value:.3f}"
    return re.sub(r"^-(0(\.0+)?)$", r"\1", f"{value:.0f}" if kind in ("integer", "count") else f"{value:.3f}" if kind == "three" else f"{value:.2f}")   # no negative zero


def macro_name(key: str) -> str:
    words = re.findall(r"[A-Za-z]+|[0-9]", key)
    return "res" + "".join(DIGITS[int(w)] if w.isdigit() else w[0].upper() + w[1:] for w in words)


def check_keys(numbers: dict) -> None:
    """Every key must give a distinct macro name and every value be a number or a boolean."""
    names = {}
    for k, v in numbers.items():
        if not isinstance(v, (int, float, bool, str, np.integer, np.floating, np.bool_)):
            raise TypeError(f"numbers[{k!r}] = {v!r} is not a plain number, boolean or text")
        if (n := macro_name(k)) in names:
            raise ValueError(f"keys {names[n]!r} and {k!r} both give the macro {n}")
        names[n] = k


# The numbers the paper cites, one block per figure or text section; the report stage fails when any is missing or unlisted.
OUTCOMES = {"H1": ("payoff", "reward", "entropy"), "H2": ("payoff", "reward", "entropy"), "H3": ("entropy",), "H5": ("payoff", "reward")}
EXPLORATORY = {"novice_vs_solo_improvement": OUTCOMES["H1"], "expert_round_change": OUTCOMES["H1"], "solo_round_change": OUTCOMES["H1"],
               "H5_round2": OUTCOMES["H5"], "social_vs_solo4": OUTCOMES["H5"], "novice_vs_solo4": OUTCOMES["H5"],
               "H2_expert_first": OUTCOMES["H1"], "H2_novice_first": OUTCOMES["H1"], "solo_play_change": OUTCOMES["H1"]}   # E5a, E5b, E13
RANDOM_SLOPES = [f"random_slopes_{o}_{f}" for o in OUTCOMES["H1"] for f in ("round_sd", "play_sd", "spec")]   # E6: slope SDs and the specification reached
EXCLUSION_RULES = ("returned", "colour_vision", "speed_floor", "span_ceiling", "no_gameplay", "incomplete_rounds", "short_play", "reload",
                   "restarted", "never_moved", "not_three_sessions")
DEMOGRAPHICS = ("age_mean", "age_sd", "age_min", "age_max", "sex_female_count", "sex_male_count", "sex_other_count")   # demographics_summary.csv
CLAIMS = ("S1a", "S1b", "S2a", "S2b", "S2c", "S3a", "S3b", "S4", "S1", "S2", "S3", "switch_round_term", "own_on_top", "social_vs_own")
POSTERIORS = ("delta_E", "ktrend_delta_kappa", "ktrend_delta_N", "eps_ROS_eps", "eps_ROS_kappa", "eps_switch_eps", "eps_switch_delta_N")   # E8, E10: mean and 95% ETI
PPC_TARGETS = ("after_novice_rise", "entropy_fall", "zigzag", "stay_win", "stay_loss")
interval = lambda stem, p=True: [f"{stem}_{f}" for f in ("estimate", "lo", "hi") + (("p",) if p else ())]
MANIFEST = {
    "provenance": ["commit", "checkpoint_sha256", "run_date", "participants", "sessions", "trials"] + [f"excluded_{r}" for r in EXCLUSION_RULES] + ["participants_kept", "sessions_kept"] + list(DEMOGRAPHICS),
    "descriptive (Fig 2, H1-H5)": [k for h, outs in OUTCOMES.items() for o in outs for k in interval(f"{h}_{o}") + ([f"{h}_{o}_p_holm", f"{h}_{o}_verdict"] if o != "reward" else [])]
                                 + [k for h, outs in EXPLORATORY.items() for o in outs for k in interval(f"{h}_{o}")] + RANDOM_SLOPES,
    "partner choice (Fig 2c, H4)": interval("H4_choice") + ["H4_choice_verdict", "H4_choice_count", "H4_choice_n"] + interval("recency"),
    "policy models (Fig 3, S1-S4)": [k for c in CLAIMS for k in interval(c, p=False)] + [f"{c}_verdict" for c in CLAIMS if c[0] == "S"] + [f"elpd_{m}" for m in REGISTRY] + ["models_not_evaluable"]
                                   + [k for s in POSTERIORS for k in interval(s, p=False)],
    "unseen expert (Fig 4)": [f"unseen_expert_round{r}_{s}" for r in (1, 2) for s in ("confidence", "rank_corr", "favourite_payoff")] + ["exploit_unseen_round1", "exploit_unseen_round2", "exploit_chance"]
                            + [f"split_{c}_{f}" for c in ("after_novice_expert_unseen_round_1", "after_novice_expert_unseen_round_2", "after_novice_expert_seen_round_1",
                                                           "after_novice_expert_seen_round_2", "round_1", "round_2") for f in ("delta", "se")],
    "checks (Fig S, PPC and recovery)": ["recovery_covered", "recovery_sign", "recovery_n", "recovery_bias", "model_recovery_correct", "model_recovery_n"]
                                       + [f"model_recovery_{c}_{k}" for c in ("S1", "S2", "S4") for k in ("own_count", "rival_count", "side_n")]
                                       + [f"model_recovery_{c}_rival" for c in ("S1", "S2")]
                                       + [f"ppc_{m}{h}_{t}_{f}" for m in ("ROS", "Switch-by-round") for h in ("", " split_half") for t in PPC_TARGETS for f in ("human", "lo", "hi", "inside")],
}


def check_manifest(numbers: dict) -> None:
    """Every manifest key present and nothing else; the error names each missing key by block and each extra key."""
    missing = {b: [k for k in ks if k not in numbers] for b, ks in MANIFEST.items()}
    extra = sorted(set(numbers) - {k for ks in MANIFEST.values() for k in ks})
    if any(missing.values()) or extra:
        lines = [f"missing in {b!r}: {', '.join(ks)}" for b, ks in missing.items() if ks] + ([f"not in the manifest: {', '.join(extra)}"] if extra else [])
        raise ValueError("numbers do not match the manifest\n" + "\n".join(lines))


def write_numbers(stage_dir: Path, numbers: dict) -> Path:
    """Validate and write ``<stage_dir>/numbers.json`` (keys in insertion order); returns its path."""
    check_keys(numbers)
    path = Path(stage_dir) / "numbers.json"
    json.dump({k: (v.item() if isinstance(v, np.generic) else v) for k, v in numbers.items()}, open(path, "w"), indent=1)
    return path
