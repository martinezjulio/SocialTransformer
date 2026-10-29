# %% numbers
"""The numbers the paper cites: each stage writes a ``numbers.json`` through ``write_numbers``; the report stage turns
every key into a LaTeX macro (``macro_name``) formatted by its quantity type (``kind_of``, ``fmt``; meta-plan 9.2).

Keys are free text (``S1a_estimate``, ``elpd_HOS+eps``, ``ppc_HOS split_half_zigzag_inside``); the macro name keeps
letters only: words split at every non-letter, digits spelled out, each word capitalised, prefixed ``res``
(``resSOneAEstimate``). Two keys may not share a macro name. Values are plain numbers or booleans.
"""

import json
import math
import re
from pathlib import Path

import numpy as np

DIGITS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
# Quantity type by key pattern, first match wins: p-values; logit-scale posteriors (S4 is δ_N); ELPD and SE as integers;
# yes/no flags; counts; readout confidence (a Beta concentration); everything else (payoff, correlation, share) two decimals.
KINDS = [(r"(_p|_p_holm)$", "pvalue"), (r"^S4_", "two"), (r"^S\d|^own_on_top|^switch_round_term|^elpd_|_delta$|_se$|_elpd$", "integer"),
         (r"_inside$|_passed$", "flag"), (r"_n$|_correct$|_sign$|_covered$|^n_|_count$", "count"), (r"confidence$", "integer")]


def kind_of(key: str) -> str:
    return next((k for pat, k in KINDS if re.search(pat, key)), "two")


def fmt(value, kind: str) -> str:
    """One value in its type's format; NaN prints as an em dash (a not-evaluable quantity)."""
    if kind == "flag":
        return "yes" if value else "no"
    if isinstance(value, float) and math.isnan(value):
        return "—"
    if kind == "pvalue":
        return "< .001" if value < 0.001 else f"{value:.3f}"
    return f"{value:.0f}" if kind in ("integer", "count") else f"{value:.2f}"


def macro_name(key: str) -> str:
    words = re.findall(r"[A-Za-z]+|[0-9]", key)
    return "res" + "".join(DIGITS[int(w)] if w.isdigit() else w[0].upper() + w[1:] for w in words)


def check_keys(numbers: dict) -> None:
    """Every key must give a distinct macro name and every value be a number or a boolean."""
    names = {}
    for k, v in numbers.items():
        if not isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_)):
            raise TypeError(f"numbers[{k!r}] = {v!r} is not a plain number or boolean")
        if (n := macro_name(k)) in names:
            raise ValueError(f"keys {names[n]!r} and {k!r} both give the macro {n}")
        names[n] = k


def write_numbers(stage_dir: Path, numbers: dict) -> Path:
    """Validate and write ``<stage_dir>/numbers.json`` (keys in insertion order); returns its path."""
    check_keys(numbers)
    path = Path(stage_dir) / "numbers.json"
    json.dump({k: (v.item() if isinstance(v, np.generic) else v) for k, v in numbers.items()}, open(path, "w"), indent=1)
    return path
