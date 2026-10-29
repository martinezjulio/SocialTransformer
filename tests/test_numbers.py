"""Macro names and number formats of ``analysis.numbers``."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from analysis.numbers import MANIFEST, check_keys, check_manifest, fmt, kind_of, macro_name, write_numbers  # noqa: E402


def test_macro_names_are_letters_only_with_digits_spelled():
    assert macro_name("S1a_estimate") == "resSOneAEstimate"
    assert macro_name("elpd_HOS+eps") == "resElpdHOSEps"
    assert macro_name("ppc_HOS split_half_zigzag_inside") == "resPpcHOSSplitHalfZigzagInside"
    assert macro_name("elpd_Switch-by-round") == "resElpdSwitchByRound" and macro_name("unseen_expert_round12_x") == "resUnseenExpertRoundOneTwoX"
    assert all(c.isalpha() for c in macro_name("H2_payoff_p_holm"))


def test_kinds_and_formats():
    assert [kind_of(k) for k in ("H1_payoff_p", "S4_lo", "S1a_estimate", "elpd_HOS", "split_round_1_se", "ppc_HOS_zigzag_inside", "recovery_n",
                                 "unseen_expert_round1_confidence", "H1_payoff_estimate")] == ["pvalue", "two", "integer", "integer", "integer", "flag", "count", "integer", "two"]
    assert fmt(0.0004, "pvalue") == "< .001" and fmt(0.051, "pvalue") == "0.051"
    assert fmt(88.29, "integer") == "88" and fmt(-2673.92, "integer") == "-2674" and fmt(2.4214, "two") == "2.42" and fmt(True, "flag") == "yes"
    assert fmt(float("nan"), "integer") == "—" and fmt(3, "count") == "3"


def test_writer_rejects_collisions_and_bad_values(tmp_path):
    with pytest.raises(ValueError):
        check_keys({"S1a_lo": 1.0, "S1a-lo": 2.0})
    with pytest.raises(TypeError):
        check_keys({"x": [1, 2]})
    p = write_numbers(tmp_path, {"a_n": np.int64(3), "b_estimate": np.float32(1.5), "c_inside": np.bool_(True)})
    assert json.load(open(p)) == {"a_n": 3, "b_estimate": 1.5, "c_inside": True}


def test_manifest_keys_give_distinct_macros_and_the_check_names_missing_and_extra():
    keys = [k for ks in MANIFEST.values() for k in ks]
    assert len(keys) == len(set(keys))
    check_keys({k: 1.0 for k in keys})                      # distinct macro names across every block
    full = {k: 1.0 for k in keys}
    check_manifest(full)
    broken = {k: v for k, v in full.items() if k != "S1a_lo"} | {"stray": 2.0}
    with pytest.raises(ValueError) as e:
        check_manifest(broken)
    assert "S1a_lo" in str(e.value) and "stray" in str(e.value) and "policy models" in str(e.value)
    assert fmt("supported", kind_of("H1_payoff_verdict")) == "supported" and kind_of("excluded_returned") == "count" and kind_of("participants") == "count"
