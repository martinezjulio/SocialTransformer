"""The choice stage's verdicts follow the registration's four outcomes (prereg 4.3): per part on its interval, per claim from its parts."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from run_analysis import claim_verdict, verdict  # noqa: E402

NAN = float("nan")


def test_part_verdicts():
    assert verdict(10.0, 2.0, 18.0) == "supported"           # ΔELPD > 2 SE
    assert verdict(-10.0, -18.0, -2.0) == "contradicted"     # the rival ahead by > 2 SE
    assert verdict(3.0, -1.0, 7.0) == "not supported" and verdict(-3.0, -7.0, 1.0) == "not supported"
    assert verdict(0.0, 0.0, 0.0) == "not supported"         # the boundaries do not support or contradict
    assert verdict(NAN, NAN, NAN) == "not evaluable"         # a fit failed the gate


def test_claim_verdicts():
    assert claim_verdict(["supported", "supported", "supported"]) == "supported"
    assert claim_verdict(["supported", "not supported"]) == "not supported"
    assert claim_verdict(["supported", "not evaluable"]) == "not evaluable"
    assert claim_verdict(["not evaluable", "contradicted"]) == "contradicted"   # a contradiction is never masked
    assert claim_verdict(["not supported", "contradicted", "supported"]) == "contradicted"
