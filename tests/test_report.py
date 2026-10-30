"""The report stage's contract: a removed or duplicated number stops the merge by name, and the generated results.tex and
tables compile in a minimal document (tectonic; skipped when it is not installed)."""
import json
import os
import shutil
import subprocess
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from analysis.numbers import MANIFEST, kind_of  # noqa: E402
from analysis.report import STAGES_WITH_NUMBERS, final_verdicts, merge_numbers, write_results, write_tables  # noqa: E402

VALUE = {"text": "supported", "flag": True, "count": 3, "pvalue": 0.0004, "integer": -2674.0, "two": -1.25, "three": 0.012}


def synthetic(tmp_path):
    """Every manifest key with a value of its kind, spread over the stages' numbers.json files; provenance apart, the final
    verdicts left to the report stage."""
    numbers = {k: VALUE[kind_of(k)] for b, ks in MANIFEST.items() if not b.startswith("final") for k in ks}
    prov = {k: numbers.pop(k) for k in MANIFEST["provenance"]}
    keys = list(numbers)
    for i, stage in enumerate(STAGES_WITH_NUMBERS):
        (tmp_path / stage).mkdir()
        json.dump({k: numbers[k] for k in keys[i::len(STAGES_WITH_NUMBERS)]}, open(tmp_path / stage / "numbers.json", "w"))
    return prov


def test_merge_names_a_removed_and_a_duplicated_number(tmp_path):
    prov = synthetic(tmp_path)
    assert len(merge_numbers(tmp_path, prov)) == sum(map(len, MANIFEST.values()))
    choice = json.load(open(tmp_path / "choice" / "numbers.json"))
    removed = next(iter(choice))
    json.dump({k: v for k, v in choice.items() if k != removed}, open(tmp_path / "choice" / "numbers.json", "w"))
    with pytest.raises(ValueError, match=removed):
        merge_numbers(tmp_path, prov)
    json.dump(choice | {"participants": 1}, open(tmp_path / "choice" / "numbers.json", "w"))
    with pytest.raises(ValueError, match="written by two stages"):
        merge_numbers(tmp_path, prov)


def test_final_verdicts_gate_claims_by_their_checks(tmp_path):
    """Prereg 4.7.3: a failed check makes its claim not evaluable and no other; a missing check counts as failed; the report
    stage derives the final verdicts and refuses them from a stage."""
    base = {"S1_verdict": "supported", "S2_verdict": "contradicted", "S3_verdict": "not supported", "S4_verdict": "supported",
            "model_recovery_S1a_passed": True, "model_recovery_S2a_passed": True, "model_recovery_S4_passed": False, "recovery_passed": True}
    final = lambda n: [final_verdicts(n)[f"S{i}_final_verdict"] for i in (1, 2, 3, 4)]
    assert final(base) == ["supported", "contradicted", "not supported", "supported"], "S4's model recovery is reported, not gating"
    assert final(base | {"model_recovery_S2a_passed": False}) == ["supported", "not evaluable", "not supported", "supported"]
    assert final(base | {"recovery_passed": False}) == ["supported", "contradicted", "not supported", "not evaluable"]
    assert final({k: v for k, v in base.items() if "passed" not in k}) == ["not evaluable", "not evaluable", "not supported", "not evaluable"]
    prov = synthetic(tmp_path)
    assert {k: v for k, v in merge_numbers(tmp_path, prov).items() if k.endswith("_final_verdict")} == {f"S{i}_final_verdict": "supported" for i in (1, 2, 3, 4)}
    choice = json.load(open(tmp_path / "choice" / "numbers.json"))
    json.dump(choice | {"S1_final_verdict": "supported"}, open(tmp_path / "choice" / "numbers.json", "w"))
    with pytest.raises(ValueError, match="report stage derives"):
        merge_numbers(tmp_path, prov)


@pytest.mark.skipif(shutil.which("tectonic") is None, reason="tectonic not installed")
def test_results_and_tables_compile(tmp_path):
    numbers = merge_numbers(tmp_path, synthetic(tmp_path))
    (tmp_path / "descriptive").mkdir(exist_ok=True); (tmp_path / "partner_choice").mkdir(exist_ok=True); (tmp_path / "choice").mkdir(exist_ok=True)
    pd.DataFrame([dict(hypothesis="H1", outcome="payoff", role="confirmatory", estimate=0.09, **{"lower.CL": 0.06, "upper.CL": 0.12, "p.value": 1e-8}, p_holm=2e-8, verdict="supported"),
                  dict(hypothesis="H1", outcome="reward", role="reported", estimate=-0.001, **{"lower.CL": -0.05, "upper.CL": 0.16, "p.value": 0.4}, p_holm=float("nan"), verdict=None)]
                 ).to_csv(tmp_path / "descriptive" / "contrasts.csv", index=False)
    pd.DataFrame([dict(role="confirmatory", term="(Intercept)", **{"p.value": 0.004}, prob_estimate=0.71, prob_lower=0.57, prob_upper=0.81, verdict="supported")]).to_csv(tmp_path / "partner_choice" / "model.csv", index=False)
    pd.DataFrame(dict(model=["HOS", "recent"], elpd=[-2673.9, -2762.2], n_bad_k=[0, 0], delta=[0.0, -88.3], se=[0.0, 18.4])).to_csv(tmp_path / "choice" / "ladder.csv", index=False)
    paper = tmp_path / "paper"
    write_results(numbers, paper), write_tables(tmp_path, paper)
    (paper / "test.tex").write_text(r"\documentclass{article}\usepackage{booktabs}\input{results.tex}\begin{document}"
                                    r"\resSOneACI, \resHOnePayoffP, \resElpdHOS, \resRecoveryPassed, \resCommit. \input{tables/hypotheses.tex}\input{tables/ladder.tex}\end{document}")
    r = subprocess.run(["tectonic", "test.tex"], cwd=paper, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    assert (paper / "test.pdf").stat().st_size > 1000
