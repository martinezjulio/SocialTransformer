"""The legacy notebook code and the new analysis pipeline never import each other.

Checked on the import statements (parsed, so comments and docstrings do not count):
`src/` never imports `legacy_analysis` or `notebooks`; `notebooks/legacy_analysis/` and
`notebooks/analysis.py` never import the new `analysis` package under `src/`.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def imported_roots(path):
    """Top-level names of every import in a Python (or `# %%` cell) file; IPython magics skipped."""
    source = "\n".join(l for l in path.read_text().splitlines() if not l.lstrip().startswith("%"))
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module.split(".")[0]


def offenders(files, forbidden):
    return sorted(f"{p.relative_to(ROOT)} imports {n}" for p in files for n in set(imported_roots(p)) if n in forbidden)


def test_src_never_imports_legacy():
    assert offenders(ROOT.glob("src/**/*.py"), {"legacy_analysis", "notebooks"}) == []


def test_legacy_never_imports_new_pipeline():
    legacy = [*ROOT.glob("notebooks/legacy_analysis/*.py"), ROOT / "notebooks" / "analysis.py"]
    assert offenders(legacy, {"analysis"}) == []
