# %% fits
"""The choice stage's cached fits, for the figures: ``load_fit(analysis_dir, name)`` finds the pickle of a registry model
by its sanitised name and hash key (``<name>_<hash>[-<hash>].pkl``), so ``ROS`` does not match ``ROS+persev``."""

import pickle
import re
from pathlib import Path


def load_fit(analysis_dir: Path, name: str):
    stem = re.sub(r"[^\w.-]", "_", name)
    files = [q for q in (Path(analysis_dir) / "choice" / "fits").glob(f"{stem}_*.pkl") if re.fullmatch(rf"{re.escape(stem)}_[0-9a-f]+(-[0-9a-f]+)?\.pkl", q.name)]
    if not files:
        raise FileNotFoundError(f"no cached fit of {name} under {analysis_dir}/choice/fits")
    return pickle.load(open(max(files, key=lambda q: q.stat().st_mtime), "rb"))
