"""Run notebooks/analysis.py top to bottom as a batch job.

The notebook is a cell file (`# %%` markers, IPython magics, display()). Each cell runs in an
IPython InteractiveShell in order, so magics work and names carry over exactly as in a kernel.
Figures are not shown but saved: plt.show is replaced by a function that writes every open
figure to $OUTPUT_DATA_DIR/analysis/<version>/figures/<cell>_<title>_<k>.png. The first failing
cell stops the run with its traceback and a non-zero exit.

    python scripts/run_notebook.py --version pilot_E --override-analysis --override-inference
    python scripts/run_notebook.py --version pilot_D --until "policy fits"   # cached fast path

The flags become the environment variables the notebook's config cell reads
(ANALYSIS_VERSION, OVERRIDE_ANALYSIS, OVERRIDE_INFERENCE).
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def split_cells(text):
    """(title, code) per `# %%` cell, in file order. Text before the first marker is a cell too."""
    cells, title, buf = [], "preamble", []
    for line in text.splitlines():
        if line.startswith("# %%"):
            if buf:
                cells.append((title, "\n".join(buf)))
            title, buf = line[4:].strip() or "untitled", []
        else:
            buf.append(line)
    if buf:
        cells.append((title, "\n".join(buf)))
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="pilot_E", help="pilot to analyse (ANALYSIS_VERSION)")
    ap.add_argument("--override-analysis", action="store_true", help="recompute sweep, tables and fits")
    ap.add_argument("--override-inference", action="store_true", help="rerun run_inference.py first")
    ap.add_argument("--until", default=None, help="stop after the first cell whose title contains this text")
    ap.add_argument("--notebook", default=str(ROOT / "notebooks" / "analysis.py"))
    args = ap.parse_args()

    os.environ["ANALYSIS_VERSION"] = args.version
    os.environ["OVERRIDE_ANALYSIS"] = "1" if args.override_analysis else "0"
    os.environ["OVERRIDE_INFERENCE"] = "1" if args.override_inference else "0"
    os.environ["MPLBACKEND"] = "Agg"
    os.chdir(ROOT)                     # the notebook locates the repo root from the cwd

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
    fig_dir = Path(os.environ["OUTPUT_DATA_DIR"]) / "analysis" / args.version / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state = {"cell": 0, "title": "", "n_fig": 0}

    def save_open_figures(*_, **__):
        slug = re.sub(r"[^a-z0-9]+", "-", state["title"].lower()).strip("-")[:50]
        for k, num in enumerate(plt.get_fignums()):
            path = fig_dir / f"{state['cell']:02d}_{slug}_{k}.png"
            plt.figure(num).savefig(path, dpi=150, bbox_inches="tight")
            print(f"[figure] {path}", flush=True)
            state["n_fig"] += 1
        plt.close("all")

    plt.show = save_open_figures

    from IPython.core.interactiveshell import InteractiveShell
    shell = InteractiveShell.instance()

    cells = split_cells(Path(args.notebook).read_text())
    print(f"{args.notebook}: {len(cells)} cells | version {args.version} | "
          f"OVERRIDE_ANALYSIS={os.environ['OVERRIDE_ANALYSIS']} OVERRIDE_INFERENCE={os.environ['OVERRIDE_INFERENCE']} | "
          f"figures -> {fig_dir}", flush=True)
    t_start = time.time()
    for i, (title, code) in enumerate(cells):
        state["cell"], state["title"] = i, title
        print(f"\n===== [{i:02d}/{len(cells) - 1}] {title}   ({time.strftime('%H:%M:%S')}, "
              f"{(time.time() - t_start) / 60:.1f} min in) =====", flush=True)
        t0 = time.time()
        result = shell.run_cell(code, store_history=False)
        if result.error_before_exec is not None or result.error_in_exec is not None:
            err = result.error_before_exec or result.error_in_exec
            print(f"\n===== FAILED in cell [{i:02d}] {title}: {type(err).__name__}: {err}", flush=True)
            sys.exit(1)
        save_open_figures()            # anything drawn without plt.show()
        print(f"----- cell [{i:02d}] done in {time.time() - t0:.0f}s", flush=True)
        if args.until and args.until in title:
            print(f"\n===== stopped after '{title}' (--until)", flush=True)
            break
    print(f"\n===== finished: {state['n_fig']} figures, {(time.time() - t_start) / 60:.1f} min total", flush=True)


if __name__ == "__main__":
    main()
