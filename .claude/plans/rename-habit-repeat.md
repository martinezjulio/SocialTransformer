# Plan: rename the habit term to repeat (H → R, HOS → ROS)

Repo copy of the approved plan (4 November 2026); chunks ticked as verified.

## Context

The win-stay lose-shift tilt on the previous arm is called the "habit" term, letter H, which names a mechanism the study does not test (the tilt could be memory, re-investigation or habit; the models only say that the previous arm and its outcome enter the policy). Decision (user, 4 Nov): call it the **repeat** term, letter **R**, so the models become R, RO, RS, ROS and the variants ROS+persev, ROS-reset, ROS+eps. The ModelSpec field `habit` becomes `repeat`; the parameters ρ0 and ρwin keep their names. Hypothesis labels H1 to H5, the legacy key `C1:H+Both` in the parity test, `%H:%M` date formats and the cross-entropy `H(...)` in `losses.py` are not model names and stay.

Inventory (read-only sweep, 4 Nov): 45 "habit" occurrences on 39 lines and 171 model-name occurrences on 106 lines across 16 tracked files; 3 bare-`H` model-name sites (`policies.py:59`, `fig3.py:24`, `ARCHITECTURE.md:548`); 3 derived LaTeX macros hard-coded in tests (`resElpdHOSEps`, `resPpcHOSSplitHalfZigzagInside`, `\resElpdHOS`). Nothing in README, scripts/, src/r/ or the sweeps. `numbers.KINDS` has no pattern on H or HOS, so every key keeps its kind; `macro_name` derives `resElpdROS` etc. with no collision (`elpd_R` → `resElpdR`, `elpd_recent` → `resElpdRecent`). Model names enter filenames through `fitting.cache_path` (`<sanitised name>_<hash>.pkl`, `+` → `_`), the checks keys `f"{c}_{a}_vs_{b}_{side}_{d:02d}"`, `ppc/<name>_draw*.json` and `split_half/<name>_split0*`; cached pickles also hold the name inside `Fit.name`/`Loo.name`, which nothing reads back as a label.

## Approach

One Python script in the scratchpad applies the substitutions to the tracked files with an asserted count per rule, so a drifted line fails loudly instead of silently; the prose rewrites that need new wording are explicit old→new pairs in the same script. A second script renames the pilot E cache files (filesystem only) after a dry-run listing. Then grep sweeps, import checks, two fast test files and the HTML render.

## Outline

- [x] 1. Code and tests (script, asserted counts)
  - [x] 1.1 Global, case-sensitive `HOS` → `ROS` in every tracked `.py`, `.md` and `.qmd` file that the inventory lists (covers `HOS+persev`, `HOS-reset`, `HOS+eps`, `eps_HOS_*`, `elpd_HOS`, `ppc_HOS`, the macro names in `tests/test_numbers.py` and `\resElpdHOS` in `tests/test_report.py`, and every prose mention); expected 171 − (HS, HO, H sites) ≈ 150 occurrences. Diff reviewed afterwards for any `ROS` that is not a model name.
  - [x] 1.2 Token sites for HS, HO and H, listed explicitly: `policies.py:58–59` (`"HS"`, `"HO"`, `"H"`); `run_analysis.py:121–122` (`("S3a", "HS", "HO")`, `("S3b", "HOS", "HO")`, `("own_on_top", "HOS", "HS")`); `fig3.py:24` RUNGS tuple `("HS", "HO", "H", ...)`, `fig3.py:6,70,74` (`HS minus HO`, `"HS − HO"`); `supp.py:25,55` REGISTERED keys and comment; `ARCHITECTURE.md:546–548` table rows; `preregistration.qmd:122,214,215` (`HS vs HO (S3a), HOS vs HO (S3b)`, `| HS |`, `| HO |`).
  - [x] 1.3 `habit` → `repeat` in code: the ModelSpec field and its comment (`policies.py:29`), the 18 `habit=True` keyword uses, `spec.habit` (`policies.py:159`, `predictive.py:75`), `test_checks.py:29–32` (`habit=`, `with_habit` → `with_repeat`, dummy model `"h"` → `"r"`, message "the repeat tilt raises the previous arm"); docstrings `policies.py:4` and `predictive.py:6`.
  - [x] 1.4 Letter expansions: `fig3.py:2`, `supp.py:10`, `CLAUDE.md:48`, `ARCHITECTURE.md:519` → "R repeat, O own experience, S social belief"; `ARCHITECTURE.md:528,539,545–548,568` prose ("repeat (win-stay/lose-shift tilt …)", "repeat tilt + own counts + …", "no repeat term", "Repeat + social", "Repeat + own", "Repeat only", "step 3 repeat/perseveration tilt").
- [x] 2. Preregistration (`reports/preregistration/preregistration.qmd`, explicit old→new pairs)
  - [x] 2.1 Line 176, the full model: "the perseveration or habit term (H)" → "the repeat term (R), a bias toward repeating the previous arm scaled by whether it just won"; "hence the name HOS" → ROS (by 1.1).
  - [x] 2.2 Line 204, the ladder: "Models missing the habit term drop the H" → "Models missing the repeat term drop the R"; line 208 table row "habit tilt (H)" → "repeat tilt (R)".
  - [x] 2.3 Lines 200 and 219: "for perseveration alone" → "for the repeat term alone"; "the perseveration, own-experience and social terms" → "the repeat, own-experience and social terms" (perseveration is the name of the ROS+persev bonus, a different term).
  - [x] 2.4 Line 298, E7: "OS (habit)" → "OS (repeat)", "habit positive" → "repeat positive".
  - [x] 2.5 Render HTML; `grep -ci 'habit' preregistration.html` = 0 and no "HOS"; unresolved refs 0.
- [x] 3. Pilot E caches (filesystem renames only, dry run printed first, no stage run)
  - [x] 3.1 `choice/fits` and `choice/loso`: filename prefix `HOS` → `ROS` (covers `HOS_eps_`, `HOS_persev_`, `HOS-reset_`), and whole-prefix `HS_` → `RS_`, `HO_` → `RO_`, `H_` → `R_` only when the prefix is followed directly by the 16-hex hash.
  - [x] 3.2 `checks/model_recovery` (datasets, `fits/`, `tables/`), `checks/ppc`, `checks/split_half` (and its `fits/`): substring `HOS` → `ROS` in filenames; these caches contain no HS/HO/H names. About 33 files in all.
  - [x] 3.3 Not touched: `choice/ladder.csv`, `paper/`, `figures/` and other stage outputs on disk still carry the old names and are regenerated by the next run; pickles keep the old name in `Fit.name` internally, which nothing reads as a label.
- [x] 4. Plan file: `.claude/plans/rename-habit-repeat.md` ticked; `.claude/plans/analysis-rebuild.md` gets one Findings line (names changed 4 Nov; older plan text keeps HOS).

## Verification (no pipeline stage is executed)

- `git grep -n -i 'habit'` → no hits; `git grep -n -E '\bHOS\b|\bHS\b|\bHO\b'` → no hits; `git grep -n '"H"'` → none outside H1–H5 contexts; the legacy `C1:H+Both` key is the only bare H allowed in tests.
- Import check only: `python -c "import analysis.policies, analysis.predictive, analysis.checks, analysis.numbers, analysis.report, analysis.figures.fig3, analysis.figures.supp"` from `src/`, in the conda env.
- Fast unit tests that touch no data or GPU: `pytest tests/test_numbers.py tests/test_report.py -q` (synthetic numbers; checks the renamed macros) and `pytest tests/test_checks.py -k "summaries or policy"` (hand-made frames and a one-shot FittedPolicy). The GPU recovery test and the parity test are not run.
- `quarto render preregistration.qmd --to html`, then the grep in 2.5.
- Cache rename: the dry run lists old → new for every file; after the rename, `ls` shows no `HOS` and the file count per folder is unchanged.
- No commit; the user runs the pipeline on Sherlock from fresh caches.

## Findings

- 4 Nov: applied in one scripted pass; zero residue of habit/HOS/HS/HO in tracked files; 36 pilot E cache files renamed; no stage run.
