# Plan: advisor's two comments on the mixed-effects models

Repo copy of the approved plan (4 November 2026); chunks ticked as verified.

## Context

The advisor asked (3 Nov) for condition in the partner-choice model and for the effect of play position within a round in the play-outcomes model. Decisions taken with the user (4 Nov):

- **Partner choice.** One model, `chose_expert ~ condition + (1 | participant)`, sum-to-zero coding. H4 stays the intercept (grand-mean log-odds of choosing the expert); the condition term (half the expert-first minus novice-first difference on the logit scale, i.e. recency) is exploratory, reported with its interval, no rule. The pipeline fits this model already as a second fit (E12); it becomes the only fit.
- **Play position.** Already a fixed factor crossed with condition and round, and balanced out of H1 and H2 by the counterbalancing; the document should say so. The pure effect of extra play on the same bandit is the solo second-minus-first play contrast, which is not registered or computed; it becomes an exploratory contrast. Participant-level variation in the play effect is E6's random-slope refit; it steps down to intercept-only when singular (pilot E: singular, slope SD 0 for payoff and entropy) but discards the variance components, so the boundary estimate is invisible in the results. E6 keeps the step-down rule and additionally records the slope standard deviations and the specification reached.
- **Pre-existing gap found while planning.** E5 (H2 within expert-first and within novice-first) is in the registered exploratory table but has no contrast in `descriptive.R` and no manifest keys. Folded in as its own chunk (strike it if unwanted).
- **Left alone, stated here.** `src/r/power_analysis.R` resamples H4 with its own intercept-only fit; the registered N is already computed and the intercept differs negligibly under the balanced design, so it is not changed. The Exploratory section is parked in `preregistration_text.md`; its table is edited there so the changes come back with it.

Findings from the code map: `numbers.py` lists every key explicitly and the report stage fails on missing **and extra** keys, so every new number needs a manifest entry; text-valued keys need a `KINDS` match or `fmt` raises; `report.py` has no exploratory or robustness table (results.md is one `| macro | key | value |` table per block); `fig_recency` in `supp.py` labels `model.csv` rows by term `(Intercept)` / `condition1`, so term names must not change; `fig2.panel_partner` and `hypothesis_table` take the first confirmatory row of `model.csv`; `fig_robustness` and `fig2.panel_forest` exclude exploratory rows, so new contrasts do not disturb them; `power_analysis.R` selects weight columns by confirmatory name, so a new column rides along unused; no test asserts on row counts or terms of these CSVs.

## Outline

- [x] 1. Preregistration document (`reports/preregistration/preregistration.qmd`)
  - [x] 1.1 Partner choice subsection: formula line becomes `chose_expert ~ condition + (1 | participant), family = binomial`; bullets: `condition` (expert-first / novice-first, sum-to-zero, so the intercept is the mean of the two conditions' log-odds and the condition term half their difference), `1` (the intercept, H4's estimand), `(1 | participant)`, `family`; Reporting paragraph adds one sentence: the condition term is reported with its 95% interval as the exploratory check of a difference between conditions (in expert-first the novice was watched last, so a nonzero term is a recency lean), no rule (~10 lines)
  - [x] 1.2 Play outcomes subsection: one short paragraph after the bullets (or after the solo-slot sentence): play position is a fixed factor crossed with condition and round; because demonstrator order is counterbalanced, first and second play games enter both sides of H1 and H2 equally, so play position cancels in them; the solo condition isolates extra play on the same bandit and its second-minus-first difference is an exploratory contrast; participant-level variation in the play effect is the random-slope sensitivity refit (~5 lines)
  - [x] 1.3 Parked exploratory table in `reports/preregistration/preregistration_text.md` (the "Moved out 3 November" block, edited in place with a dated note): E6 row adds "the slope standard deviations of every specification tried and the specification reached are reported"; new row E13 "practice within a round: solo second minus first play game, all outcomes, from the model of @sec-descriptive; expectation payoff near zero or small rise, entropy small fall"; E12 row reworded to "the condition term of the model of @sec-partner-choice"; contrasts table gains E5a/E5b and E13 expressions (~8 lines)
  - [x] 1.4 Render HTML and PDF; `grep -c 'Section ??\|Equation ??'` = 0; page count (0)
- [x] 2. Pipeline code
  - [x] 2.1 `src/r/partner_choice.R`: one `glmer(chose_expert ~ condition + (1 | participant))`; `fit_row` takes the fitted model and a term; H4 row = `(Intercept)` (confirmatory, sign +1), recency row = `condition1` (exploratory); header comment and the "one mixed logistic model each" comment updated; numbers keys unchanged (~8 lines)
  - [x] 2.2 `src/r/descriptive.R` contrasts: add to `CONTRASTS` `solo_play_change = m(demo == "none" & play == "second") - m(demo == "none" & play == "first")` (OUTCOMES, sign NA, exploratory) and E5: `H2_expert_first`, `H2_novice_first` = H2's difference within one condition (OUTCOMES, sign NA, exploratory); header comment (~6 lines)
  - [x] 2.3 `src/r/descriptive.R` robustness: `fit_reduced` keeps every attempted fit; new `robustness_variance.csv`, one row per outcome × spec tried: `sd_intercept, sd_round, sd_play, singular, converged, reached`; numbers `random_slopes_<outcome>_round_sd`, `_play_sd` (from the uncorrelated-slopes fit, the cleanest per-participant spread) and `random_slopes_<outcome>_spec` (the specification reached); header comment (~20 lines)
  - [x] 2.4 `src/analysis/numbers.py`: `EXPLORATORY` gains `solo_play_change`, `H2_expert_first`, `H2_novice_first` (all three outcomes); descriptive block gains the nine `random_slopes_*` keys; `KINDS` text pattern gains `_spec$`; `ARCHITECTURE.md` lines 499–500 updated (robustness_variance.csv; "mixed logistic model with condition and a participant intercept") (~8 lines)
  - [x] 2.5 `src/analysis/figures/supp.py` `fig_robustness`: legend label of the random-slopes fit names the specification reached, read from `robustness_variance.csv` (e.g. "random slopes (intercept only reached)") (~5 lines)
- [x] 3. Results files and verification on pilot E
  - [x] 3.1 Back up `$OUTPUT_DATA_DIR/analysis/pilot_E/{descriptive,partner_choice,paper,figures}` to the scratchpad (stages overwrite unconditionally) (0)
  - [x] 3.2 `bash scripts/run_analysis.sh --version pilot_E --stages descriptive,partner_choice,figures,report`; checks: `model.csv` has two rows from one fit (same `participant_sd`), H4 numbers close to the previous 0.71 [0.57, 0.81]; `contrasts.csv` gains 9 exploratory rows; `robustness_variance.csv` shows slope SD 0 and `reached = intercept_only`; `results.md` descriptive block lists the new keys; `paper/` builds with no missing/extra key error (0)
  - [x] 3.3 `pytest tests/test_numbers.py tests/test_report.py` (0)

## Verification

- Document: both renders clean, no unresolved refs; diff of the qmd limited to the two subsections.
- Code: pilot E run in 3.2 is the end-to-end test (R stages → manifest → report). Compare the new `model.csv` H4 row and `contrasts.csv` confirmatory rows against the backup: confirmatory estimates unchanged to three decimals (descriptive) and H4 within rounding (partner choice).
- No commit until asked; everything since `200ca2d` stays uncommitted.

## Findings

- 4 Nov: pilot E rerun clean; pre-existing contrast estimates unchanged; H4 0.705 -> 0.708 [0.571, 0.816] from the condition model; random-slope attempts singular for every outcome, intercept only reached; 7 tests pass.
