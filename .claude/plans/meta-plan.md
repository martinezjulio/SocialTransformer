# Meta-plan: decisions for the analysis overhaul and the pre-registration

A decisions document, no code. It replaces the premises of `analysis-rebuild.md` and
`preregistration.md` (both kept in place as reference until 10.3) after advisor feedback:
primary estimands come from the descriptive model, policy models are secondary, the design is
fixed at two rounds, the old notebook stays runnable as a separate thing, pilot E is the
development data, and the unseen-expert readout gets its own analysis.

Once every item is ticked, two concrete plans are written from it in Increment Mode:
`analysis-rebuild.md` v2 (the pipeline) and `preregistration.md` v2 (the document).

## Protocol

- One section per turn: brainstorm with options, pilot E evidence and a recommendation; the user
  decides; the choice goes on the item's `Decision:` line; the item is ticked; stop.
- Never renumber. A later section that changes an earlier decision adds a line under `## Findings`
  and edits the earlier `Decision:` line.
- Branch `analysis-module-overhaul`. Commits only when asked.

## Sources

- Pilot E (38 participants, 76 social sessions, 1,824 play choices, 2 rounds):
  `.claude/docs/analysis_update_10_19_2026/analysis_update_10_19_2026.md`.
- Pilot D (39 participants, 73 sessions, 3 rounds): `.claude/docs/analysis_update_10_12_2026/`.
- Model equations: the Model specification section of `preregistration.md` (moved from the deleted `prereg_brainstorming.md`).
- Old plans: `.claude/plans/analysis-rebuild.md` (architecture, modelling strand, orientation,
  open decisions), `.claude/plans/preregistration.md` (hypotheses, dependencies, open decisions).
- Legacy code: `notebooks/analysis.py`, `src/analysis/policy.py`, `src/analysis/descriptive.py`,
  `scripts/run_notebook.py`.

## 1. Legacy notebook isolation (first stage of the rebuild plan)

The notebook keeps running as it is today, with its supporting module moved under `notebooks/`,
and shares nothing with the new analysis package. Decided 24 Oct 2026.

- [x] 1.1 Layout: where the notebook, its module and the new package live; the legacy package name.
  Decision: `notebooks/analysis.py` and `analysis.ipynb` stay put. `src/analysis/{policy,descriptive,__init__}.py`
  move to `notebooks/legacy_analysis/` (package `legacy_analysis`). The new package takes over `src/analysis/`.
- [x] 1.2 The minimal edits that keep the notebook running (import path, module imports, the
  `fit_many` subprocess target).
  Decision: six spots and nothing else: the notebook's `sys.path` line gains `ROOT/"notebooks"`; its three
  `analysis.*` imports become `legacy_analysis.*`; `descriptive.py`'s import; `fit_many`'s PYTHONPATH and
  `-m legacy_analysis.policy` target; the notebook's cache root constant (`SWEEP_CACHE`, line 127) and
  `run_notebook.py`'s figure dir move to the `legacy_analysis` output root (1.4). Docstrings that name
  `src/analysis/` are updated in the same pass.
- [x] 1.3 `scripts/run_notebook.py`: stays in `scripts/` or moves beside the notebook.
  Decision: stays in `scripts/`; only its figure-dir root changes (1.4).
- [x] 1.4 Output and cache separation between legacy and the new pipeline.
  Decision: two sibling roots under `$OUTPUT_DATA_DIR`: `legacy_analysis/` (the notebook's `cached_data/`
  and `<version>/figures/`, moved once from today's `analysis/` on disk so the cached fast path still
  works) and `analysis/` (the new pipeline, `<experiment>/<version>/`). Neither writes into the other.
- [x] 1.5 Inference CSVs under `$OUTPUT_DATA_DIR/eval/<version>/`: shared read-only input, or may
  the new pipeline rewrite them.
  Decision: read-only input produced by `run_inference.sh`; the new pipeline checks provenance and has
  no `--inference run` stage.
- [x] 1.6 No cross-imports rule between `notebooks/` and `src/analysis/`, and how it is checked.
  Decision: nothing under `notebooks/` imports `src/analysis`; nothing under `src/` imports `notebooks/`.
  One short grep test plus one line in CLAUDE.md.
- [x] 1.7 Acceptance check for the move.
  Decision: `python -c "import legacy_analysis.policy"` from `notebooks/` (the subprocess path), then
  `python scripts/run_notebook.py --version pilot_E` on the cached fast path runs to the end and
  reproduces the pilot E figures and printed tables.

## 2. Primary estimands: the descriptive model

Decided 24 Oct 2026. Pilot E anchors (per-participant means, ± SE over 38): after the expert
payoff 0.62 → 0.64, reward 0.65 → 0.63, entropy 0.99 → 0.88 bits (round 1 → 2); after the novice
0.52 → 0.56, 0.50 → 0.56, 1.50 → 1.14; solo (slots 2 and 4) 0.56 → 0.54, 0.56 → 0.54, 1.27 → 1.06;
solo slot 4 alone 0.57 → 0.56, 0.56 → 0.57, 1.19 → 0.99. Round-2 minus round-1 change: after novice
payoff +0.04 ± 0.02, reward +0.06 ± 0.03, entropy −0.37 ± 0.09; after expert +0.02 ± 0.02,
−0.01 ± 0.03, −0.11 ± 0.10; solo −0.01 ± 0.02, 0.00 ± 0.03, −0.21 ± 0.08.

- [x] 2.1 Outcome set and the primary outcome among them.
  Decision: three co-primary outcomes per play game: payoff (mean true payoff of the chosen arms),
  reward (mean observed reward, same scale as payoff), entropy (bits over the six choices). Repeat/stay
  is a PPC statistic for §3, not a descriptive outcome.
- [x] 2.2 The claims and their contrasts; which are primary, secondary, or merely reported.
  Decision: directional hypotheses, each on payoff, reward and entropy unless stated.
  H1 (confirmatory): after the expert beats after the novice, averaged over rounds and conditions.
  H2 (confirmatory): after-novice play games improve from round 1 to round 2. The after-expert round
  change is exploratory, expected non-negative (a "≥ 0" claim is not testable without a margin).
  H3 (confirmatory): entropy falls from round 1 to round 2, averaged over social play games.
  H4 (confirmatory, payoff and reward): the round-2 improvement after the novice exceeds the solo
  improvement. The solo round effect is reported with its interval. The entropy version is
  exploratory: solo entropy falls too, so an entropy drop is not by itself evidence of social learning.
  H5 (confirmatory, payoff and reward): after-expert play beats solo's final game of the round
  (slot 4), averaged over rounds; round 2 also reported. Exploratory: social averaged over
  demonstrators vs solo slot 4 (expected smaller, positive); after the novice vs solo slot 4
  (expected similar).
  Reported only: Con-2, Con-5, the demonstrator effect on entropy, solo slots 1 and 3.
- [x] 2.3 Practice or social learning behind the round effect: the condition breakdown and the
  solo sessions as a control; whether solo sessions are in the full design.
  Decision: solo sessions stay in the full design (one per participant, as in pilot E) and are the
  practice control via H4 and H5. Solo play is mapped to the social play slots by information state:
  solo slots 2 and 4 (6 and 18 trials revealed before the game) are solo's first and second play; slots
  1 and 3 are exploratory. The condition breakdown of the round effect (novice-first first play game of
  round 2 = expert unseen) is reported, not tested.
- [x] 2.4 Estimator: numpyro hierarchical regression vs R (`lmer`/`brms`); random-effects
  structure; covariates; factor coding.
  Decision: R, not numpyro. `lmerTest::lmer` with Satterthwaite df and `emmeans` with explicit contrast
  weight vectors (precedent: `reports/r/reward_model.R`, `learning_vs_anchoring.R`,
  `notebooks/reward_mixed_models.ipynb`). One model per outcome on per-play-game rows:
  `outcome ~ condition * play * round + (1 | participant)`, condition ∈ {expert-first, novice-first,
  solo}, play ∈ {first, second}, round ∈ {1, 2}: a 12-cell factorial; demonstrator is derived
  (expert-first first play and novice-first second play = after the expert). Every hypothesis is a
  weight vector over the 12 cells. Exploratory robustness fits: random slopes for round and play by
  participant; session order (randomised per participant) as a covariate. The pipeline writes the
  per-game CSV; one R stage under the existing renv (lme4, lmerTest, emmeans) writes the emmeans tables.
- [x] 2.5 Decision rules and multiplicity; what counts as disconfirmation.
  Decision: two-sided α = 0.05 with the predicted sign, 95% CI reported. Holm across the outcomes within
  a hypothesis; no correction across hypotheses. Three verdicts: supported, not supported, contradicted
  (significant in the opposite direction). No smallest effect of interest, no equivalence testing.
- [x] 2.6 Which pilot E numbers the pre-registration quotes as predictions.
  Decision: none as predictions; hypotheses are directional only. Pilot E numbers appear as descriptive
  context and as the assumptions of the sample-size analysis, labelled exploratory. Sample size (feeds
  7.2): simulation-based power on pilot E (resample participants, refit the lmer at each N), 90% power
  for the weakest confirmatory test, reported beside the closed-form paired-difference N and the chosen
  N, inflated for the exclusion rate and bounded by budget.
- [x] 2.7 Exclusion rules (feeds 5.3).
  Decision: participant-level: red-green colour-vision prescreen on Prolific plus an in-task self-report
  item (demonstrators are told apart by colour); duplicate Prolific IDs; any one session in which every
  play shot stays on the game's starting position (the person is not playing, all sessions dropped);
  all three sessions faster than the speed floor. Session-level: incomplete session, missing game, any
  page refresh inside a main-session game (refreshes in instructions or practice do not count).
  Speed floor `[TODO: experimenter's fastest of three speed runs, measured per session]`; per-entry
  timestamps exist in the raw logs and must be carried into `preprocess.R`. No exclusion for
  randomness (it is the behaviour under study). No comprehension check exists. Every rule is applied to
  pilot E and its exclusion rate reported in the pre-registration.

## 3. Secondary outcomes: the policy models

Decided 24 Oct 2026. Pilot E anchors: habit +701 (SE 101) over Both; readout over own shots +269
(SE 42); expert source over recent +89 (SE 18), over novice +264 (SE 41); belief over copying the
last demonstration +44 (SE 16) round 1, +95 (SE 27) round 2; ω_N 0.52 → 0.90, Δ logit +2.15
[1.14, 3.23], P(>0) = 1.00; ω_E 0.95 → 0.96, Δ +0.22 [−1.14, 1.54]; C4 over C3 +11.5 (SE 5.9);
partner choice 53/76 expert, 70% [59, 79].

The strand is told in this order, which is also the order of the pipeline stages and of Fig 3:
1. the full policy model H+Both (belief from the readout plus own experience, Thompson sampling,
   habit, lapse), with the fitting and comparison method stated once;
2. whose belief: the expert's readout against the recent, novice and naive sources (S2);
3. a belief, not copying: the expert's readout against the visible shots of the last demonstration
   and of the expert (S3);
4. ablations of the identified model: without habit, without own experience, without the readout
   (reported);
5. the switch model: the weight on the expert's belief rises across rounds (S1);
6. why the expert's readout matters before the expert is seen (exploratory; content in §6);
7. the final partner choice (S4);
8. per-participant characterisation (exploratory, no committed method).

- [x] 3.1 The headline mechanistic claim and its estimand, with its rule.
  Decision: S1: in Switch-by-round, Δ = logit ω_N(round 2) − logit ω_N(round 1) > 0 at the group level,
  rule: the 95% ETI of Δ excludes zero on the predicted side (§10.1: 95% intervals everywhere). The estimand is ω_N because the model has one weight per last demonstrator and
  ω_E is at ceiling from round 1; the paper's headline reads "they rely on the expert's belief more in
  round 2 than in round 1", followed by the sentence that the change is carried by choices made right
  after the novice. Δ logit ω_E is reported as exploratory with the stated expectation that ω_E does not fall and may rise (a "≥ 0" claim is not testable without a margin, as for the after-expert round change in H2). Switch-by-round vs fixed-weight Switch
  ΔELPD is reported, not required.
- [x] 3.2 Ladder contrasts that get a rule vs the exploratory ones.
  Decision: S2: the expert source beats the recent source under H+Both. S3: H+Both with the expert
  readout beats the visible-pulls model of the last demonstration (the expert's own visible pulls
  reported alongside). Rule for every model-comparison claim: ΔELPD > 2 paired SE over participants.
  Reported only: habit, own experience, the readout vs own shots, novice and naive sources, carry vs
  reset, κ trends.
- [x] 3.3 Keep or drop: nuisance round trends on κ and ρ; per-participant round effect; fixed lapse.
  Decision: the headline switch model has no nuisance trends; one robustness fit adds the group-level
  trend on log κ (the existing C2 term) to show Δ survives generic sharpening. Trends on ρ dropped (never
  implemented). Per-participant round effect dropped (12 after-novice trials per round). Lapse fixed at 0.02.
- [x] 3.4 Comparison method and gates (PSIS-LOO, diagnostics, recovery preconditions).
  Decision: PSIS-LOO, paired SE over participants, Pareto-k fallback to leave-one-session-out. Gate per
  fit: R̂ < 1.01, bulk/tail ESS ≥ 400, zero divergences; failures are refit, never reported. Recovery only
  for the models behind a rule: parameter recovery of δ_N in Switch-by-round (90% intervals cover ≥ 85%
  over simulated datasets; sign recovered at the pilot effect) and model recovery for the three
  comparisons behind S1–S3. No recovery for the rest of the ladder.
- [x] 3.5 Posterior predictive targets (free-running) and the known entropy miss.
  Decision: free-running simulation of the fitted H+Both and Switch-by-round policies with the transformer
  in the loop. Named targets: the after-novice payoff rise (H2), the entropy fall (H3), the novice-first
  zig-zag, stay after a win and after a loss. Pass = human mean inside the 90% band of simulated means;
  these are checks, not hypotheses. One exploratory rung for the entropy miss (humans 0.3–0.5 bits more
  concentrated): H+Both plus a perseveration bonus on any arm already chosen in the game.
- [x] 3.6 Final partner choice: secondary with a rule, or exploratory.
  Decision: S4, confirmatory: P(choose the expert) > 0.5, mixed logistic model with a participant
  intercept, two-sided α = 0.05 with sign (it is model-free, so the §2 rule applies). Exploratory: no
  recency pull (expert-first vs novice-first), relation to ω_N.
- [x] 3.7 Per-participant characterisation (BMS, latent classes): exploratory or dropped.
  Decision: exploratory with no committed method; RFX-BMS and latent classes leave the pipeline plan. The
  existing per-participant ΔELPD dot plots stay as exploratory figures.

## 4. Two rounds, fixed

Decided 24 Oct 2026.

- [x] 4.1 Code consequence: no `N_ROUNDS` plumbing; the pipeline asserts two rounds.
  Decision: no round parameter anywhere in the new pipeline. The data loader asserts two rounds per
  session, four play games per social session, eight per solo session, six shots per play game, and
  fails loudly otherwise. Round is a two-level factor in every table and model.
- [x] 4.2 Model consequence: one δ per weight; Con-3/4/5 are single differences.
  Decision: Switch-by-round has two round terms, δ_N (the S1 estimand, = logit ω_N(2) − logit ω_N(1))
  and δ_E (reported), with round coded ±½. Every descriptive round contrast is a single difference.
- [x] 4.3 The three-round synthetic test set: readout-quality section stays (truncated) or moves to `eval.py`.
  Decision: the synthetic test-set evaluation of the readout (payoff prediction vs baselines,
  correlation with true payoffs, inferred experience by cutoff; the old notebook's section 1) leaves
  the analysis pipeline and lives with model evaluation (`src/eval/eval.py`, `scripts/run_eval.sh`).
  The analysis pipeline consumes only human-session readouts. A readout-accuracy panel for the paper,
  if wanted, is made from the eval outputs. No two-round synthetic set is generated.
- [x] 4.4 Pilot D is legacy-notebook only.
  Decision: yes. The new pipeline never reads a three-round dataset.

## 5. Pilot E as the development data

Decided 24 Oct 2026.

- [x] 5.1 Build and validate on `pilot_E`; parity targets are the 19 October tables.
  Decision: the pipeline is built on `pilot_E` and must reproduce the 19 October tables within Monte
  Carlo error: ladder ELPDs and contrasts, belief-source table, C4 weights by round, C5 split,
  partner-choice counts. Descriptive model (now R): lmer cell means equal the raw cell means, and
  H1–H3 agree in sign and rough size with the old numpyro contrasts.
- [x] 5.2 Use of the solo sessions.
  Decision: the practice control (H4, H5; §2). Because `run_inference.py` skips sessions without watch
  games, the descriptive stage reads `preprocessed_data.csv` directly (all sessions) and the
  policy-model stages read the inference CSVs (social sessions only); one provenance check confirms
  both describe the same participants.
- [x] 5.3 Exclusion rules frozen from `reports/r/preprocess.R` and the pilot E rates.
  Decision: all rules of 2.7 live in `preprocess.R`, which runs on anonymised data (5.5): the existing
  integrity rules (incomplete rounds, wrong shot count) gain page refresh and the speed floor; the
  participant-level rules (fixed position, colour vision, duplicate IDs) are added. One filtered CSV
  for every downstream reader. The pipeline reports each rule's exclusion count for the dataset it runs on.
- [x] 5.4 Status of pilots C, D, E in the pre-registration; whether another pilot precedes registration.
  Decision: the pre-registration discloses, in the existing-data section, that several pilots were run
  to work out the arm payoffs, the expert demonstrators' experience level, the framing and reward
  labelling, and the demonstrator, zone and expedition labelling throughout the game; the last pilot
  (pilot E, 38 participants) ran the final version and the models and hypotheses were developed on it.
  None of the pilot data enters the confirmatory dataset. No further pilot: the next run is the full
  experiment with N from the power analysis. If the task code changes (colour-vision item, timestamps),
  a technical check of the new fields by the experimenter or a handful of runs, not a pilot.
- [x] 5.5 Anonymisation split (new).
  Decision: `anonymize_participant_data.py`, uncommitted (gitignored), reads the raw logs and the Prolific
  demographics export, writes anonymised logs and demographics (including the colour-vision answer)
  and the ID mapping, all under the experiment data directory outside the repo. `preprocess.R`,
  committed, runs on anonymised data only and applies every exclusion rule, so the rules are pinned by
  the registration commit. Today `preprocess.R` does both and is split in the rebuild plan.

## 6. The unseen-expert readout

Decided 24 Oct 2026. Pilot E anchors: expert readout before the expert is seen on the bandit has
confidence 90–109, rank correlation with truth 0.18–0.22, favourite arm worth 0.53; after being
seen 140–146, 0.36–0.52, 0.76; the novice readout is flat (23–29, 0.00–0.07, 0.44). Expert over
novice belief on after-novice choices: round 1 +2 (SE 31) unseen / −4 (SE 24) seen; round 2 +124
(SE 34) unseen / +84 (SE 47) seen.

- [x] 6.1 A named pipeline section with fixed outputs (the readout-entering-play table; the elpd
  split by expert-seen).
  Decision: yes, two fixed outputs. (a) The readout-entering-play table by readout (expert, novice) ×
  expert seen on this bandit yet × round: confidence (total Beta count), rank correlation with the true
  payoffs, the favourite arm's true payoff, and the entropy of the Thompson policy on that readout.
  (b) The ELPD split of the expert belief against the novice belief on after-novice choices, by
  expert-seen × round. Both are ad hoc session code today. (§8.4: (b) is a supplementary table; the
  main-figure evidence is the exploratory eight-weight switch model.)
- [x] 6.2 What the unseen-expert readout is compared against.
  Decision: the copying belief, Beta(1 + wins, 1 + losses) from the novice's visible shots: rank
  correlation across arms between the two beliefs' means, and the ELPD of a policy on each. The
  ideal-observer posterior under the permutation prior is not done.
- [x] 6.3 The human side: exploit the evidence vs imitate the novice, in novice-first round 2 first play game.
  Decision: one model-free statistic, the share of shots on arms where the novice succeeded, by round in
  that cell, next to the ELPD split (6.1b) and the C5 copying contrast restricted to the cell. Exploratory.
- [x] 6.4 Role in the pre-registration and whether it is a paper figure.
  Decision: a planned exploratory analysis, described in advance in the pre-registration. Candidate
  Figure 4 of the paper (settled in §8).
- [x] 6.5 The interpretation committed to in writing, and the follow-up experiment it motivates.
  Decision: the cautious reading: in round 2 participants stop imitating a novice's demonstration and
  treat it as evidence to exploit, and the readout anticipates this; the data cannot separate "learned
  this demonstrator is a novice" from "learned how to use a demonstration", because the novice's belief
  is flat by design and both readings predict the same choices. This caveat is stated in the
  pre-registration next to S1. The two follow-up experiments from the pilot E note (choose a partner each
  round; expertise tied to an environment) belong in the paper's discussion, not in the pre-registration.

## 7. Pre-registration structure

Decided 24 Oct 2026.

- [x] 7.1 Confirmatory primary set, confirmatory secondary set, exploratory list; one table.
  Decision: one table (claim, estimand, model, rule, role) is the spine of the document. Confirmatory
  primary: H1–H5 (descriptive, lmer). Confirmatory secondary: S1–S4. Exploratory, as a closed list:
  after-expert round change; ω_E change; social-averaged and after-novice vs solo slot 4; entropy version
  of H4; Con-2, Con-5; the reported ladder rungs; the κ-trend robustness fit; the unseen-expert analysis
  (§6); per-participant ΔELPD plots; recency check on the partner choice; the perseveration rung; solo
  slots 1 and 3; the session-order and random-slope sensitivity fits. Anything not listed is post hoc.
- [x] 7.2 Sample size by simulation-based power analysis; power target; stopping rule.
  Decision: simulation on pilot E (resampled participants, refits at each candidate N) for H1–H5 and S4
  only (cheap lmer and logistic fits). S1 by scaling the pilot posterior SD of δ_N by √(38/N), labelled
  approximate; S2, S3 not powered (model comparisons, reported). Target 90% power for the weakest
  simulated test, reported at the pilot effect and at half of it, beside the closed-form paired N and the
  chosen N (inflated for the exclusion rate, bounded by budget). Stopping rule: recruit in batches until N
  participants with all three sessions complete survive the exclusions; no interim looks at any
  confirmatory outcome.
- [x] 7.3 Frozen artefacts: commit, checkpoint sha256, no retraining.
  Decision: the registration commit hash (a release tag), the checkpoint's sha256, pinned
  `requirements.txt` and renv lockfile, `preprocess.R` with the rules and the speed floor filled in. No
  retraining after registration; a new checkpoint would be a registered sensitivity analysis only. Any
  later deviation is reported as a deviation.
- [x] 7.4 Format (Quarto to PDF on OSF; OSF questions + modelling sections); length.
  Decision: Quarto to PDF, registered on OSF before recruitment of the full run. Sections follow the OSF
  Preregistration template plus the cognitive-modelling additions (model specification, priors,
  recovery, inference criteria). About ten pages plus an appendix with the model equations and prior
  tables (the Model specification section of `preregistration.md`). After collection the same document grows a results
  section. OSF mechanics (decided): the "OSF Preregistration" form (not Open-Ended), each field filled
  with the matching section's text; uploaded to the OSF project and frozen by the registration: the PDF, a
  hashes file (commit, checkpoint sha256, zip sha256), a zip of the repo at the release tag, and the
  checkpoint if under the per-file limit; the GitHub tag URL in the form and the PDF. The registration is
  embargoed and made public when the user decides.
- [x] 7.5 Preconditions and fallbacks for the secondary models.
  Decision: lmer: the main model is intercept-only and expected to converge; the robustness fits reduce
  random effects in a stated order (drop correlations, then slopes). NumPyro: the gate of 3.4, with
  pre-specified refits (longer warm-up, then the non-hierarchical ω version); a model that still fails
  makes its claim "not evaluable", reported as such, never replaced by another test. Recovery thresholds
  must be met on simulated data before the human result is interpreted; the document states what is
  reported if they are not. Pareto-k fallback to leave-one-session-out for S2 and S3.

## 8. Paper figures and results

Decided 24 Oct 2026. Four main figures in story order: what we did, what happened, what explains it,
the surprise. Plot convention: blue = expert, red = novice (in the game, expertise is hidden).

- [x] 8.1 Fig 1: schematic (task, conditions, data collection, readout, policy-model pipeline).
  Decision: four panels, hand-drawn in a vector tool. (a) The session: two rounds of watch, play, watch,
  play, the partner choice at the end; session types solo (all play) and social, the latter divided into
  expert-first and novice-first; built around a screenshot of a watch game from asteroidadventure.com
  showing the two starships. (b) The demonstrators: expert with 128 prior shots, novice with none;
  participants tell them apart only by the starship's colour, image and name, expertise hidden.
  (c) The transformer: demonstration tokens in, one Beta per arm per demonstrator out, with one real
  readout from a pilot session as a pipeline-rendered inset. (d) The policy-model pipeline: readout plus
  own shots → belief → Thompson sampling → habit → lapse → likelihood of the human choice.
- [x] 8.2 Fig 2: primary results (descriptive).
  Decision: (a) session-in-order curves for payoff, reward and entropy, humans only, one line per session
  type, demonstrations' values marked at the watch games; (b) forest plot of H1–H5 on all three outcomes
  with 95% CIs (the hypothesis table drawn); (c) the partner choice (S4) as a small panel. No separate
  cell-means panel: the lmer cell estimates are the points of (a).
- [x] 8.3 Fig 3: policy models.
  Decision: four panels in the §3 story order: (a) whose belief, ΔELPD of each source vs the expert's
  (S2); (b) belief vs copying by round (S3); (c) ablations, ΔELPD of each rung vs the full model;
  (d) ω after the novice and after the expert by round (S1, the registered estimand). The free-running
  PPC bands go to the supplement.
- [x] 8.4 Fig 4: candidates (unseen-expert readout; belief vs copying; PPC).
  Decision: the unseen expert, as a question and its answer, three panels. (a) The weight on the expert's
  belief in each play game laid out in session order, one line per condition, from an exploratory
  eight-weight switch model (one group-level weight per last demonstrator × round × condition, so each
  point is estimated from its own trials rather than copied from the round value as in the old C4
  figure), with the games where the expert has not yet been seen marked: the hook. (b) What the readout
  contains: confidence and favourite-arm payoff of the expert and novice readouts by expert-seen × round,
  with one session's arm-by-arm beliefs as an inset (expert readout before and after being seen, novice
  readout, copying belief). (c) The human exploit statistic, share of shots on the novice's successful
  arms by round in the unseen-expert cell. The ELPD split (6.1b) becomes a supplementary table. Fig 3(d)
  keeps the four-weight S1 estimand; Fig 4(a) is the per-cell picture. Provisional: revisited on the full
  data, with the PPC figure and the belief-vs-copying-by-participant figure as named fallbacks.
  Scratch estimate on pilot E (other parameters fixed at the C4 posterior, flat prior per cell): after the
  novice R1 0.36 (expert-first) / 0.62 (novice-first), R2 0.85 / 0.97 (unseen); after the expert 0.90–0.95
  throughout. Same shape as the old figure, hook sharper.
- [x] 8.5 Supplementary figures and tables.
  Decision: readout accuracy on the synthetic test set (from eval); the PPC with all named targets; the
  full ladder table and parameter posteriors; per-participant ΔELPD plots; the partner-choice recency
  check; the robustness fits (random slopes, session order, κ trend); the exclusion counts; the power
  curves; solo slots 1 and 3; the ELPD split by expert-seen × round (6.1b). Every supplementary figure
  comes from a pipeline section.
- [x] 8.6 The numbers each figure and the text need (input to §9).
  Decision: each pipeline section declares the numbers it emits (a manifest); the report stage fails if a
  declared number is missing. The list is written with the rebuild plan, one block per figure.

## 9. Results document for LaTeX

Decided 24 Oct 2026. The paper's numbers are never typed by hand: the pipeline emits them, the LaTeX
source includes one file, a rerun updates every number.

- [x] 9.1 Form: generated `results.tex` macros, plus `results.md` and CSV tables; figure files.
  Decision: one folder with `results.tex` (one `\newcommand` per number), `results.md` (every macro
  name and value as a readable table, also read by the pre-registration's results section),
  `tables/*.tex` (booktabs, from pandas: the hypothesis table, the ladder), `figures/*.pdf` with fixed
  names (`fig2_primary.pdf`, `figS3_ppc.pdf`, …) plus PNG copies. Each pipeline section writes a
  `numbers.json`; the report stage merges them.
- [x] 9.2 Macro naming scheme and number formatting.
  Decision: letters-only camel case with a common prefix: `\resHOnePayoffEst`, `…Lo`, `…Hi`, `…P`, plus
  one composed macro per quantity (`\resHOnePayoffCI` → "0.09 [0.08, 0.11]"). Formats fixed per quantity
  type: payoff, reward, logit, bits: two decimals; ELPD and SE: integers; percentages: integers; p-values:
  "< .001" below that, three decimals otherwise; intervals in square brackets; counts plain. The table of
  quantity types and formats goes in the rebuild plan.
- [x] 9.3 Location and provenance macros.
  Decision: `$OUTPUT_DATA_DIR/analysis/<experiment>/<version>/paper/`; the paper repository copies the
  folder, the pipeline knows nothing about the paper. Provenance macros in the same file: commit hash,
  checkpoint sha256, run date, participants and sessions after exclusions, trials, one count per
  exclusion rule.
- [x] 9.4 Which stage writes it; how a missing number fails.
  Decision: the report stage, last in the pipeline, merges every `numbers.json`, checks it against the
  manifest (8.6) and stops with an error naming any missing or extra number. No empty macros, no silent
  defaults. `results.md` is regenerated every run so a diff shows what changed.

## 10. Inherited decisions and the follow-on plans

Decided 24 Oct 2026.

- [x] 10.1 Audit of the old plans' decisions: keep, drop, revisit.
  Decision. Keep: NumPyro for the policy models; Thompson win probabilities in JAX, with a benchmark
  stage deciding tables vs the table-free GPU likelihood; PSIS-LOO with paired SE over participants;
  free-running PPC only; non-centred hierarchy and the diagnostics gate; λ_carry free; the provenance
  check of inference CSVs; per-dataset output directories; the inference rebuild scoped down
  (`run_inference.py` keeps its CLI and CSV schemas; session tokenisation and readout become importable;
  the only new piece is simulation with a fitted policy in the loop, for PPC and recovery).
  Drop: `deprecated/` (replaced by `notebooks/legacy_analysis/`); the round parameter; Switch-by-round as
  primary and the nuisance ρ trends; the pilot D snapshot and parity tests; RFX-BMS and latent classes;
  the numpyro descriptive models D1–D3 (lmer replaces D2; D3, the visible-rewards control, is dropped);
  recovery for the whole ladder; the synthetic test-set sweep inside the analysis.
  Revisit, settled: 95% intervals everywhere (95% CI for lmer, 95% ETI for Bayesian parameters; S1's
  rule is the 95% ETI excluding zero on the predicted side; ΔELPD > 2 SE unchanged). `scripts/run_analysis.sh`
  becomes the new pipeline's driver; the old Quarto R report script is renamed
  `scripts/run_quarto_analysis_report.sh`. Sampler budget default 4 × (1000 warm-up + 1000), raised only
  if ESS < 400. Recovery datasets: for Switch-by-round only, about 20–30.
- [x] 10.2 The two plans to write next and their top-level stages.
  Decision. `analysis-rebuild.md` v2 (code, Increment Mode):
  0. Preparation: JAX CUDA build and arviz, the `analysis/` output root, the Quarto script rename.
  1. Legacy isolation (§1).
  2. Preprocessing: `anonymize_participant_data.py` (uncommitted), the `preprocess.R` split, timestamps, the new
     exclusion rules and their counts (§2.7, §5.5).
  3. Inference, minimal: importable sessions and readout, the policy-in-the-loop simulator, schemas unchanged.
  4. Descriptive stage in R: per-play-game CSV writer (Python), `descriptive.R` with lmer and emmeans
     weight vectors (H1–H5, Con-2, Con-5, robustness fits), the partner-choice logistic model (S4), the
     power analysis (simulation plus closed form).
  5. Policy models: trial data, Thompson p_win with the benchmark gate, model registry (H+Both, belief
     sources, visible pulls, ablation rungs, Switch, Switch-by-round, κ-trend robustness, the exploratory
     eight-weight and perseveration variants), fitting with gate and cache, PSIS-LOO.
  6. Checks: δ_N parameter recovery, model recovery for S1–S3, free-running PPC with the named targets.
  7. The unseen-expert section (§6): readout table, copying-belief comparison, ELPD split table, exploit statistic.
  8. Figures 2–4 and the supplement, plus the Figure 1 inset.
  9. Report: `numbers.json` manifest, `results.tex` and `results.md`, tables, provenance; the
     `run_analysis.sh` driver; CLAUDE.md update.
  `preregistration.md` v2 (document):
  0. Skeleton `.qmd` with the OSF sections and the modelling appendix; pilot E inputs for the sampling
     plan only (effect sizes and variances for the power analysis, exclusion rates), never pilot results.
  1. Study information: question, background, the hypothesis table (H1–H5, S1–S4), the closed exploratory list.
  2. Design: task, three session types, demonstrators, randomised session order, practice session.
  3. Sampling: Prolific and prescreens, the exclusion rules with pilot E rates, existing-data statement
     (several pilots, as worded in 5.4), the three sample-size numbers, stopping rule.
  4. Models: the lmer with its weight vectors; policy-model equations and priors moved from
     `prereg_brainstorming.md`; gates, recovery, fallbacks.
  5. Analysis plan: rules, Holm, verdicts, robustness fits, PPC targets, planned exploratory analyses.
  6. Freeze and register: tag, hashes, zip, checkpoint upload, the OSF Preregistration form, embargo.
  Dependency: pre-registration stages 3–4 wait for rebuild stages 4–6 to have run on pilot E.
- [x] 10.3 Delete `.claude/plans/analysis-rebuild.md`, `.claude/plans/preregistration.md` and
  `.claude/docs/prereg_brainstorming.md` once the plans in 10.2 are written. Nothing is removed before then.
  Decision: as stated. Status 24 Oct 2026: both v2 plans are written in place (`analysis-rebuild.md`,
  `preregistration.md` overwritten; v1 text remains in git history, commits 8aaa404/d4c9bbd).
  `prereg_brainstorming.md` §3–4 moved into `preregistration.md` (Model specification) and the file
  deleted the same day; v1 plans overwritten in place. 10.3 done.

## Findings

- §10 (24 Oct 2026): intervals are 95% everywhere, so S1's rule in 3.1 changed from P(Δ > 0) ≥ 0.95 to
  the 95% ETI excluding zero. The old Quarto report script is renamed to free `run_analysis.sh`.
- §4–5 (24 Oct 2026): the synthetic test-set readout evaluation leaves the analysis pipeline for
  `eval.py`; the descriptive stage reads `preprocessed_data.csv` (solo sessions are skipped by
  inference); `preprocess.R` is split into an uncommitted anonymisation step and a committed cleaning
  and exclusion step (5.5).
- §3 (24 Oct 2026): the pre-registration's confirmatory secondaries are S1–S4 (Δ logit ω_N; expert vs
  recent source; belief vs copying; partner choice). Recovery and BMS stages of the old plan shrink to
  the models behind S1–S3; the C5 visible-pulls stage moves ahead of the ablations in the story and Fig 3.
- §2 (24 Oct 2026): the descriptive model moves from numpyro to R (`lmer` + `emmeans`); 5.3 now
  takes its exclusion rules from 2.7, and 7.2 its sampling plan from 2.6. H4/H5 add the solo sessions
  to the confirmatory design; the pipeline's per-game CSV must include solo play games and session order.

- 26 Oct 2026: recovery coverage (3.4: '90% intervals cover ≥ 85%') and the PPC band (3.5: 90%) superseded by the 95%-everywhere decision: recovery passes when 95% intervals cover the truth in ≥ 22 of 25 datasets; PPC band 95%. Recorded in both v2 plans.
- 27 Oct 2026 (user): 3.2 widened. S2 (whose belief) is now S2a expert vs recent and S2b expert vs novice; S3 (a readout, not counts) is S3a vs last-visible, S3b vs expert-visible, S3c vs the naive pooled count; each part ΔELPD > 2 paired SE, all parts required, per-part verdicts reported (the S5 pattern). Only habit and own experience stay reported-only rungs. Reason: the registered claim is that the expert readout beats every rival; the comparisons were already in the ladder. Pilot E (new pipeline): +89 (20), +247 (41), +53 (42), +169 (40), +200 (49).
- 27 Oct 2026 (user): 3.6 revised: the partner choice is H6, a sixth primary (model-free, glmer binomial, powered with H1–H5 in 7.2), so the primaries are exactly the descriptive claims and the secondaries exactly the policy-model claims. Secondaries renumbered in the §3 story order: S1 whose belief (old S2), S2 a readout, not counts (old S3), S3 social beats own (old S5), S4 the weight on the expert's belief rises (old S1). Earlier entries keep the old labels.
- 28 Oct 2026 (advisor meeting, via the user): §2 revised. Reward is reported with 95% CIs but carries no rule (payoff plus sampling noise, less power); the former H4 (after-novice improvement > solo improvement) is exploratory on every outcome; the partner choice is H4 (was H6) and precedes the solo comparison H5; Holm over payoff and entropy within H1 and H2, H3 and H5 single tests. N = 350 analysed participants (§7.2). Applied in the rebuild at 4.12 (descriptive.R, partner_choice.R, power_analysis.R, Fig 2). Earlier entries keep the old labels.
