# Plan: the paper (organisation, figures, supplement)

Started 26 Oct 2026. A living document: what the paper contains and where each piece comes from. It
collects the paper-facing decisions of `meta-plan.md` (§3 story order, §6 unseen expert, §8 figures,
§9 results file) and `preregistration.md` (what is registered), and grows as things come up. It is
not an Increment Mode checklist; sections are revised in place and changes are logged under `## Findings`.

Conventions: blue = expert, red = novice, grey = solo; 95% intervals everywhere (CI for lmer, ETI for
posteriors, paired SE for ΔELPD); every number in the text and figures comes from the pipeline's
`results.tex` / `results.md` (rebuild stage 9), never typed by hand; figure files have fixed names
(`fig2_primary.pdf`, `figS3_ppc.pdf`, …) with PNG copies [MP 9].

## Working title and claim

`[TODO: user]` title. One-sentence claim: participants learn whom to follow from demonstrations and
carry it to a new bandit; a transformer trained to read demonstrators' beliefs predicts their choices,
and the weight they place on the expert's belief rises across rounds.

## Main sections

1. **Introduction.** Learning whom to follow; demonstrations as evidence about a demonstrator's belief,
   not just about the arms; the readout as a belief estimate; the two-round transfer design.
2. **Results**, in the registered story order:
   1. What we did: task, three session types, demonstrators (expert 128 prior shots, novice 0;
      expertise hidden), the transformer readout, the policy-model pipeline (Fig 1).
   2. What happened (H1–H5, descriptive): after the expert beats after the novice; after-novice play
      improves across rounds; entropy falls; the partner choice H4; after-expert play beats solo's
      final game (Fig 2). Reward reported beside payoff without a rule; the after-novice improvement
      vs solo practice is exploratory.
   3. What explains it (policy models, Fig 3): the full model H+Both stated once with the fitting and
      comparison method; social beats own experience (S3a, S3b; the decomposition); whose belief (S1a
      expert over last-watched, S1b over the novice); a readout, not counts (S2a last-visible, S2b expert-visible, S2c naive); ablations
      (reported); the switch model: the weight on the expert's belief rises across rounds (S4), with
      Switch vs Switch-by-round ΔELPD reported to show the round term earns its keep.
   4. The surprise (Fig 4, exploratory, planned): the expert's readout matters before the expert is
      seen on the bandit; what the readout contains; the human exploit statistic.
   5. Per-participant characterisation (exploratory, ΔELPD dot plots in the supplement).
3. **Discussion.** The cautious reading of the round-2 switch (cannot separate "learned this
   demonstrator is a novice" from "learned how to use a demonstration" [MP 6.5]); the entropy miss
   and the perseveration rung; the two follow-up experiments (choose a partner each round; expertise
   tied to an environment); limits of the readout (frozen checkpoint, synthetic training).
4. **Methods.** Participants, exclusions and the registration (OSF link, tag, checkpoint hash); task;
   the transformer (architecture, training data, losses) in brief with a pointer to the model paper or
   supplement; the descriptive model (lmer, 12 cells, weight vectors, Holm); the policy models
   (equations and priors from the preregistration appendix; tables, cubic interpolation; NUTS gate and
   ladder; PSIS-LOO and paired SE); recovery and PPC; the unseen-expert analyses.

## Main figures [MP 8]

| fig | panels | source stage |
|---|---|---|
| 1 Schematic (hand-drawn, vector) | (a) the session: two rounds of watch, play, watch, play, partner choice; solo vs social, expert-first vs novice-first, around a screenshot of a watch game; (b) the demonstrators; (c) the transformer: tokens in, one Beta per arm per demonstrator out, with one real readout as a pipeline-rendered inset; (d) readout + own shots → belief → Thompson → habit → lapse → likelihood | 8.8 (inset only) |
| 2 Primary results | (a) session-in-order curves of payoff, reward, entropy, one line per session type, demonstrations marked at the watch games; (b) forest plot of the confirmatory contrasts (H1, H2, H3, H5) × payoff and entropy with 95% CIs, reward beside them as a reported outcome; (c) the partner choice (H4) | 8.2, 8.3 |
| 3 Policy models | (a) whose belief: ΔELPD of each source vs the expert's (S1a, S1b); (b) a readout vs counts by round (S2a–S2c); (c) ablations: ΔELPD of each rung vs the full model (S3 reads off here: H+Own, H+Social, H+Both); (d) ω after the novice and after the expert by round (S4) | 8.4, 8.5 |
| 4 The unseen expert | (a) Switch-8 weights laid out in session order, one line per condition, unseen-expert games marked (the hook); (b) what the readout contains: confidence and favourite-arm payoff by expert-seen × round, with one session's arm-by-arm beliefs as an inset; (c) the exploit statistic by round in the unseen cell | 8.6 |

Fig 4 is provisional [MP 8.4]: revisited on the full data, with the PPC figure and the
belief-vs-copying-by-participant figure as named fallbacks.

## Supplementary figures and tables [MP 8.5, plus additions]

Every supplementary item comes from a pipeline section; no hand-made figures.

- S-Fig: readout accuracy on the synthetic test set, from `eval.py` (payoff prediction vs baselines,
  correlation with true payoffs, inferred experience by cutoff).
- S-Fig: the readout's predicted total experience N_tot (α + β) on simulated sessions against the true count,
  by cutoff, from `eval.py` (the readout-accuracy figure extended to the experience readout; user, 26 Oct).
- S-Fig: free-running PPC with every named target (after-novice payoff rise, entropy fall, zig-zag,
  stay after win / loss), human mean against the 95% band, for H+Both and Switch-by-round; plus the split-half
  version (fit on half the participants, simulate the other half from the hierarchy) as an exploratory panel.
- S-Table: the full ladder (every rung's ELPD and its difference from H+Both with SE), including the
  reset rung and the perseveration rung; the S3 decomposition row.
- S-Fig/Table: parameter posteriors of H+Both and Switch-by-round (group level and per participant).
- S-Fig: per-participant ΔELPD dot plots (source, copying, ablations).
- S-Table: robustness fits: lmer random slopes and session order (H1, H2, H3, H5); κ trend and free lapse
  (κ, δ_N) for the policy models.
- S-Table: recovery: δ_N coverage and sign at the pilot effect; the S1, S2 and S4 model-recovery confusion table.
- S-Table: the unseen-expert ELPD split by expert-seen × round [MP 6.1b].
- S-Table: exclusion counts per rule (pilot E rates in the registration; full-run counts here).
- S-Fig: power curves by N at the pilot and half effect (from the registration's sampling plan).
- S-Fig: solo slots 1 and 3 (exploratory cells) beside slots 2 and 4.
- S-Fig: partner-choice recency check (expert-first vs novice-first).

## Where the numbers come from

`$OUTPUT_DATA_DIR/analysis/<version>/paper/`: `results.tex` (one macro per number, composed CI
macros, provenance: commit, checkpoint sha256, run date, participants, sessions, trials, exclusion
counts), `results.md`, `tables/*.tex` (hypothesis table, ladder), `figures/*.pdf|png`. The paper
repository copies the folder; the manifest check fails the pipeline on any missing number [MP 9].

## Open items

- Title, authors, target venue `[TODO: user]`.
- Whether the transformer gets its own methods subsection or a separate model paper.
- Fig 4 final form, after the full data.

## Findings

- 26 Oct (user): file created. New supplementary item: "model simulated sessions for predicted N_tot".
  Read as: the transformer's predicted total experience (α + β of the readout) on simulated sessions
  against the true count, by cutoff, i.e. the readout-accuracy figure from `eval.py` extended to the
  experience readout. Confirmed by the user the same day; it comes from `eval.py`, not the analysis pipeline.
- 26 Oct (user): split-half generative check added to the PPC supplementary figure (rebuild 6.8).
- 27 Oct (user): S2 and S3 widened in the registration (S2a/S2b, S3a–S3c, all parts required); results 2.3 and Fig 3(a, b) updated to match.
- 27 Oct (user): partner choice promoted to H6; secondaries renumbered in story order (old S2/S3/S5/S1/S4 → S1/S2/S3/S4/H6). Section and figure lists above use the new labels.
- 28 Oct (advisor, via the user): reward is a reported outcome (no rule); the former H4 (after-novice improvement vs solo) is exploratory; the partner choice is H4, before the solo comparison H5. Section and figure lists above use the new labels; earlier entries keep the old ones.
