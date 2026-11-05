# Plan: pre-registration v2 of the full transfer-bandits study

Written 24 Oct 2026 from `.claude/plans/meta-plan.md` ([MP n] points to its sections). Replaces the
first `preregistration.md`. Companion: `analysis-rebuild.md` (the pipeline whose numbers this pins).

## Context

Several pilots worked out the arm payoffs, the expert demonstrators' experience level, the framing and
reward labelling, and the demonstrator, zone and expedition labelling. The last pilot (pilot E, 38
participants, three sessions each) ran the final version; the models and hypotheses were developed on
it. The next run is the full experiment, with N from the power analysis, pre-registered on OSF before
recruitment, embargoed until the user decides [MP 5.4, 7.4].

The question: do participants learn whom to follow and carry it to a new bandit? Two rounds per
session, two bandits; three conditions per participant (expert-first, novice-first, solo), in
randomised order after a practice session.

Format: `reports/preregistration/preregistration.qmd` → PDF; OSF Preregistration form, each field
filled from the matching section; the PDF and the pilot E aggregate CSVs uploaded to the OSF project; the repos and checkpoint are identified by tag and hash and linked, not uploaded (advisor, 28 Oct) [MP 7.4].

Equations: every model or estimand that the document relies on is a numbered display equation
(`$$ ... $$ {#eq-name}`, referenced with `@eq-name`, including from the claims table); software calls
(`lmerTest`, `emmeans`, `glmer`, the formula string) stay inline in the text as the implementation of
the numbered equation. Set by the user on 27 Oct at 1.2; 1.3, 4.x and Appendix A follow it.

## What is registered (from the meta-plan)

**Confirmatory primary [MP 2.2]** (revised 28 Oct after the advisor meeting: reward and the former H4 are exploratory; the partner choice is H4, before the solo comparison), each on payoff and entropy per play game unless stated;
`lmer(outcome ~ condition * play * round + (1 | participant))`, condition ∈ {expert-first, novice-first,
solo}, play ∈ {first, second} (solo = round slots 2 and 4), round ∈ {1, 2}; contrasts as `emmeans`
weight vectors over the 12 cells; two-sided α = 0.05 with the predicted sign; Holm across outcomes
within a hypothesis; verdicts supported / not supported / contradicted; 95% CIs.

| | claim | cells |
|---|---|---|
| H1 | after the expert > after the novice (payoff ↑; entropy ↓) | social cells, averaged over rounds and conditions |
| H2 | after-novice play improves round 1 → 2 (payoff ↑; entropy ↓) | after-novice cells |
| H3 | entropy falls round 1 → 2 | social cells averaged |
| H4 (was H6) | participants choose the expert as partner: P(choose the expert) > 0.5, mixed logistic (glmer, binomial) with a participant intercept, two-sided α = 0.05 with sign on the intercept | one choice per session, outcome `chose_expert` |
| H5 | after-expert play > solo's final game (slot 4), averaged over rounds (payoff) | after-expert vs solo final play |

Reward (the realised Bernoulli outcome of the chosen arm) is reported for every contrast with 95% CIs but carries no rule: given the arms chosen it is payoff plus sampling noise, so it tests the same claim with less power (at half the pilot effect its H2 and former-H4 versions never reached 0.9 on the grid); it stays in the document as the quantity participants experienced and were paid on. The former H4 (after-novice improvement > solo improvement) is exploratory on all three outcomes. Holm is now over payoff and entropy within H1 and H2; H3, H4 and H5 are single tests.

**Confirmatory secondary [MP 3]**, in story order: S1 whose belief: the expert readout (HOS) beats the recent model (S1a) and the novice model (S1b) under HOS; S2 a readout, not counts: HOS beats
naive-recent (S2a), naive-expert (S2b) and naive (S2c); S3 social learning predicts choices better than own experience alone: S3a HS beats HO, S3b HOS beats HO; every ΔELPD part > 2 paired SE, all parts of a claim required, per-part verdicts reported; S4 Δ = logit ω_N(2) − logit ω_N(1) > 0 in Switch-by-round, 95% ETI
excludes 0.

**Exploratory, closed list [MP 7.1]** (stated in the document as analyses that may or may not be run, not a commitment; 28 Oct): after-expert round change (expected ≥ 0); ω_E change (expected
not to fall); social-averaged and after-novice vs solo slot 4; the former H4, after-novice improvement vs solo improvement, on payoff, reward and entropy (absorbs the entropy version of H4); the reward versions of H1, H2 and H5;
reported ladder rungs (habit, own experience); S vs O (expected positive: the no-habit version of S3a); κ-trend robustness fit; lapse robustness fits (HOS+eps, Switch-by-round+eps: ε, group κ and δ_N reported); the unseen-expert analysis (readout table, copying belief, ELPD split,
exploit statistic, Switch-8); per-participant ΔELPD plots; recency in the partner choice; the
perseveration rung; solo slots 1 and 3; random-slope and session-order sensitivity fits; split-half generative check (fit on half the participants, simulate the other half from the hierarchy, same PPC targets).

**Caveat registered with S4 [MP 6.5]**: the ω_N rise cannot separate "learned this demonstrator is a
novice" from "learned how to use a demonstration".

**Sampling [MP 2.6, 7.2]**: simulation-based power on pilot E (resampled participants, lmer refits)
for H1–H5 (H4 the partner choice); S4 by √(38/N) scaling of the pilot posterior SD, labelled approximate; 90% power for
the weakest simulated test at the pilot effect and at half of it; the closed-form paired N; the chosen
N = 350 analysed participants (advisor, 28 Oct; recruitment runs until N survive the exclusions, no budget bound). Stopping: recruit in batches until N participants with
all three sessions complete survive the exclusions; no interim looks.

**Exclusions [MP 2.7]** (all in `preprocess_participant_data.R`, rates on pilot E reported): participant-level: Prolific's colour-vision
prescreen (the export's Colourblindness column; no in-task item); any session with every play shot on the starting
position; whole-study logged span (practice included) under 300 s, the experimenter's fastest run being 4 min 30 s; more than one page load; not exactly three main sessions surviving the session-level rules. Session-level: incomplete session, missing game, reload during the session (refresh count above 1 or changing within it). No exclusion for randomness; no comprehension check exists.

**Frozen [MP 7.3]**: release tag and commit hash of both repos, checkpoint sha256, `requirements.txt`, `renv.lock`,
`preprocess_participant_data.R` with the speed floor filled in; no retraining of the transformer. Identified in the document by tag and hash; the repos, environment and checkpoint are not uploaded to OSF (advisor, 28 Oct).

## Orientation (to resume after /clear)

- **State on 29 Oct evening (main, 00d5af0).** Stages 0, 2–5 and 7 done (7 committed chunk by chunk on 29 Oct evening, after rebuild stage 10 landed in b20649b); 1.1 waits on the user's introduction rewrite (draft in `preregistration_text.md`); stage 6 waits for the tag. Resume in Increment Mode (off by default after /clear). This session edits only the qmd, `preregistration_text.md` and the plans; after every qmd edit rerender HTML and PDF; prose removed for rewriting is backed up verbatim in `preregistration_text.md`.
- **Why stage 7 exists.** On 29 Oct the document was read against the committed pipeline: the registry had moved to the HOS names, the power table had been regenerated with Holm over two outcomes, the pilot anchors in 1.3, 3.4 and 5 were legacy numbers, X2 reads Prolific's export (the game asks no colour-vision question), and the recovery paragraph described a pilot-fit truth and a reuse of datasets the code does not do. The user chose: truth from the analysed data's own fit; X2 on Prolific's column, game unchanged; S4 gated by δ_N recovery alone with its model recovery reported; demographics summarised in preprocessing; single consent, so the public release is the preprocessed file minus timestamps plus ledgers and aggregates.
- **Pilot numbers drift between reruns** (S4 2.49 → 2.42 across two days of cache rebuilds); every pilot number in the document is re-checked against the tagged run at 6.2, and any pilot verdict quoted from `results.md` carries the note that the pilot checks were a reduced run.

## Dependencies on the rebuild

| pre-reg chunk | needs rebuild stage | status 26 Oct |
|---|---|---|
| 0.3 pilot inputs, 3.2 exclusion rates | 2 (preprocess on pilot E) | done; `exclusions.csv` on disk |
| 3.4 sample size | 4.9 (power analysis) | done; `power/*.csv` on disk |
| 4.1 lmer spec, 4.4 partner choice | 4.4–4.7 (weight vectors as code) | done; `descriptive/*.csv` on disk |
| 4.2, 4.3, 4.5, 4.6 policy models, gates, PPC, unseen expert | none: text from the Model specification section and the 26 Oct criteria | unblocked |
| 6 freeze | the rebuild complete, tagged | waits |

Every chunk but stage 6 can be drafted now; `[TODO]` markers only where a value is still unknown.

## Plan file and progress protocol (Workflow rule)

Same as `analysis-rebuild.md`: one leaf at a time, verify, tick, report, stop; never renumber; splits
nest; dropped chunks struck through; changes via `## Findings`. Chunk sizes are lines of `.qmd`.

**Branch and commits.** Both plans work on `main` since 29 Oct (the overhaul branch was merged), same working tree, no
pre-reg branch: the two plans touch disjoint files, and the registration tag must hold both. Commits only
when asked, and always by path, never `git add -A` or `git commit -a`:
`git add reports/preregistration .claude/plans/preregistration.md` (plus `README.md` at 6.3), then
commit; check `git status` first so no rebuild file is staged. The rebuild session stages its own paths
the same way.

**User inputs are asked for when reached, not left as placeholders.** When a chunk contains
`[TODO: user]` (or needs a value only the user has), stop before implementing it and ask for the
value in chat; write the answer into the chunk's code or text and into the plan, then proceed. The
chunks that will trigger a question: 2.1 (what participants are told; incentives), 3.2 (speed floor value, if not yet set in rebuild 2.3), 3.4 (budget bound on N), 6.3 (OSF project, authors, embargo end), 0.1 (title and authors).

## Outline

- [x] 0. Skeleton
  - [x] 0.1 `reports/preregistration/preregistration.qmd`: YAML (PDF, title and authors as in Findings 26 Oct, date), OSF section headings + modelling appendix headings, `[TODO]` markers — qmd (75 lines, 53 non-blank; see Findings)
  - [x] 0.2 `quarto render` to PDF works standalone — shell (0); `pdf-engine: tectonic` (see Findings)
  - ~~0.3 Pilot E inputs for the sampling plan only: `reports/preregistration/pilot_inputs.csv` (effect sizes and SDs per outcome and cell from `descriptive` CSVs; exclusion rates from `exclusions.csv`); gitignored, regenerable — script (~25); waits for rebuild 2, 4~~ (struck 26 Oct: no code chunks in the document; the attachment is the preprocessed file, see Findings)
  - [x] 0.4 Adopt the Psych 251 report template (`github.com/psych251/experiment-template`, MIT): HTML with margin TOC as the reading format, PDF for OSF, narrative sections
    - [x] 0.4.1 Head: YAML (html + pdf/tectonic, email and ORCID on the first author, template credit comment), project-links callout (live task, experiment repo tag, analysis repo tag, OSF registration `[6.3]`, checkpoint `[6.1]`), AI use statement (Claude Code, Fable 5.1, helped set up the document and write and check the pipeline; the text is the author's own; no data generated or altered) — qmd (~35)
    - [x] 0.4.2 Body: registered-claims table slot (filled at 1.2–1.3); narrative headings with chunk pointers (Introduction; Hypotheses; Methods: existing data, participants and sampling, power analysis, materials, procedure, exclusions, analysis plan; Planned exploratory analyses; Variables; Frozen artefacts; Results placeholder; Data and code availability; Appendix A model specification; Appendix B OSF field-to-section mapping); render both formats — qmd (~40)
    - [x] 0.4.3 Appendix B: table mapping every OSF Preregistration form field (study information, design plan, sampling plan, variables, analysis plan, other) to the section whose text fills it — qmd (~28)
- [ ] 1. Study information
  - ~~1.1 Research question and background~~ (struck 2 Nov: the user wrote the unnumbered Introduction paragraph themselves; no separate section): learning whom to follow; the readout as a belief estimate; the existing-data statement in the user's wording [MP 5.4] — qmd (~35) (reopened 28 Oct: the draft was moved out of the qmd into `preregistration_text.md` for the user's own rewrite, so the section is a `[TODO]` marker again; the rewrite is short, about 15 lines, per the advisor, and drops the former H4 from its list of confirmatory claims)
  - [x] 1.2 Hypothesis table: H1–H5 with cells, outcomes, rules, verdicts — qmd (~35)
  - [x] 1.3 S1–S4 in story order with estimands and rules; the S4 caveat; ~~the S3 decomposition~~ — qmd (~30)
  - [x] 1.4 Exploratory closed list; post hoc rule — qmd (~20)
  - [x] 1.5 Advisor revisions of 28 Oct to the written claims
    - [x] 1.5.1 Claims table and 1.2: reward leaves the rule (reported with CIs, rationale stated: realised outcome of the chosen arm, payoff plus noise, lower power); the former H4 leaves the primaries; the partner choice becomes H4 and precedes the solo comparison H5; Holm over payoff and entropy within H1 and H2; every H label in 1.2, 1.3, 1.4, 3.4 and the figures text renumbered — qmd (~25)
    - [x] 1.5.2 Exploratory list 1.4: a bold statement at the top that the list is the set of analyses that may be run, none of them a commitment, and that anything outside it is post hoc; new entries for the former H4 on all outcomes (absorbing the entropy version) and for the reward versions of H1, H2 and H5; E labels renumbered — qmd (~15)
    - [x] 1.5.3 Plain-language claims: brainstorm in chat how the Claim and Estimand columns of the claims table read without cell jargon (each row a sentence a reader outside the project follows), then apply the chosen wording — qmd (~12); the reading of the advisor's "plan language" as "plain language" is to be confirmed by the user
- [x] 2. Design
  - [x] 2.1 Task: 12 arms, payoff set, 2 rounds × (watch, play, watch, play) + partner choice; solo = 2 rounds × 4 play games; demonstrators (expert 128 prior shots, novice 0; identified by starship colour, image, name; expertise hidden); what participants are told `[TODO: user]`; incentives `[TODO: user]`; links: live task https://asteroidadventure.com/ (the version deployed at registration) and the experiment repo https://github.com/martinezjulio/AsteroidMiningAdenture at a tagged commit — qmd (~40)
  - [x] 2.2 Three conditions per participant, randomised order, practice session, bandit permutation randomisation — qmd (~20)
- [x] 3. Sampling plan
  - [x] 3.1 Population and eligibility (Prolific prescreens: colour vision, English first language or at minimum English fluency, `[TODO: user, ask at this chunk]` any residence/approval-rate filters; no pilot participants); check that the consent text permits public sharing of anonymised trial data (the OSF attachments go public when the embargo lifts) — qmd (~15)
  - [x] 3.2 Exclusion rules verbatim from `preprocess_participant_data.R`, with pilot E rates; participant- vs session-level; speed floor 300 s and its rationale — qmd (~30); waits for rebuild 2
  - [x] 3.3 Stopping rule, batches, no interim looks — qmd (~10)
  - [x] 3.4 (written 27 Oct; revised 28 Oct: chosen N = 350; cost table out; reward and former-H4 rows out of the power tables, H6 rows relabelled H4; the target paragraph recomputed on the remaining tests: 1.00 for every test at the pilot effect from N = 200, at half the effect the weakest is H2 payoff at 0.87 for N = 350 and H5 payoff at 0.99; budget bound stated by the user) Sample size: the simulation (method, assumptions from pilot E labelled as such), power table and figure by N at pilot and half effect, closed-form N, chosen N — qmd (~40); waits for rebuild 4.9
- [x] 4. Variables and analysis plan
  - [x] 4.1 Descriptive model: outcomes per play game, factors and coding, the lmer formula, the 12-cell weight vectors for H1, H2, H3, H5 (confirmatory) and for the exploratory contrasts including the former H4; reward as a reported outcome; Holm over payoff and entropy; verdicts; robustness fits; state that the full 12-cell grid plus solo slots 1 and 3 is reported with 95% CIs (solo slots 1 and 3 left the exploratory list, 27 Oct) — qmd (~40); waits for rebuild 4.5
  - [x] 4.2 Policy models: H+Both equations and priors, ladder table, Switch and Switch-by-round, Switch-8 (exploratory), perseveration (exploratory); from the Model specification section of this plan — qmd (~40)
  - [x] 4.3 Fitting and inference: NUTS settings, gate, refit ladder, PSIS-LOO and paired SE, Pareto-k fallback (exact leave-one-session-out for sessions with k > 0.7), recovery preconditions (95% intervals cover the truth in ≥ 22 of 25 simulated datasets; sign recovered at the pilot effect; model-recovery confusion diagonal-dominant) and what is reported if unmet, "not evaluable" rule; state that the per-participant ELPD differences behind every paired SE are reported and plotted (left the exploratory list, 27 Oct) — qmd (~35)
  - [x] 4.4 Partner choice model (H4, formerly H6, glmer) and recency check — qmd (~10)
  - [x] 4.5 Posterior predictive checks: free-running simulation, named targets, pass criterion (95% band); the split-half variant (fit on half the participants, simulate the other half from the hierarchy, same targets and band; left the exploratory list, 27 Oct) — qmd (~22)
  - [x] 4.6 Planned exploratory analyses: the unseen-expert section as it will be run — qmd (~25)
  - [x] 4.7 Variables summary: manipulated, measured, derived (readout from the frozen checkpoint; own counts; policy parameters) — qmd (~25)
- [x] 5. Appendix
  - [x] 5.1 Priors table and notation; the 12-cell weight vectors of H1, H2, H3, H5 and the exploratory contrasts as a table (one column per hypothesis, one row per cell, copied from `descriptive.R`'s printed table) with a definition column giving each contrast as a cell-selection rule (e.g. H1 = mean(after expert) − mean(after novice); H4 = the difference of the two round differences), plus the three calls once beside it: the `lmerTest` formula, the `emmeans(fit, ~ condition * play * round)` grid, and `contrast(emm, method = list(w = w))` with a sentence that each table column is the `w` argument — qmd (~45)
  - [x] 5.2 Results-section placeholder reading `results.md` after collection — qmd (~10)
- [ ] 6. Freeze and register
  - [ ] 6.1 Versioned task URL, release tag and hashes
    - [x] 6.1.1 Task URL <https://asteroid-mining-game-31128277324.us-east4.run.app/> in the links block and 2.3 (user, 29 Oct) — qmd (2)
    - [ ] 6.1.2 Release tag on https://github.com/martinezjulio/SocialTransformer; hashes file (analysis commit, experiment repo commit, checkpoint sha256) kept in the analysis repo at the tag and quoted in the document; no repo zip, no checkpoint or environment upload (advisor, 28 Oct); `renv.lock` and `requirements.txt` pinned at the tag — shell (0)
  - [ ] 6.2 Consent text, X2 clause and the final consistency pass
    - [x] 6.2.1 The consent page's text quoted verbatim in 3.1 (single consent covers participation and de-identified public sharing; revised 29 Oct to name the Prolific-provided fields and to promise no identifying information; the game is being updated to this exact text) and the X2 clause of 3.2 — qmd (~8)
    - [ ] 6.2.2 ~~The X2 clause (done in 6.2.1)~~; render; consistency pass: every number has a source, every hypothesis a rule, no `[TODO]` — shell (0)
  - [ ] 6.3 OSF: project, upload PDF, ~~hashes file, repo zip, checkpoint~~ (links to the tagged repos and the hashes in the document suffice; advisor, 28 Oct), ~~the anonymised preprocessed trial file~~ (pilot participants did not consent to sharing, 27 Oct) the pilot E aggregate outputs: `descriptive/{cell_means,contrasts,robustness}.csv`, `partner_choice/{model,rates}.csv`, `choice/ladder.csv`, `power/{decision_rates,closed_form_n}.csv`, `exclusions.csv` counts; the full run's public release, stated in the data-availability section: `preprocessed_participant_data_public.csv` (rebuild 10.10: the preprocessed file without timestamps), `exclusions.csv`, `demographics_summary.csv` and the aggregate outputs, no per-participant demographics file, from which inference and every stage from `games` on rerun with the tagged code and checkpoint; OSF Preregistration form from Appendix B; embargo; record the registration URL in the qmd and `README.md` — (~5)

- [x] 7. Consistency with the pipeline (29 Oct review of the document against the committed stages and the pilot E outputs)
  - [x] 7.1 Model names in the HOS convention (rebuild 8.10, commit 450564e: HOS, recent, novice, naive, naive-recent, naive-expert, HS, HO, OS, S, O, H, Random, HOS+persev, HOS-reset, HOS+eps; Switch names unchanged)
    - [x] 7.1.1 Sections 1 and 8: claims-table Model column (S1–S3), 1.3 prose (full model, rivals, S3 models), E7 and E10 rows, results item 4 ("difference from HOS") — qmd (~20)
    - [x] 7.1.2 Sections 4.7.2 (one sentence giving the letter code: H habit, O own experience, S social belief; rivals named by their belief source; ladder table and Switch prose), 4.7.3 (fallback and recovery sentences), 4.7.5, 5 (c), A.1 Models column — qmd (~30)
  - [x] 7.2 Power section 3.4 from the current `power/decision_rates.csv` (28 Oct 13:42, Holm over payoff and entropy): half-effect rows for H2, H3, H5 (H2 payoff .67 .72 .86 .91 .93 .96 .98 1.00 1.00; H2 entropy .98 .99 1.00 …; H5 payoff .95 .97 .99 1.00 …); delete the "run with reward still among the tested outcomes, Holm over three" sentence; "Chosen sample size" paragraph (H2 payoff 0.91 at N = 350); 4.7.1's "4,500 resampled datasets" → 9,000 (500 × 9 N × 2 effects) — qmd (~15). (precondition verified 29 Oct: the roles were committed 28 Oct 13:31, the table regenerated 13:42 in b5bedc2)
  - [x] 7.3 Pilot numbers from the current pipeline run (all re-checked at the tag in 6.2): S4 in 3.4 → δ_N 2.42 [1.38, 3.60] (`choice/verdicts.csv`), projection sentence recomputed; 1.3's "S2a fell short in pilot E" sentence replaced (current run +134, SE 32, supported); section 5 anchors replaced by `unseen/readout_table.csv` and `unseen/numbers.json` values (expert unseen confidence 91 and 109, rank correlation 0.22 and 0.18, favourite 0.55 and 0.52; seen 134 and 145, 0.46 and 0.46, 0.75 and 0.77; novice 23–29, 0.01–0.06, 0.44–0.45) and the ELPD split stated as HOS minus recent: +10.5 (SE 7.6) unseen and −1.4 (8.7) seen in round 1, +42.1 (6.9) and +39.9 (12.5) in round 2 — qmd (~15); the old anchor paragraph goes to `preregistration_text.md`
  - [x] 7.4 Exclusions and colour vision: 3.1 drops "an in-task self-report item asks the same question" and states that the prescreen is Prolific's stored profile self-report, unverified and possibly old, so it removes people who know they are colour blind and no more, with the partners and asteroids also told apart by name and image (user, 29 Oct: the game stays as it is); 3.2 X2 rule "answered yes to Prolific's colour-blindness prescreen (the export's Colourblindness column)"; the "colour-vision item" examples in 2.1 (existing data) and 3.2's closing sentence removed; X4 pilot count 1 and the timed-out participant attributed to X4 alone (X11 = 0, the ledger applies X4 first); the order sentence names X3 and X4 as participant-level rules on the logs before the session rules; 2.2 drops "ends with a short demographics questionnaire" (no questionnaire exists; user, 29 Oct); section 6 Measured: "Per participant: age and sex from Prolific's profile export, summarised by the preprocessing script over the analysed sample (mean, SD, range; counts) and reported descriptively; they enter no analysis and are not released per participant" (rebuild 10.9); 3.1's consent paragraph: a single consent covers taking part and the anonymised public sharing of the data (user, 29 Oct), so the sentences on a separate sharing item, on the public dataset being a subset of the analysed one and on the released file stating how many it holds are removed — qmd (~18)
  - [x] 7.5 Analysis-plan wording: 1.2 and 4.7.1 reward versions exist for H1, H2, H5 (not H3); 4.7.3 λ_c interpolation (log axis above 0.25, linear through the λ_c = 0 knot below); 4.7.3 recovery paragraph: the truth is a draw of the Switch-by-round fit on the data being analysed, run in the same pipeline pass before any verdict is read, cached per version, no pilot reuse (user, 29 Oct); pass rules as the code will compute them after rebuild 10.2 (coverage ≥ 22 of 25 and the interval on the truth's side in ≥ 22; model recovery, gating S1a and S2a only: the richer or registered model A wins a majority of the datasets generated from A and at most a minority of those generated from its rival, undecided counting as correct on the rival's side; the S4 model recovery is reported beside E9 and S4 is gated by δ_N parameter recovery alone (user, 29 Oct); the "not evaluable on a failed check" sentence names the report stage as where the masking happens (rebuild 10.5); section 5 (d): the third column is the share of shots on arms the novice tried at all; 1.3's comparison-rule sentence: a claim is *not evaluable* when a fit failed the gate or when the claim's registered check failed, as decided in the report stage — qmd (~32)
  - [x] 7.6 Reported quantities named as pipeline outputs (rebuild 10.3–10.5 done; 10.9 for the demographics keys): E7 Social − Own as a paired row of the choice stage; E8 δ_E as posterior mean and 95% ETI; E10 lists exactly the reported numbers, δ_κ and δ_N from the κ-trend fit, ε and group κ from HOS+eps, ε and δ_N from Switch-by-round+eps, each with its 95% ETI; 4.7.3 fallback sentence states the choice stage applies it and reports the sessions refitted; section 8 item 2 names the check-gated final verdict with the choice stage's verdict beside it as the description, item 4 adds the sessions refitted — qmd (~14)
  - [x] 7.7 Plan file only: the "What is registered" summary drops the in-task item, says H1–H5 with H4 the partner choice, drops "bounded by budget" and uses the HOS names; the Model specification block renames H+Both and friends to HOS, HS, HO, OS, S, O, H and `seg` → `curr` — .claude/plans/preregistration.md (~15)

## Verification

- 0.2 and 6.2: PDF renders with no `[TODO]`, no missing citations.
- 3.4: power table printed by `power_analysis.R` matches the qmd numbers.
- 4.1: the weight vectors in the qmd equal those in `descriptive.R` (copied from its printed table).
- Before 6.3: `bash scripts/run_analysis.sh` runs end to end on pilot E at the tagged commit.
- 7.1: `grep -n 'H+Both\|src:\|vis:\|H+Social\|H+Own' preregistration.qmd` returns nothing.
- 7.2: the qmd half-effect rows equal `power/decision_rates.csv` pivoted (`rate` by N, two decimals).
- 7.3: every pilot anchor equals its CSV/JSON value at two decimals; each chunk of stage 7 rerenders HTML and PDF with no new `[TODO]`.

## Open decisions (default in bold)

1. Power target: **0.9 at the pilot effect, reported also at half**; no budget bound: recruitment runs until 350 survivors (user, 28 Oct).
2. Length: **~10 pages + appendix**.
3. Embargo length: **until the user lifts it** (max allowed by OSF).

## Model specification (source text for chunks 4.2 and 5.1)

Moved here from the deleted `prereg_brainstorming.md` §3–4, in the two-round form. The descriptive
model of that document (§2, numpyro D2) is superseded by the lmer specification above.

**Notation.** $i$ participant, $s$ session, $r \in \{1, 2\}$ round, $g \in \{1, 2\}$ play game within the
round, $t \in \{1, \dots, 6\}$ trial; $k \in \{1, \dots, 12\}$ arm, $a_t$ the arm chosen; $c$ condition,
$d$ last demonstrator watched; $\alpha^{\text{demo}}_k, \beta^{\text{demo}}_k$ the readout (the transformer's
estimate of a demonstrator's belief about arm $k$ as a Beta); $s^{\text{curr}}_k, s^{\text{prev}}_k$ the
participant's own (wins, losses) on arm $k$ in this play game and in the earlier play game of the round;
$v_k$ the pulls visible on arm $k$ in a demonstration (wins, losses).

**HOS, four steps.**

1. Belief: $\alpha_k = \kappa_i\, \alpha^{\text{demo}}_k + \lambda_{\text{w}}\, s^{\text{curr}}_{k,\text{win}} + \lambda_{\text{c}}\, s^{\text{prev}}_{k,\text{win}}$, and the same for $\beta_k$ with losses.
   $\kappa_i$ scales the readout ($1$ at face value, $< 1$ flatter, $> 1$ sharper); $\lambda_{\text{w}}$ weights own
   outcomes in the current play game, $\lambda_{\text{c}}$ those carried from the earlier play game of the round.
2. Thompson sampling: $p_k = P(\theta_k = \max_j \theta_j) = \int_0^1 f_k(x) \prod_{j \ne k} F_j(x)\, dx$, with
   $f_k, F_k$ the density and distribution function of $\text{Beta}(\alpha_k, \beta_k)$.
3. Win-stay lose-shift: $q_k = p_k \exp\big((\rho_{0,i} + \rho_{\text{win},i}\, r_{t-1})\, \mathbb{1}[k = a_{t-1}]\big)$,
   $r_{t-1} \in \{0, 1\}$ the previous reward; no bonus on the first trial of a play game.
4. Choice: $P(a_t = k) = (1 - \varepsilon)\, q_k / \sum_j q_j + \varepsilon / 12$, $\varepsilon = 0.02$ fixed (robustness: refitted free at group level, see priors).

**Hierarchy and priors.** Non-centred: $\log \kappa_i = \mu_\kappa + \tau_\kappa z_i$, $z_i \sim \mathcal{N}(0, 1)$;
likewise $\rho_{0,i}$, $\rho_{\text{win},i}$. Own-outcome weights are group level.

| parameter | prior | reads as |
|---|---|---|
| $\mu_\kappa$, $\tau_\kappa$ | $\mathcal{N}(0, 0.5)$, $\text{HalfNormal}(0.3)$ | $\kappa$ centred on 1, most mass in $[0.4, 2.5]$ |
| $\log \lambda_{\text{w}}$ | $\mathcal{N}(0, 1.5)$ | own outcomes count about as one readout pull, give or take a factor of 10 |
| $\lambda_{\text{c}}$ | $\text{HalfNormal}(4)$ | includes 0: carrying nothing from the earlier game |
| $\mu_{\rho_0}$, $\mu_{\rho_{\text{win}}}$ | $\mathcal{N}(0, 2)$ | stay or shift of either sign |
| $\tau_{\rho_0}$, $\tau_{\rho_{\text{win}}}$ | $\text{HalfNormal}(1)$ | |
| $\mu_{E}, \mu_{N}$ (logit $\omega$) | $\mathcal{N}(0, 1.5)$ | mixture weights, switch models |
| $\tau_{E}, \tau_{N}$ | $\text{HalfNormal}(1)$ | |
| $\delta_E, \delta_N$ | $\mathcal{N}(0, 1)$ | round change of the logit weights |
| $\delta_\kappa$ | $\mathcal{N}(0, 0.5)$ | κ-trend robustness fit only |
| $\varepsilon$ | $\text{Beta}(1, 9)$ | lapse robustness fit only (HOS, Switch-by-round); fixed at 0.02 everywhere else |

**The ladder.** Every rung is HOS with terms switched off; comparison by PSIS-LOO ELPD summed over
trials, SE of a difference over participants. S3 reads the ladder as a decomposition: ELPD(HOS) − ELPD(HO) = [ELPD(HOS) − ELPD(HS)] + [ELPD(HS) − ELPD(HO)], the second bracket being S3a; the first is reported (own experience on top of the readout).

| rung | belief | WSLS | asks |
|---|---|---|---|
| Random | uniform, nothing fitted | no | chance, $\log(1/12)$ per trial |
| Naive | $\text{Beta}(1,1)$ + every pull seen on this bandit, own and both demonstrators' | no | counting without attribution |
| H | uniform | yes | WSLS alone |
| O | $\text{Beta}(1,1) + \lambda_{\text{w}} s^{\text{curr}} + \lambda_{\text{c}} s^{\text{prev}}$ | no | own experience alone |
| S | $\kappa_i \alpha^{\text{demo}}$ | no | the readout alone |
| OS | step 1 | no | readout and own experience |
| HO, HS, HOS | as above | yes | does WSLS add to each belief |
| recent, novice | HOS reading the expert's, novice's, or last-watched readout | yes | whose belief (S1a: expert vs last-watched; S1b: expert vs novice) |
| naive-recent, naive-expert, naive | HOS with the readout replaced by $\text{Beta}(1,1)$ + the pulls visible in a demonstration (the last one, or the expert's), scaled by $\kappa_i$; or the naive count, $\text{Beta}(1,1)$ + every pull seen this round, own included | yes | a readout, not counts (S2a: vs the last demonstration; S2b: vs the expert's; S2c: vs the naive count) |

**Switch.** Two HOS policies at the same $\kappa_i, \lambda, \rho$, one on the expert's readout and one on
the novice's, mixed after step 3:
$P(a_t = k) = (1 - \varepsilon)\,[\omega_t\, \tilde q^E_k + (1 - \omega_t)\, \tilde q^N_k] + \varepsilon/12$,
$\tilde q_k = q_k / \sum_j q_j$, with $\omega_t = \omega_{E,i}$ after the expert and $\omega_{N,i}$ after the
novice; $\text{logit}\,\omega_{E,i} = \mu_E + \tau_E z_{E,i}$, same for $N$.

**Switch-by-round.** $\text{logit}\,\omega_{N,i,r} = \mu_N + \tau_N z_{N,i} + \delta_N x_r$, $x_r = -\tfrac12$ in
round 1 and $+\tfrac12$ in round 2, so $\delta_N = \text{logit}\,\omega_N(2) - \text{logit}\,\omega_N(1)$ is the
S4 estimand $\Delta$; the same for $E$ with $\delta_E$ (reported). No per-participant round effect.

**Exploratory variants.** Switch-8: eight group-level logit weights, one per last demonstrator × round ×
condition, same participant hierarchy as Switch. Perseveration: HOS plus one group-level bonus on any
arm already chosen in the current play game.

## Findings
- 1 Nov: repo cleanup before publication. `preregistration2.qmd` renamed to `preregistration.qmd` (old version kept untracked as `preregistration_v1.*`); its HTML and PDF are tracked. `preregistration_text.md`, this plan and all of `.claude/`, `notebooks/` and `reports/` outside the preregistration are gitignored, and the old `preregistration.qmd`/`preregistration_text.md` history was erased with git filter-repo (main rewritten and force-pushed, 405 → 264 commits).

- 24 Oct: language is a Prolific prescreen (first language English; fluency at minimum), not a `preprocess_participant_data.R` exclusion; pilot E had 3/40 non-English first language. Data-sharing note for the registration: the shared trial file is the anonymised one minus timestamps; the demographics file holds only status, time taken, colour vision, countries, student and employment status.
- 25 Oct: the registered own-experience model is the carry version with two weights (λ_w current play game, λ_c earlier play game of the round). The reset rung (λ_c fixed at 0) is still fitted and may be reported, but it is not mentioned in the registration; reporting more than registered is fine.
- 25 Oct: Con-2 and Con-5 (legacy contrast table) are not registered; the condition breakdown of the round effect and the unseen-expert section cover the same question.
- 26 Oct: lapse ε = 0.02 fixed in every registered model (pilot E: free group-level ε = 0.012 [0.000, 0.028], other parameters unchanged, no ELPD gain). Registered robustness fit: H+Both and Switch-by-round with ε free at group level; report κ and δ_N unchanged. Per-participant ε is not fitted (confounded with κ_i).
- 26 Oct (user): S5 added as a confirmatory secondary, two rules both required: S5a H+Social beats H+Own, S5b H+Both beats H+Own (ΔELPD > 2 paired SE each). Social vs Own joins the exploratory list with the expectation 'positive'. Pilot E (legacy fits): H+Both − H+Own +284 (SE 42), H+Social − H+Own +269 (SE 42), H+Both − H+Social +15 (SE 11), Both − Own +985 (SE 131). Reason: 'social learning predicts better than none' is the study's basic outcome and S2/S3 presuppose it.
- 26 Oct (user): split-half generative check added to the exploratory list (the in-sample free-running PPC stays the registered check).
- 26 Oct (user): recovery and PPC criteria aligned with 95% intervals: recovery passes at ≥ 22 of 25 datasets covered (95% intervals) with the sign recovered; PPC pass = human value inside the 95% band. The 90% figures in meta-plan 3.4/3.5 are superseded. Chunk 4.3 no longer waits for stage 6 to run: the criteria are stated in advance; only the freeze (chunk 6) waits for the code.
- 26 Oct: dependency table updated: rebuild stages 2 and 4 are done, so only the freeze (6) waits for the rebuild; branch-and-commit rule added (one branch, two sessions, staging by path).
- 26 Oct (user): title "Does inferred demonstrator expertise transfer to a new environment? Learning from demonstrations in multi-armed bandits"; authors Julio Martinez, Michael C. Frank, Nick Haber, all Stanford University (affiliation lines assumed, to be confirmed). "Whom to follow" was dropped from the framing: the object of learning is the environment, demonstrator expertise is inferred, and the transfer is of that inference; the paper plan's claim sentence still says "learn whom to follow" and should be revised with the paper title.
- 26 Oct: 0.1 came to 75 lines (53 non-blank) against the ~40 estimate: the OSF form has 16 fields, each with a heading, a chunk-pointer comment and a marker. Kept as one chunk: a headings-only skeleton has no coherent half.
- 26 Oct (user): ORCID 0000-0003-2092-422X on the first author; affiliations stay university-only. Links registered: live task https://asteroidadventure.com/ (2.1), experiment repo https://github.com/martinezjulio/AsteroidMiningAdenture at a tagged commit (2.1, hash in 6.1), analysis repo https://github.com/martinezjulio/SocialTransformer at the release tag (6.1, 6.3). Every code link points at a tag or commit, never a branch.
- 26 Oct: 0.2: no TeX on this aarch64 box and TinyTeX ships no aarch64 binaries, so `tectonic` (conda-forge, in the SocialTransformer env) is the PDF engine, set in the qmd YAML; it fetches LaTeX packages on first use. Render: 2 pages, 21 headings, 16 markers. Appendices render as numbered sections 7 and 8 for now; proper appendix lettering is a 6.2 cosmetic. `preregistration.pdf` is left untracked until 6.1 decides whether the registered PDF is committed at the tag.
- 26 Oct (user): the document follows the Psych 251 replication-report template (Quarto HTML with margin TOC, folded code, project-links callout, AI use statement first, narrative Methods; MIT) minus its replication-specific parts and guidance boxes, with PDF kept for OSF. No code chunks: the registration is static text and tables; the power table is typed from `power/*.csv` and checked against them at 3.4 (static is cleaner for a document rendered once and frozen). 0.3 struck: instead of copied inputs, the OSF attachments include the anonymised preprocessed trial file (timestamps dropped; the games, descriptive and partner-choice stages do not use them) from which the tagged scripts regenerate the whole sample-size chain, plus the two power outputs. Contact email juliomz@stanford.edu on the first author only. AI statement wording agreed: Claude Code helped set up the document and the pipeline; the text is the author's own.
- 26 Oct: 0.4.2 came to 65 body lines with the mapping table still to write, so the table is split out as 0.4.3. AI use statement and the appendices are unnumbered so the PDF numbering starts at the claims table.
- 27 Oct (user): S2 and S3 (old labels; S1 and S2 after the renumbering below) widened so every rival of the expert readout sits under a rule, S5-style (all parts required, per-part verdicts reported). Whose belief: recent, novice. A readout, not counts: last-visible, expert-visible, naive (a count, so grouped with the visible models). Pilot E (new pipeline, H+Both minus rival): recent +89 (SE 20), novice +247 (41), last-visible +53 (42), expert-visible +169 (40), naive +200 (49); S3a is the only part under 2 SE. Caveats to state at 1.3: the novice's belief is flat by design, so S2b mostly shows the readout carries information; S3a is the sharp test. Recovery (rebuild 6.4.2) covers the hardest part per claim only; the Pareto-k fallback covers all five rivals.
- 27 Oct (user): the partner choice promoted to H6 (model-free, glmer binomial, powered by simulation like H1–H5; the primaries are now exactly the descriptive claims and the secondaries exactly the policy-model claims), and the secondaries renumbered in story order (identify, ablate, dynamics): old S2 → S1 (whose belief), old S3 → S2 (a readout, not counts), old S5 → S3 (social beats own), old S1 → S4 (the weight rises), old S4 → H6. Lines above the Findings use the new labels; findings dated before today keep the old ones. Code labels (registry roles, choice-stage verdicts, recovery labels, the power output's "S4") move in rebuild 5.21.
- 27 Oct (user): equations convention (see the protocol): central models and estimands are numbered display equations, software calls stay inline. Applied to 1.2: @eq-descriptive (cell-means lmer), @eq-contrast (weight vector), @eq-partner (H6 glmer); the claims table's Model column points at them.
- 27 Oct (user): the S3 decomposition identity dropped from the document (trivially true, no rule depends on it); one sentence says S3b − S3a is H+Both vs H+Social, reported without a rule. It stays a row of the paper's supplementary ladder table.
- 27 Oct (user): vocabulary: 'policy model' replaces 'choice model' throughout the document and the plans (a policy: a rule mapping what the participant has seen so far to a probability over the arms; matches `policies.py` and the Thompson policy; pairs with 'descriptive model'). The pipeline's `choice` stage directory keeps its name. 'Partner choice' is unchanged.
- 27 Oct (user): 1.4 edits: E2 reworded as H5's comparison with the other social cells; E4 reworded (novice-first round 2 after-novice game is on a new bandit before the expert is seen, so improvement is carried over); solo slots 1 and 3 dropped from the exploratory list (a figure of cell means, not an analysis) and moved to a reporting sentence in 4.1; the list is E1–E13.
- 27 Oct (user): per-participant ΔELPD plots and the split-half generative check left the exploratory list (a figure of computed quantities; a model-adequacy check) for reporting sentences in 4.3 and 4.5; the list is E1–E11.
- 27 Oct (user): 2.1 facts: cover story Fortune Frontier / starship Prospector; payoff set {0, .1, .2, .3, .35, .4, .45, .5, .55, .6, .75, .8} permuted per bandit, asteroid images and zone backgrounds drawn per bandit; partners unlabelled, new name and image per session; instructions say some partners are much better and consistent across zones; the choice game ends the session with nothing after it; bonus 2 cents per play-game point, rolling across sessions, nothing for watch or choice games; demonstrator filter: expert's summed true payoffs ≥ novice's on the bandit, ~2% of draws, ≤ 2 draws in pilot E; practice session = play, watch, play, not used for exclusion; the deployed URL may become versioned (6.1 TODO). Choice-game prompt (user, 27 Oct): "Imagine you are about to play again in a new zone and could watch only one of the partners from this expedition shoot in that zone first. Which partner would you choose to help you play better?", partner images with names beneath, select one.
- 27 Oct: 2.2 written from the pilot E data: session order random per participant (all six orders occurred, 3–9 each), sessions indexed 1–3 with the practice session absent from the preprocessed file, a new partner pair per social session (two demonstrator ids per social session), a demographics questionnaire at the end (anonymised demographics file exists). Not confirmed by the user: that the end questionnaire is demographics only, and the practice partner's type.
- 27 Oct (user): 3.1 facts: Prolific; US residents; ages 18–50; fluent in English; colour-vision prescreen (from MP 5, plus the in-task item); block list of all pilot and related-experiment participants; base $2 for an estimated 10 min ($12/h) plus the 2-cent bonus. Assumed, not confirmed: no approval-rate prescreen. Flag for the user: pilot E median completion was 12.1 min (IQR 10.3–14.5), so the $2 base is ~$10/h before bonus; either raise the base or restate the estimate. The consent-text data-sharing check stays a `[TODO: user]` in the document (experiment repo not on this machine).
- 27 Oct (user): 3.1 decisions: base $2 stated as assuming 10 min at $12/h, pilot median kept beside it; prescreens add approval ≥ 95% and ≥ 10 previous submissions; the block list is a Prolific participant group; decline of consent → Prolific 'Request a return' code, unpaid, not counted; the consent page has a separate data-sharing opt-in (share with researchers / make public) that can be declined while taking part, recorded by the game, so the public dataset is the subset who agreed. Pilot E participants gave no sharing consent: the pilot trial file leaves the OSF attachments (6.3 lists the aggregate outputs instead); the IRB protocol permits internal use for the power analysis; the document says the chain is checkable from the attached tables but not re-runnable. Consent text to be pasted at 6.2.
- 27 Oct (user): exclusions revised: X1 is 'returned on Prolific' only; a Prolific time-out is not a rule (timed-out finishers kept if they pass every rule, paid by bonus); new X3b, logged span over 60 min (mirror of the 300 s floor). Evidence: the slowest pilot E completer took 47 min on Prolific's clock but 14 min of logged main-session play with ordinary pacing and performance; the one time-out logged two sessions over 113 min. Script change (split the status filter, add the ceiling, ledger rows 'returned' and 'span_ceiling') proposed to the rebuild plan as 2.x for the other session; pilot E counts unchanged (8 returned, the time-out falls under X3b and X10).
- 27 Oct (user): 3.4 written in full as one chunk (62 lines: method, both decision-rate tables, closed-form table with @eq-closed, cost table from the hand-made `cost/N_cost.csv` with the pilot mean bonus $1.06 added, S4 √(38/N) projection done by hand from the 5.15 interval). Left unticked: the user shows the tables to the advisor, then chooses N (recommendation 300; budget note says 500) and the section is trimmed (cost table out). Not script-generated: the S4 projection and the cost table.
- 28 Oct (user): 5.1 widened: the appendix carries the emmeans contrast code for H1–H5, not only the weight vectors. Each contrast gets a definition column (cell-selection rule, as `descriptive.R` builds it) and the generic grid and `contrast()` calls appear once; the vectors are the `w` argument. Keeps the equations convention (software calls inline) while making every contrast auditable and reproducible from the document.
- 28 Oct (advisor meeting, via the user): N = 350 analysed participants (power 1.00 for every remaining test at the pilot effect; at half the effect H2 payoff 0.87, everything else ≥ 0.99; cost about $997 base plus about $396 bonus for 374 paid). Reward demoted to a reported outcome (rationale in the registered section above). The former H4 made exploratory on all outcomes; the partner choice moved before the solo comparison, so the primaries are H1 after the expert > after the novice, H2 after-novice improvement, H3 entropy fall, H4 partner choice, H5 after-expert > solo final play. Claims to be restated in plain language (1.5.3, wording to be brainstormed). Introduction kept short. OSF gets the PDF and aggregate CSVs only; repos and checkpoint linked by tag and hash. The exploratory list to say in bold that it is not a commitment. Title unchanged. Consequences for the other plans, not yet applied: `descriptive.R` and `power_analysis.R` roles and labels (reward and the former H4 exploratory, H6 → H4), rebuild 5.21's registry, and the H labels in `paper.md` and `meta-plan.md`.
- 28 Oct (user): no budget bound; N = 350 survivors is the only stopping condition, so the budget clause left 3.3 and 3.4. 3.4 trimmed: cost table out, reward and former-H4 rows out of the three power tables, partner choice relabelled H4, S4 projection at 350 (SD ≈ 0.19). The decision rates were simulated with Holm over three outcomes, so they are slightly conservative for the registered two-outcome rule; the document says so. When the rebuild reruns `power_analysis.R` with the new roles, the tables are refreshed from its output.
- 28 Oct (user): 1.5.3 done as option A, both columns rewritten in place: Claim a plain sentence per row (user's H4 wording), Estimand the measured quantity in words. H5's claim corrected from "the same amount of own experience" to "even the solo condition's most practised game" (the after-expert game has 6 or 18 revealed shots by condition, solo's final game 18, so solo is never less informed); the narrative clause in 1.2 fixed the same way. Claims-table column widths retuned (Claim 14, Estimand 19, Model 8, Rule 8, Role 6 dashes); the table runs pages 2–4 of the PDF.
- 28 Oct (user): claims table is the single hypothesis table: the Rule column now carries the predicted sign per outcome (payoff +, entropy −, β₀ > 0) and names the Holm scope; the cells-compared table of 1.2 deleted as a duplicate (its content lives in the Estimand column and, exactly, in the Appendix A weight vectors); "as H1" references replaced by the full rule in every row.
- 28 Oct: 4.3 written from the code (fitting.py gate and ladder, compare.py PSIS-LOO and the k > 0.7 session refits, checks.py recovery constants: 25 datasets × 100 participants, cover ≥ 22, model recovery S1a/S2a/S4 with 5 datasets per side). Mismatch to flag to the rebuild session: `run_analysis.py` verdicts give only supported / not supported / not evaluable; the registration also has *contradicted* (rival ahead by > 2 SE) per part and per claim, and the claim-level 'not evaluable' must not override a contradicted part. The document states the registered rule; the code should follow it.
- 28 Oct: 5.1 written as Appendix A.1 (priors, with the perseveration bonus N(0, 2) added from the code) and A.2 (contrast definitions as cell-selection rules, two weight-vector tables copied from `descriptive/contrast_weights.csv`, the three calls). The script's `H5_round2` contrast had no entry in the exploratory list; E2 now names its three parts E2a (after-novice vs solo final), E2b (social average vs solo final), E2c (H5 within round 2). `solo_round_change` is listed as E3s, the solo term of E3.
- 29 Oct (review): the document was read against the committed pipeline and the pilot E outputs. Stage 7 added: model names (the registry moved to the HOS convention in 450564e), the power tables (regenerated 28 Oct 13:42 with Holm over two outcomes), pilot anchors (S4, S2a, the unseen-expert numbers are from the legacy analysis and differ from the pipeline's), X2 (reads Prolific's export column; the game asks no colour-vision question, user), X4's pilot count, wording of the reward versions, λ_c interpolation, recovery truth (the analysed version's own fit, user) and pass rules, the exploit statistic. Code gaps (Pareto-k fallback unwired, check-gated verdicts, E7/E8/E10 numbers) went to rebuild stage 10; 7.6 waits for it.
- 29 Oct (user): no questionnaire exists; demographics from Prolific's export via preprocessing (rebuild 10.9); consent is single and includes public sharing, so the public release is the preprocessed file minus timestamps (rebuild 10.10) plus the ledgers and aggregates; 7.2's precondition verified; 7.4–7.7 widened after the post-stage-10 review; both plans on main.

- 29 Oct (user): the full run is deployed at https://asteroid-mining-game-31128277324.us-east4.run.app/ (6.1.1). The consent page's text was reviewed against the document: it is a single consent whose terms include de-identified public sharing; two sentences were revised so the text matches what is collected (the Prolific-provided age, sex and colour-vision fields, reported only in summary; no name, contact details or other identifying information) and the user is updating the game to the exact wording quoted in 3.1 (6.2.1). 6.1 and 6.2 nested: the tag, hashes and the final no-TODO pass wait for the tag.

- [x] 8. Restructure to remove redundancy (2 Nov; plan and findings in `preregistration-restructure.md`): Key Claims table becomes section 2 Hypotheses after Task and design; Hypotheses prose merged into section 3 Models and analysis; Exploratory list with the unseen-expert sub-table as section 4; Sampling plan as section 5; Variables, Results and Appendix A removed (priors table moved into Policy models, E contrasts into the exploratory section). Supersedes the section list of 0.4.2.
