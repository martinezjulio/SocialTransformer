# Plan: restructure preregistration.qmd to remove redundancy

File: `reports/preregistration/preregistration.qmd` (the only document edited; `preregistration2.qmd` no longer exists, the user merged it in). After approval this outline is also saved as `.claude/plans/preregistration-restructure.md` in the repo (Workflow rule) and ticked chunk by chunk.

## Context

The document states the same content up to four times: the Key Claims table (claim, estimand, model, rule, contrast call), the Hypotheses section 3.1/3.2 (models, rules, verdict vocabulary in prose), Analysis plan 4.5 (fits, rules, verdicts again) and Appendix A.2 (the contrasts a fourth time). Variables and Results are template remnants. The user's decisions (2 Nov):

- Key Claims table moves after Task and design and becomes section **Hypotheses**; the table itself is not changed.
- Sampling plan (existing data, participants, power, exclusions) goes after Exploratory, as section 5.
- Unseen-expert analyses survive as a five-line sub-table under E11; pilot anchor numbers dropped.
- Priors table (A.1) moves into Policy models; A.2 (definitions, weight vectors, calls) dropped, with E1–E3 `m()` expressions added to the exploratory table so nothing is lost.
- Variables and Results sections dropped (demographics sentence moves to consent; one reporting sentence moves to Frozen artefacts).

Target outline:

```
AI use statement (unnumbered)         Introduction (unnumbered)
1 Task and design: 1.1 Materials, 1.2 Procedure
2 Hypotheses: intro paragraph, the table (untouched), one shared-vocabulary paragraph
3 Models and analysis: 3.1 Descriptive model, 3.2 Partner choice, 3.3 Policy models (incl. priors),
                       3.4 Fitting and model comparison, 3.5 Posterior predictive checks
4 Exploratory analyses: one paragraph, E1–E12 table, unseen-expert sub-table
5 Sampling plan: 5.1 Existing data, 5.2 Participants, 5.3 Power and sample size, 5.4 Exclusions
6 Frozen artefacts, reporting and data availability
```

Expected size: about 9,500 words / 22 pages (from 11,679 / 28). Estimates below are qmd lines; stage 1 is a verbatim block move and is flagged as the Increment-Mode exception.

Rules for every chunk: prose that is deleted (not moved) is appended verbatim to `reports/preregistration/preregistration_text.md` under a dated heading; the Key Claims table block must stay byte-identical (checked by diff against the backup); render HTML after each chunk, PDF at the end of each stage.

## Outline

- [x] 0. Backup
  - [x] 0.1 Copy the current qmd to the scratchpad (`preregistration_before_restructure.qmd`) for the identity diffs — shell (0)
- [x] 1. Reorder blocks (moves only; **exception**: ~450 lines move verbatim via one script, no prose edited)
  - [x] 1.1 Python script cuts the file into blocks by heading line and reassembles in the target order: Key Claims after Procedure; Analysis plan subtree after it (Descriptive, Partner choice, Policy models, Fitting, PPC); Exploratory list after that; Planned exploratory beneath it (merged at 4.3); Methods (Existing data, Participants, Power, Exclusions) after; Frozen artefacts; then Variables, Results, Appendix A at the end awaiting deletion. Verify: sorted line multiset identical to the backup — script in scratchpad (0 edited lines)
  - [x] 1.2 Headings and anchors: `# Key Claims {#sec-claims}` → `# Hypotheses {#sec-hypotheses}` (old `# Hypotheses` / `## Confirmatory primary|secondary` headings removed, their bodies kept for stages 2–3); `## Analysis plan` → `# Models and analysis {#sec-analysis}` with its `###` → `##`; `### Fitting and inference` → `## Fitting and model comparison`; `## Exploratory` → `# Exploratory analyses`; `# Methods` → `# Sampling plan`; `# Frozen artefacts and data availability` → `# Frozen artefacts, reporting and data availability`; `@sec-claims` → `@sec-hypotheses` (3 refs) — qmd (~12)
  - [x] 1.3 Render HTML and PDF; confirm no unresolved refs and the table block is byte-identical — shell (0)
- [x] 2. Hypotheses (section 2)
  - [x] 2.1 Intro paragraph: fix the wrong pointer ("shown below in section @sec-design" → the models are in @sec-analysis), keep the `emm` / `m(sel)` / `demo` sentence — qmd (~5)
  - [x] 2.2 Shared-vocabulary paragraph after the table, stated once: verdicts *supported / not supported / contradicted / not evaluable* (gates in @sec-fitting), two-sided α = 0.05 with the predicted sign, Holm within a hypothesis only, 95% intervals for every estimate, no smallest effect of interest or equivalence test, and the one-line roles sentence (H1 manipulation check, H2–H3 learning, H4 discrimination, H5 vs practice; round-change and solo comparisons exploratory) — qmd (~8)
- [x] 3. Models and analysis (section 3)
  - [x] 3.1 Descriptive model: merge old 3.1 "Outcomes" and "Model and rule" (eq-descriptive, eq-contrast) into the subsection in the order Data → Model (the two equations) → Fit → Decision procedure → Reward (no rule, why) → Sensitivity refits; delete the duplicate fit formula, duplicate rule sentence and the Appendix pointer — qmd (~35)
  - [x] 3.2 Partner choice: merge old "H4, the partner choice" (eq-partner) into the subsection: Data → Model (eq-partner) → Fit and rule → Recency check (E12); delete the duplicate `glmer` sentence — qmd (~15)
  - [x] 3.3 Policy models, part 1: fold the old 3.2 opening paragraph (what a policy model is, HOS letters, story order) into "Data and notation" / "The full model"; move eq-switch into "The Switch models" — qmd (~20)
  - [x] 3.4 Policy models, part 2: replace the four S1–S4 prose paragraphs by one "**What each comparison tests.**" paragraph, one or two sentences per claim (S1a the sharp part; S2a copying; S3b − S3a reported without rule; S4 Δ > 0 with the registered caveat and ω_E expectation) placed after the ladder table — qmd (~15 new, ~30 deleted)
  - [x] 3.5 Priors: move A.1 (intro sentence + table) into Policy models as "**Priors.**" after the Switch models; drop the "equations are …" sentence — move (~18, 2 edited)
  - [x] 3.6 Fitting and model comparison: fold old "Comparison rule" (eq-elpd, 2 SE rule, part/claim verdict rule, *not evaluable*) into "Comparison" and "Verdicts"; delete the duplicates — qmd (~20)
  - [x] 3.7 Render HTML and PDF; check every `@eq-` and `@sec-` resolves — shell (0)
- [x] 4. Exploratory analyses (section 4)
  - [x] 4.1 Opening paragraph: keep closed list / no rules / post hoc labelling; drop the pointer to the planned-exploratory section — qmd (~4)
  - [x] 4.2 Table: add the `m()` expressions for E1, E2a, E2b, E2c, E3 and E3s to their rows (from A.2 definitions, in the Key-Claims form); E4 and E12 pointers updated — qmd (~8)
  - [x] 4.3 Unseen-expert sub-table: lead-in sentence (the question, no rule, read with the S4 caveat, scripts) + five rows: (a) readout table, (b) readout vs copying, (c) ELPD split, (d) exploit statistic, (e) Switch-8; delete the old section with its (a)–(e) paragraphs and pilot anchors (backed up) — qmd (~15 new, ~17 deleted)
  - [x] 4.4 Render HTML — shell (0)
- [x] 5. Sampling plan (section 5)
  - [x] 5.1 Consent paragraph: add the demographics sentence from Variables (age and sex from Prolific's export, summarised over the analysed sample, never released per participant); replace `@sec-variables` — qmd (~3)
  - [x] 5.2 Delete the Variables section (backed up) — qmd (−9)
- [x] 6. Frozen artefacts, reporting and data availability (section 6)
  - [x] 6.1 Add a "**Reporting.**" sentence: results come from `results.md` written by the report stage at the tagged commit, no number typed by hand, every deviation from this document listed; delete the Results section and its callout (backed up) — qmd (~4 new, −12)
- [x] 7. Cleanup
  - [x] 7.1 Delete the Appendix A remnant (A.2 definitions table, two weight-vector tables, "The calls") (backed up) — qmd (−75)
  - [x] 7.2 Cross-references: `@sec-primary` → `@sec-hypotheses` or `@sec-descriptive`; `@sec-secondary` → `@sec-fitting`; `[Appendix A](#sec-appendix-models)` → `@sec-policy-models`; `@sec-planned-exploratory` → `@sec-exploratory-list`; chunk-pointer comments updated; grep for dangling anchors — qmd (~12)
  - [x] 7.3 Final render HTML and PDF; `pdfinfo` page count; rasterise the table pages and check the grid table and sub-table lay out — shell (0)
  - [x] 7.4 `.claude/plans/preregistration.md`: add stage 8 (this restructure) as done, strike 1.1 (Introduction now written by the user), note under Findings that 0.4.2's section list is superseded — plan (~15)

## Verification

- After every chunk: `quarto render preregistration.qmd --to html`; grep the output for `WARN` and the html for `?@` (unresolved refs).
- After stages 1, 3 and 7: `--to pdf` in the `SocialTransformer` conda env (tectonic); `pdfinfo` pages.
- Identity checks: after 1.1, `sort` of both files identical; after every stage, `diff` of the Key Claims table block against the scratchpad backup is empty.
- Word count by section (`awk` on headings) reported at the end against the 11,679 baseline.
- No commits; `preregistration_v1.*` and `preregistration_text.md` untouched except the appended backups.

## Findings

- 2 Nov: executed in one pass at the user's request (Increment Mode off), not chunk by chunk. All chunks done; nothing committed.
- 4.2: the `m()` expressions did not fit in the exploratory table's "What is reported" column (the code wrapped badly in the PDF), so they sit in a separate two-column table "Contrasts of E1 to E3" under the main table; the E rows keep their prose, E3's row now names E3s.
- 2.1: the user's own intro paragraph is kept with its typos fixed and the pointer corrected; the sentence defining `emm`, `m(sel)` and `demo` was restored because the table uses them.
- 3.4: the pilot E number quoted beside S2a (HOS ahead of naive-recent by 160, SE 35) was dropped with the other pilot anchors.
- Result: 10,051 words, 25 pages (from 11,679 words, 28 pages); the plan's 9,500 / 22 estimate was optimistic. Removed prose is appended to `reports/preregistration/preregistration_text.md`.
- 2 Nov, later: the claims grid table was replaced by an itemised list (one bullet per claim with Estimand, Model, Rule and Contrast sub-bullets, text verbatim); Materials trimmed to registration material; Design cells paragraph cut to Outcomes with the solo-counterpart clause moved into the Descriptive model Data paragraph. 9,240 words, 23 pages.
