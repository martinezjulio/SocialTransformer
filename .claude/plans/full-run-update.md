# Full-run update: what changes between the pilot chain and the full-data submission

Written 4 Nov 2026 while the pilot chain (jobs 51143898-51143909) runs; companion to `sherlock-setup.md` (its item 7).
Purpose: a reviewable list of every setting, check and step for the first full-data run on Sherlock, maintained as the
pilot run teaches us more. Review it together before the submission; tick items as they are done. Same protocol as the
other plans: one item at a time, verify, tick, report, commit on approval. Timeline assumed: study Friday 7 Nov,
submission Friday night, results wanted by end of Monday 10 Nov.

## A. Settings to change before the full submission

- [x] A.1 (4 Nov: `SLURM_FINAL_TIME` added to `submit.sh` and `.env.example`; 24:00:00 set in the Sherlock `.env`; applied to the pilot chain as a test of the configuration) Final run time limit: `submit.sh` hard-codes `--time=12:00:00` for the final run; the full run needs 24 h (`hns` and
  `normal` allow 7 d). Make it an `.env` variable (`SLURM_FINAL_TIME`, default 12:00:00, documented in `.env.example`) and set
  24:00:00 in the Sherlock `.env`. Set from the pilot's measured final-run time × ~8 (the choice stage was estimated at ~7 h
  at 300 participants on the lab box).
- [x] A.2 (4 Nov: `N_TABLE_BLOCKS=32`, `SLURM_TABLE_CPUS=4`, `SLURM_TABLE_MEM=16G` in the Sherlock `.env`, the two new variables read by `submit.sh`; dry run gives 288 level-0 units; applied to the pilot chain) Table blocks: `N_TABLE_BLOCKS=32` in `.env` (288 level-0 tasks instead of 72). Each block then takes ~25 min at 350
  participants instead of ~1.5 h, fits scheduler gaps better and loses less to a preemption. Check that `--list-units`
  gives 288 level-0 units and that level 0's `--cpus-per-task=16` still suits a 32-block task (the builder is threaded;
  consider 8 cores and halve the memory if the pilot's `seff` shows headroom).
- [x] A.3 (4 Nov: pilot 8-block tasks on 16 cores peaked at 5.8-6.2 GB (`MaxRSS`), cached ones 2.8 GB, 15-26 min; a 32-block pilot task is a quarter of the rows, so 16 GB is ample; at 350 participants a 32-block task holds ~2.3× the rows of a pilot 8-block task, projected peak ~10-14 GB: set `SLURM_TABLE_MEM=24G` for the full run and re-check with `seff` on the pilot's 32-block tasks) Level-0 memory: read the pilot's peak (`seff <jobid>_<task>` on a few completed block tasks, `MaxRSS` in
  `sacct -o JobID,MaxRSS`) and scale by rows per block at 350 participants (a 32-block task holds about the same rows as a
  pilot 8-block task: 350/38 × 1/4 ≈ 2.3×). Keep `--mem` ≥ 1.5× the projected peak.
- [ ] A.3b (4 Nov 22:20, pilot level 3): a recovery dataset unit ran 16 min with `MaxRSS` 21.9 GB against the 32 GB GPU-unit request; loso refits up to 10 GB; the simulated datasets are the same size in the full run, so raise the GPU levels' `--mem` to 48G (12 GB/core on `owners`, within its 48 GB/core cap) before the full submission; make it an `.env` variable (`SLURM_GPU_MEM`, default 32G) like the table settings.
- [ ] A.4a (4 Nov: `SLURM_GPU_OPTS=--gpu_cmode=shared` is required on Sherlock for every GPU level; without it the fit worker cannot open a CUDA context; set in `.env` by the setup script) GPU compute mode.
- [ ] A.4 GPU unit time limit (2 h): confirm from the pilot that the slowest level-3/4 unit (recovery dataset, loso refit,
  split fit) finished well inside 2 h; fits at 350 are expected at ~35 min. Raise to 4 h only if the pilot's slowest unit
  exceeded 1 h (the simulated datasets do not grow with N; the fits and loso refits do).
- [x] A.5 (4 Nov: `mcfrank` AllowGroups includes mcfrank, 24 cores, 192 GB, 7 d, 8 GB/core; `SLURM_CPU_PARTITION=owners,normal,mcfrank` in `.env`; applied to the pilot chain) Partitions: CPU levels `owners,normal,mcfrank` (the group's own 24-core node, 7 d, no campus competition); GPU
  levels stay `owners` (the `gpu` partition's QOS refuses arrays of 130+ tasks); final `hns,normal`. Check `mcfrank`
  eligibility first: `scontrol show partition mcfrank` (AllowGroups) and a 1-minute test job.
- [ ] A.6 `sh_quota` before submitting: tables at 350 ≈ 9 × 3.5 GB plus the recovery datasets' tables; scratch has room (100 TB),
  group home (55% used) is not touched by the run.
- [ ] A.7 Lab clone on `main`, pulled, with the Sherlock `.env` values mirrored where they matter (`N_TABLE_BLOCKS` only affects the cluster).

## B. Thursday review of the pilot chain (feeds A)

- [ ] B.1 (interim, 4 Nov 20:30: level 0 submitted 19:19, 288 tasks done by ~20:20, 13-23 min each, peak 45-104 concurrent, 0 failed, a few preemptions requeued; level 1's 9 assemblies done within minutes of level 0; level 2 pending on `owners` at 20:30; third submission: cached levels 0-1 passed through in ~35 min of scheduler cycling (288 + 9 one-minute tasks, a cost of resubmitting worth knowing), level 2 started 21:32 and had 17 of 21 fits done by 21:41, pilot fits 20-35 s of sampling plus ~1 min of start-up, R-hat ≤ 1.005, ESS ≥ 1,700, 0 divergences) Per-level wall time and the slowest unit per level: `sacct -j <jobid> -o JobID,State,Elapsed,MaxRSS,ExitCode -P`
  for each of the six jobs; record a table here (level, tasks, median, max, failures, preemptions).
- [ ] B.2 Preemptions: count of `PREEMPTION` lines in `level*-*.out`; how much work they cost (elapsed before the kill).
- [ ] B.3 Any task that failed and why; the fix, committed, before Friday.
- [ ] B.4 Results against the lab table in `sherlock-setup.md` (6.1): `paper/results.md`, `verdicts.csv`, the check row counts;
  record differences there, not here.
- [ ] B.5 `paper/` copied to `$GROUP_HOME/$USER/SocialBanditsProject/results/pilot_E_<date>` (6.2); the set-aside folders
  `analysis/pilot_E.old-2026-11-04` and `pilot_E.prerun-2026-11-04` deleted by the user.

## C. Friday: from the last participant to the submission (lab machine, then Sherlock)

- [ ] C.1 Lab machine: `anonymize_participant_data.py` (uncommitted) on the raw logs and the Prolific export; then
  `src/r/preprocess_participant_data.R` → `$EXPERIMENT_DATA_DIR/<full>/{preprocessed_participant_data.csv,exclusions.csv}`;
  check the exclusion counts and that exactly 350 participants with three sessions survive (the stopping rule).
- [ ] C.2 Lab machine: `bash scripts/run_inference.sh` for `<full>` on the lab GPU (checkpoint `epoch_119_best.pt`, unchanged);
  runtime not yet measured at 350 (pilot: minutes). Note `eval/<full>/metadata/inference_config.yaml` records lab paths; the
  pipeline ignores them (9.13).
- [ ] C.3 Transfer through the DTN (README §6 lines with `<full>` for `pilot_E`; `mkdir -p` the two nested targets first);
  only the two anonymised files from `experiments/<full>/`, never the folder.
- [ ] C.4 (the pilot's run of this configuration: jobs 51150993-51150998, 4 Nov) On Sherlock, dev session: checksums of the two files and the checkpoint against the lab machine's `sha256sum`;
  `bash scripts/run_analysis.sh --version <full> --stages games,inference-check,descriptive,partner_choice` (minutes) as the
  smoke test; `bash scripts/slurm/submit.sh --version <full> --dry-run` and the unit counts; then the real submission.
  Record the job ids here. Do not edit the repo while the chain runs.

## D. Weekend monitoring and the rule for failures

- [ ] D.1 Saturday morning and Sunday morning, from a dev session (`claude --continue`, "check the run"): `squeue --me`,
  `sacct` per job, grep the logs for `Traceback` and `PREEMPTION`, blocks and fits counted on disk. Checks at least a
  minute apart, no loops.
- [ ] D.2 On a failed task: read its log, fix, commit; cancel the whole remaining chain (`scancel` the dependents and the
  pending tasks, let running tasks finish) before resubmitting, so no two tasks ever compute the same unit at once
  (`data.atomic_write` uses a deterministic temp name). Cached units then exit in seconds.
- [ ] D.3 Monday: `final-*.out` ends with the report line; `paper/` copied to group home (scratch purges after 90 days);
  `git status` clean, every fix pushed.

## E. Known residual risks after a clean pilot (why B and D exist)

Time limits (A.1, A.4), memory at 350 (A.3), longer tasks more exposed to preemption on `owners` (A.2), data-dependent
paths that the pilot never exercised (Pareto-k fallback refits on more sessions, recovery pairs chosen from the full
ladder, manifest keys triggered only by particular values), and scheduler quirks on a different node mix (like the
shared-cache race of 4 Nov). None is expected to fail; each is checked or bounded by an item above.

## Findings

- 4 Nov 19:47, throughput in the full-run configuration: 104 table tasks running at once (102 on `owners`, 1 on `normal`) against 2-3 with 16-core tasks; 40 of 288 blocks in the first 25 min; per task 13-23 min on 4 cores, `MaxRSS` 5.5-6.7 GB, so a 32-block task uses about as much memory as an 8-block one (fixed overhead dominates: 2.8 GB even for a cached task) and 16 GB holds; projection for 350 participants stays ~10-14 GB, `SLURM_TABLE_MEM=24G`. Three preemptions in 25 min, requeued. Level 0 ETA under an hour instead of overnight.
- 4 Nov: the pilot chain is being rerun in the full-run configuration (A.1, A.2, A.5 applied) so that configuration is what the pilot tests; the five 8-block files from the first chain (`*.block00Xof008.npy`) stay in `tables/` unused (block files are keyed by their split) and can be deleted with the set-aside folders.
