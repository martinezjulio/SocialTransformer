#!/bin/bash
# Run on the lab machine: pulls one version's analysis outputs from Sherlock into this machine's $OUTPUT_DATA_DIR/analysis/<version>/
# through the data transfer node. Reads the lab .env for OUTPUT_DATA_DIR.
#   bash scripts/fetch_results.sh --version 2026_11_05_full_study [--sunet juliomz] [--fits ROS,Switch-by-round]
# The two rsyncs share one SSH connection (ControlMaster, kept for 10 min), so Duo asks once. The script ends with a
# '== done' line listing the fit pickles present; if that line is missing, an rsync failed and its error is just above.
#
# What the plain fetch brings (about 7 MB for the full study), enough to remake every figure and table that is drawn from
# summaries or trial-level data:
#   paper/            the report: results.tex (one macro per number), results.md, tables/*.tex, figures/*.{pdf,png}
#   games/            play_games.csv, one row per play game (participant, session, condition, round, slot, play, demonstrator,
#                     payoff, reward, entropy): the source of Fig 2a and of any per-game descriptive plot
#   descriptive/      cell_means.csv (lmer cell estimates), contrasts.csv (every hypothesis and exploratory contrast with CI,
#                     p, Holm p, verdict), contrast_weights.csv, robustness.csv, robustness_variance.csv: Fig 2b, Fig S robustness
#   partner_choice/   choices.csv (one row per social session), rates.csv, model.csv: Fig 2c
#   choice/           ladder.csv (ELPD, ΔELPD vs ROS, SE, Pareto-k counts, sampler diagnostics per model), ladder_by_round.csv,
#                     elpd_per_participant.csv (the per-participant differences behind every paired SE), verdicts.csv (S1-S4
#                     with the posterior summaries of the exploratory parameters): Fig 3a-c, Fig S ladder and per-participant dots
#   checks/           recovery.csv (δ_N recovery per dataset), model_recovery.csv and model_recovery_summary.csv, ppc.csv (human
#                     value, 95% band and inside flag per target), the per-draw JSONs under ppc/ and split_half/: Fig S PPC, recovery
#   unseen/           readout_table.csv, copying.csv, elpd_split.csv, exploit.csv: Fig 4b, 4c, the supplementary split table
#   */numbers.json    every number each stage emitted (the report merges these)
#
# What it leaves on Sherlock's scratch (about 134 GB), and what cannot be made without it:
#   tables/           the belief tables (18 GB): only needed to refit or re-score a policy model
#   choice/fits/      the fit pickles (~300 MB each; 1.3 GB for Switch-8 and Switch-by-round+ktrend, which needed the ladder's last
#                     rung): the posterior draws and pointwise log-likelihoods. --fits pulls only the named models' final pickles
#                     (not the cached rungs, not the +eps/+ktrend variants that share a name prefix). Needed for any panel or
#                     new figure drawn from draws: Fig 3d (ω after the expert and after the novice by round, from Switch-by-round),
#                     Fig 4a (the eight weights, from Switch-8), the supplementary posterior figure (ROS, Switch-by-round, ROS+eps,
#                     Switch-by-round+eps), densities or new summaries of any parameter. The report's numbers.json hold means and
#                     95% intervals of the cited parameters only. Fetch what is needed with --fits <name,...> (names as in the
#                     ladder: ROS, recent, novice, naive-recent, naive-expert, naive, RS, RO, R, O, S, OS, Random, Switch,
#                     Switch-by-round, Switch-by-round+ktrend, Switch-8, ROS+persev, ROS-reset, ROS+eps, Switch-by-round+eps).
#   choice/loso/      the Pareto-k fallback refits (pointwise ELPD after refitting flagged sessions); their effect is already in
#                     ladder.csv and verdicts.csv
#   checks/recovery/, checks/model_recovery/   the simulated datasets and their fits (109 GB); summarised in the CSVs above
#   slurm/            unit lists and job logs
#
# Re-running figure code on the lab machine against the fetched folder works for everything in the first list; a figure that
# calls load_fit(<model>) needs that model's pickle fetched first.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; source "${ENV_FILE:-.env}"; set +a
VERSION=""; SUNET="${SUNET:-juliomz}"; FITS=""
while [ $# -gt 0 ]; do case "$1" in --version) VERSION="$2"; shift 2 ;; --sunet) SUNET="$2"; shift 2 ;; --fits) FITS="$2"; shift 2 ;; *) echo "unknown argument: $1" >&2; exit 2 ;; esac; done
[ -n "$VERSION" ] || { echo "--version is required (e.g. 2026_11_05_full_study)" >&2; exit 2; }
SRC="$SUNET@dtn.sherlock.stanford.edu:/scratch/users/$SUNET/SocialBanditsProject/output/analysis/$VERSION"
DST="$OUTPUT_DATA_DIR/analysis/$VERSION"
mkdir -p "$DST" ~/.ssh
SSH="ssh -o ControlMaster=auto -o ControlPath=$HOME/.ssh/fetch-%r@%h:%p -o ControlPersist=600"
# Excludes are anchored to the version folder, so paper/tables/ (hypotheses.tex, ladder.tex) is kept while the belief tables,
# the fit pickles, the fallback refits, the simulated recovery datasets and the Slurm logs stay on scratch.
rsync -avP -e "$SSH" --exclude='/tables/' --exclude='/slurm/' --exclude='/choice/fits/' --exclude='/choice/loso/' \
  --exclude='/checks/*/fits/' --exclude='/checks/*/tables/' --exclude='*.pkl' --exclude='*.block*.npy' "$SRC/" "$DST/"
if [ -n "$FITS" ]; then
  mkdir -p "$DST/choice/fits"; INC=()
  for m in ${FITS//,/ }; do stem=$(python3 -c "import re,sys; print(re.sub(r'[^\w.-]', '_', sys.argv[1]))" "$m"); INC+=(--include="${stem}_????????????????-????????????????.pkl"); done
  rsync -avP -e "$SSH" --exclude='*_rung*.pkl' "${INC[@]}" --exclude='*' "$SRC/choice/fits/" "$DST/choice/fits/"
fi
$SSH -O exit "$SUNET@dtn.sherlock.stanford.edu" 2>/dev/null || true
echo "== done: $(find "$DST" -type f | wc -l) files, $(du -sh "$DST" | cut -f1); fit pickles: $(ls "$DST/choice/fits/" 2>/dev/null | grep -c '\.pkl$')"
