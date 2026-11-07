#!/bin/bash
# Status of the latest submission of one version (scripts/slurm/submit.sh records its jobs in analysis/<version>/slurm/jobs.tsv):
# per job, the task states from sacct; the running tasks with their nodes; tracebacks and preemptions in that submission's logs;
# with --missing, the units whose cache piece is still absent (runs the driver inside the image, ~20 s).
#   bash scripts/slurm/status.sh --version pilot_E [--missing]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
set -a; source "${ENV_FILE:-.env}"; set +a
VERSION=""; MISSING=0
while [ $# -gt 0 ]; do case "$1" in --version) VERSION="$2"; shift 2 ;; --missing) MISSING=1; shift ;; *) echo "unknown argument: $1" >&2; exit 2 ;; esac; done
[ -n "$VERSION" ] || { echo "--version is required" >&2; exit 2; }
DIR="$OUTPUT_DATA_DIR/analysis/$VERSION/slurm"
[ -f "$DIR/jobs.tsv" ] || { echo "no submission recorded in $DIR/jobs.tsv"; exit 0; }
STAMP=$(tail -1 "$DIR/jobs.tsv" | cut -f1)
echo "submission $STAMP of $VERSION ($(date +%H:%M))"
awk -F'\t' -v s="$STAMP" '$1 == s' "$DIR/jobs.tsv" | while IFS=$'\t' read -r _ level job n; do
  states=$(sacct -j "$job" -n -X -o State -P 2>/dev/null | sed 's/ by .*//' | sort | uniq -c | awk '{printf "%s %s  ", $2, $1}')
  printf "  %-6s job %-9s %4s tasks: %s\n" "$level" "$job" "$n" "${states:-not in sacct yet}"
  if [ "$level" = final ]; then logs=("$DIR/final-$job.out"); else logs=("$DIR/level$level-$job"_*.out); fi
  [ -e "${logs[0]}" ] || continue
  tb=$( (grep -l "Traceback" "${logs[@]}" 2>/dev/null || true) | wc -l); pre=$( (grep -l "PREEMPTION" "${logs[@]}" 2>/dev/null || true) | wc -l)   # grep's "no match" must not trip set -e/pipefail
  [ "$tb" -gt 0 ] && echo "    tracebacks in $tb log(s): $(grep -l Traceback "${logs[@]}" | head -3 | xargs -n1 basename | tr '\n' ' ')" || true
  [ "$pre" -gt 0 ] && echo "    preempted and requeued: $pre" || true
  [ "$level" = final ] && { grep -E "^== |^report:" "${logs[0]}" || true; } | tail -2 | sed 's/^/    /'
done
running=$(squeue --me -h -o "%i %P %M %N" 2>/dev/null | grep -E "^($(awk -F'\t' -v s="$STAMP" '$1 == s {print $3}' "$DIR/jobs.tsv" | paste -sd'|'))(_|$)" || true)
[ -n "$running" ] && { echo "  running or queued:"; echo "$running" | sed 's/^/    /' | head -20; }
if [ "$MISSING" -eq 1 ]; then
  echo "  units still missing:"; bash scripts/with_env.sh python src/run_analysis.py --version "$VERSION" --list-units --missing 2>/dev/null | cut -f1 | sort | uniq -c | awk '{printf "    level %s: %s\n", $2, $1}'
fi
