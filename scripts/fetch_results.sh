#!/bin/bash
# Run on the lab machine: pulls one version's analysis outputs from Sherlock into this machine's $OUTPUT_DATA_DIR/analysis/<version>/
# through the data transfer node. Fetches the report (paper/: results.tex, results.md, tables, figures) and every stage's
# CSV and JSON outputs; leaves the caches (belief tables, fit pickles, fallback refits, simulated datasets) on scratch unless
# fits are named. Reads the lab .env for OUTPUT_DATA_DIR. One Duo prompt per rsync.
#   bash scripts/fetch_results.sh --version 2026_11_05_full_study [--sunet juliomz] [--fits ROS,Switch-by-round]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; source "${ENV_FILE:-.env}"; set +a
VERSION=""; SUNET="${SUNET:-juliomz}"; FITS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="$2"; shift 2 ;;
    --sunet) SUNET="$2"; shift 2 ;;
    --fits) FITS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$VERSION" ] || { echo "--version is required (e.g. 2026_11_05_full_study)" >&2; exit 2; }
SRC="$SUNET@dtn.sherlock.stanford.edu:/scratch/users/$SUNET/SocialBanditsProject/output/analysis/$VERSION"
DST="$OUTPUT_DATA_DIR/analysis/$VERSION"
mkdir -p "$DST"
echo "== outputs of $VERSION -> $DST (caches excluded)"
rsync -avP --exclude='tables/' --exclude='fits/' --exclude='loso/' --exclude='*.pkl' --exclude='slurm/' --exclude='*.block*.npy' "$SRC/" "$DST/"
if [ -n "$FITS" ]; then
  echo "== cached fits: $FITS"
  mkdir -p "$DST/choice/fits"
  for m in ${FITS//,/ }; do
    stem=$(python3 -c "import re,sys; print(re.sub(r'[^\w.-]', '_', sys.argv[1]))" "$m")
    rsync -avP --include="${stem}_*.pkl" --exclude='*' "$SRC/choice/fits/" "$DST/choice/fits/"
  done
fi
echo "== done: $(find "$DST" -type f | wc -l) files, $(du -sh "$DST" | cut -f1)"
