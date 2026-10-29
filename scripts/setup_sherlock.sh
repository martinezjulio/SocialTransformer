#!/bin/bash
# One-time setup of the analysis pipeline on Sherlock: miniforge and the SocialTransformer env (Python, R, tectonic)
# under group storage, the data roots under $SCRATCH, and the repo's .env. Run from a login node inside the clone:
#   bash scripts/setup_sherlock.sh
# Variables (all optional): ENV_ROOT (default $GROUP_HOME/$USER: persistent, 1 TB shared with the group; conda env ~10 GB),
# DATA_ROOT (default $SCRATCH/SocialTransformer: 100 TB, files untouched for 90 days are purged, so copy paper/ out after
# a run), CUDA_MAJOR (12, the build every current driver supports; 13 needs a driver >= 580 on the GPU nodes).
# Then copy, per version, $EXPERIMENT_DATA_DIR/<version>/ (the anonymised preprocessed file and exclusions.csv only),
# $OUTPUT_DATA_DIR/eval/<version>/ and the checkpoint named in its metadata/inference_config.yaml, and submit with
# scripts/slurm/submit.sh. See README "Analysis on Sherlock".
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ENV_ROOT="${ENV_ROOT:-${GROUP_HOME:?GROUP_HOME is not set: not a Sherlock login shell}/$USER}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH:?SCRATCH is not set}/SocialTransformer}"
CUDA_MAJOR="${CUDA_MAJOR:-12}"
TORCH_INDEX="https://download.pytorch.org/whl/cu$([ "$CUDA_MAJOR" = 13 ] && echo 130 || echo 126)"
mkdir -p "$ENV_ROOT" "$DATA_ROOT"/{experiment_data,outputs,training_data}

if [ ! -x "$ENV_ROOT/miniforge/bin/conda" ]; then
  curl -L -o "$ENV_ROOT/Miniforge3.sh" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash "$ENV_ROOT/Miniforge3.sh" -b -p "$ENV_ROOT/miniforge" && rm "$ENV_ROOT/Miniforge3.sh"
fi
source "$ENV_ROOT/miniforge/etc/profile.d/conda.sh"
conda env list | grep -q "^SocialTransformer " || conda create -y -n SocialTransformer -c conda-forge python=3.11 r-base=4.5 tectonic \
  cairo fribidi libpng zlib harfbuzz libxml2 libxml2-devel fontconfig libjpeg-turbo libtiff pkg-config freetype   # R package build deps
conda activate SocialTransformer
pip install "jax[cuda${CUDA_MAJOR}]==0.10.2"
pip install torch==2.9.1 torchvision==0.24.1 --index-url "$TORCH_INDEX"
pip install -r requirements-sherlock.txt

cat > .env <<ENV
export CONDA_SETUP=$ENV_ROOT/miniforge/etc/profile.d/conda.sh
export TRAINING_DATA_ROOT_DIR=$DATA_ROOT/training_data
export OUTPUT_DATA_DIR=$DATA_ROOT/outputs
export EXPERIMENT_DATA_DIR=$DATA_ROOT/experiment_data
export WANDB_MODE=offline
export SLURM_CPU_PARTITION=normal
export SLURM_GPU_PARTITION=owners
export SLURM_FINAL_PARTITION=hns
export N_TABLE_BLOCKS=8
ENV
Rscript -e 'renv::restore()'   # the R packages of renv.lock, built from source (takes a while; compilers are on the login nodes)
python -c "import jax, torch; print('jax', jax.__version__, jax.devices(), '| torch', torch.__version__)"   # CPU-only here: the GPUs are on the nodes
echo "done: env under $ENV_ROOT/miniforge, data under $DATA_ROOT, .env written"
