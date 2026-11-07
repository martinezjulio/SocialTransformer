#!/bin/bash
# One-time setup of the analysis pipeline on Sherlock: miniforge and the SocialTransformer env (Python, R, tectonic)
# beside the clone under group storage, a Rocky 9 Apptainer image next to them (scripts/sherlock.def: the nodes run CentOS 7,
# glibc 2.17, which the pinned jax, torch, numpy and scipy wheels refuse; every python and Rscript of the pipeline runs inside
# the image, see scripts/with_env.sh), the data roots under $SCRATCH (the lab layout, split by storage tier:
# $GROUP_HOME/$USER/SocialBanditsProject/{SocialTransformer,miniforge} and $SCRATCH/SocialBanditsProject/{data,experiments,output}),
# and the repo's .env. Run inside an sh_dev session in the clone (the installs are too heavy for a login node):
#   sh_dev -c 4 -m 16GB -t 2:00:00; bash scripts/setup_sherlock.sh
# Variables (all optional): ENV_ROOT (default $GROUP_HOME/$USER/SocialBanditsProject: persistent, 1 TB shared with the group; the env ~10 GB),
# DATA_ROOT (default $SCRATCH/SocialBanditsProject: 100 TB, files untouched for 90 days are purged, so copy paper/ out after
# a run), CUDA_MAJOR (12, the build every current driver supports; 13 needs a driver >= 580 on the GPU nodes).
# renv's package cache goes to $ENV_ROOT/renv-cache (RENV_PATHS_CACHE, also written to .env): Sherlock sets XDG_CACHE_HOME=/tmp,
# node-local and wiped at job end, where renv would otherwise put the cache the project library links into.
# Then copy, per version, $EXPERIMENT_DATA_DIR/<version>/ (the anonymised preprocessed file and exclusions.csv only),
# $OUTPUT_DATA_DIR/eval/<version>/ and the checkpoint named in its metadata/inference_config.yaml, and submit with
# scripts/slurm/submit.sh. See README "Analysis on Sherlock".
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ENV_ROOT="${ENV_ROOT:-${GROUP_HOME:?GROUP_HOME is not set: not a Sherlock shell}/$USER/SocialBanditsProject}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH:?SCRATCH is not set}/SocialBanditsProject}"
CUDA_MAJOR="${CUDA_MAJOR:-12}"
TORCH_INDEX="https://download.pytorch.org/whl/cu$([ "$CUDA_MAJOR" = 13 ] && echo 130 || echo 126)"
mkdir -p "$ENV_ROOT" "$DATA_ROOT"/{data,experiments,output}

if [ ! -x "$ENV_ROOT/miniforge/bin/conda" ]; then
  curl -L -o "$ENV_ROOT/Miniforge3.sh" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash "$ENV_ROOT/Miniforge3.sh" -b -p "$ENV_ROOT/miniforge" && rm "$ENV_ROOT/Miniforge3.sh"
fi
source "$ENV_ROOT/miniforge/etc/profile.d/conda.sh"
conda env list | grep -q "^SocialTransformer " || conda create -y -n SocialTransformer -c conda-forge python=3.11 r-base=4.5.3 tectonic \
  compilers make cmake nlopt icu cairo fribidi libpng zlib harfbuzz libxml2 libxml2-devel fontconfig expat libjpeg-turbo libtiff pkg-config freetype   # R package build deps (conda R compiles with the conda toolchain; expat for fontconfig.pc)
conda activate SocialTransformer
IMAGE="$ENV_ROOT/rocky9.sif"
[ -f "$IMAGE" ] || APPTAINER_CACHEDIR="$DATA_ROOT/.apptainer-cache" APPTAINER_TMPDIR=/tmp apptainer build "$IMAGE" scripts/sherlock.def   # Rocky 9 (glibc 2.34) plus the base tools R expects; ~1 min, unprivileged

cat > .env <<ENV
export CONDA_SETUP=$ENV_ROOT/miniforge/etc/profile.d/conda.sh
export TRAINING_DATA_ROOT_DIR=$DATA_ROOT/data
export OUTPUT_DATA_DIR=$DATA_ROOT/output
export EXPERIMENT_DATA_DIR=$DATA_ROOT/experiments
export WANDB_MODE=offline
export SLURM_CPU_PARTITION=owners,normal
export SLURM_GPU_PARTITION=owners
export SLURM_ARRAY_CHUNK=96
export SLURM_FINAL_PARTITION=hns,normal
export SLURM_GPU_OPTS="--gpu_cmode=shared -C (GPU_GEN:AMP|GPU_GEN:HPR|GPU_GEN:LOV|GPU_GEN:VLT)&(GPU_MEM:48GB|GPU_MEM:80GB|GPU_MEM:141GB)"
export SLURM_GPU_MEM=48G
export SLURM_GPU_TIME=4:00:00
export SLURM_FINAL_MEM=128G
export N_TABLE_BLOCKS=8
export RENV_PATHS_CACHE=$ENV_ROOT/renv-cache
export CONTAINER=$IMAGE
ENV
IN="bash scripts/with_env.sh"   # inside the image, env active
$IN pip install --only-binary=:all: "jax[cuda${CUDA_MAJOR}]==0.10.2"
$IN pip install --only-binary=:all: torch==2.9.1 torchvision==0.24.1 --index-url "$TORCH_INDEX"
$IN pip install --only-binary=:all: -r requirements-sherlock.txt
# R packages of renv.lock, built from source inside the image (renv keys its library by OS, so they must be built where they run;
# about 10 min on 4 cores). stringi's configure links a test executable without R's LDFLAGS, and inside the image the system's
# stub libdl hides dladdr from the linker, so it gets the flag R itself uses.
STRINGI_LDFLAGS="-Wl,--allow-shlib-undefined" MAKEFLAGS="-j${SLURM_CPUS_PER_TASK:-4}" $IN Rscript -e 'options(Ncpus = as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "4"))); renv::restore()'
$IN python -c "import jax, torch; print('jax', jax.__version__, jax.devices(), '| torch', torch.__version__)"   # CPU-only here: the GPUs are on the nodes
echo "done: env under $ENV_ROOT/miniforge, data under $DATA_ROOT, .env written"
