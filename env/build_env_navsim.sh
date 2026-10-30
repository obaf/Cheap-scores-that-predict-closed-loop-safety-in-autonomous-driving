#!/bin/bash
# Build the "navsim" conda env exactly as NAVSIM v2.2 documents it (environment.yml + requirements.txt +
# pip install -e .), except that torch/torchvision are the CPU wheels of the pinned versions.
# NAVSIM pins nuplan-devkit @ tag nuplan-devkit-v1.2 (installed from git, non-editable).
set -euo pipefail
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
ENV=navsim
CODE=/data/code
export PIP_CACHE_DIR=/data/code/.pip-cache
export PIP_DISABLE_PIP_VERSION_CHECK=1
step() { echo "$(date -u +%FT%TZ) STEP $*"; }

if ! conda env list | grep -qE "^$ENV "; then
  step "conda create"
  conda create -y -q -n "$ENV" -c conda-forge python=3.9 pip=23.3.1
fi
conda activate "$ENV"
python --version

step "torch cpu (pinned versions of navsim requirements.txt)"
nice -n 10 pip install -q "torch==2.0.1+cpu" "torchvision==0.15.2+cpu" --extra-index-url https://download.pytorch.org/whl/cpu

step "navsim requirements.txt"
nice -n 10 pip install -q -r "$CODE/navsim/requirements.txt"

step "navsim editable"
nice -n 10 pip install -q -e "$CODE/navsim"

step "pip check"
pip check || true
step "done"
