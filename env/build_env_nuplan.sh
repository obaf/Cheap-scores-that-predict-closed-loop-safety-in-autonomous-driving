#!/bin/bash
# Build the main "nuplan" conda env (Python 3.9, CPU-only torch 2.0.1).
# Runs: nuplan-devkit e9241677 (simulation + training framework), tuplan_garage, interPlan,
#       Diffusion-Planner, planTF and pluto model code.
# Shared third-party versions are constrained to nuplan-devkit's own requirements_lock.txt.
set -euo pipefail
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
ENV=nuplan
CODE=/data/code
OUT=/data/code/envs
mkdir -p "$OUT/logs" "$OUT/constraints"
export PIP_CACHE_DIR=/data/code/.pip-cache
export MAX_JOBS=6
export PIP_DISABLE_PIP_VERSION_CHECK=1
step() { echo "$(date -u +%FT%TZ) STEP $*"; }

# constraints from nuplan-devkit's lock file (strip hashes and extras); torch lock not used (CUDA 1.9 build)
grep -oE '^[A-Za-z0-9_.-]+(\[[a-z0-9,]+\])?==[^ ]+' "$CODE/nuplan-devkit/requirements_lock.txt" \
  | sed -E 's/\[[a-z0-9,]+\]//' > "$OUT/constraints/nuplan_lock_constraints.txt"
wc -l "$OUT/constraints/nuplan_lock_constraints.txt"

if ! conda env list | grep -qE "^$ENV "; then
  step "conda create"
  conda create -y -q -n "$ENV" -c conda-forge python=3.9 pip=23.3.1
fi
conda activate "$ENV"
python --version

step "torch cpu"
nice -n 10 pip install -q "torch==2.0.1+cpu" "torchvision==0.15.2+cpu" \
  --extra-index-url https://download.pytorch.org/whl/cpu -c "$OUT/constraints/nuplan_lock_constraints.txt"

step "nuplan requirements.txt (lock-constrained)"
nice -n 10 pip install -q -r "$CODE/nuplan-devkit/requirements.txt" -c "$OUT/constraints/nuplan_lock_constraints.txt"

step "nuplan-devkit editable"
nice -n 10 pip install -q --no-deps -e "$CODE/nuplan-devkit"

step "extra ML deps (planTF/pluto/Diffusion-Planner/tuplan_garage pins)"
nice -n 10 pip install -q -c "$OUT/constraints/nuplan_lock_constraints.txt" \
  "pytorch-lightning==2.0.1" "torchmetrics==0.10.2" "timm==1.0.10" "tensorboard==2.11.2" \
  "wandb==0.14.2" "rich==13.3.4" numba mmengine imageio \
  "scikit-learn==1.2.2" "positional-encodings==6.0.1"

step "tuplan_garage, interPlan, Diffusion-Planner editable"
nice -n 10 pip install -q --no-deps -e "$CODE/tuplan_garage"
nice -n 10 pip install -q --no-deps -e "$CODE/interPlan"
nice -n 10 pip install -q --no-deps -e "$CODE/Diffusion-Planner"

step "natten 0.14.6 from source (CPU)"
nice -n 10 pip install -v --no-build-isolation "natten==0.14.6" > "$OUT/logs/natten_build.log" 2>&1 || echo "NATTEN BUILD FAILED (see logs/natten_build.log)"

step "pip check"
pip check || true
step "done"
