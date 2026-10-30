# Experiment 2 environment variables. Usage:
#   source /data/code/envs/env.sh            # variables only
#   source /data/code/envs/env.sh nuplan     # variables + activate conda env "nuplan"
#   source /data/code/envs/env.sh navsim     # variables + activate conda env "navsim"

# --- required by nuplan-devkit / NAVSIM ---
export NUPLAN_DATA_ROOT=/data/nuplan/dataset
export NUPLAN_MAPS_ROOT=/data/nuplan/dataset/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_EXP_ROOT=/data/nuplan/exp
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NAVSIM_EXP_ROOT=/data/navsim/exp
export NAVSIM_DEVKIT_ROOT=/data/code/navsim

# --- convenience paths (used in the documented commands) ---
export NUPLAN_DEVKIT_ROOT=/data/code/nuplan-devkit
export TUPLAN_GARAGE_ROOT=/data/code/tuplan_garage
export INTERPLAN_ROOT=/data/code/interPlan
export DIFFUSION_PLANNER_ROOT=/data/code/Diffusion-Planner
export PLANTF_ROOT=/data/code/planTF
export PLUTO_ROOT=/data/code/pluto
export EXP2_CKPT=/data/checkpoints
# DB folders. nuplan's scenario_builder=nuplan defaults to $NUPLAN_DATA_ROOT/nuplan-v1.1/trainval (absent here):
# always pass  scenario_builder.db_files=<dir|db|[list]>  (see README.md, "DB layout").
export NUPLAN_DB_VAL=$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/val
export NUPLAN_DB_TEST=$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/test
export NUPLAN_DB_TRAIN_PIT=$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/train_pittsburgh

# --- threading: one BLAS/OpenMP/torch thread per process by default (sharded runner = one sim per vCPU).
#     Override before sourcing, e.g. OMP_NUM_THREADS=4 source env.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-$OMP_NUM_THREADS}
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-$OMP_NUM_THREADS}

export HYDRA_FULL_ERROR=1
# exp2tools (StepTimerCallback etc.) importable from every env
case ":${PYTHONPATH:-}:" in *":/data/code/envs/tools:"*) ;; *) export PYTHONPATH=/data/code/envs/tools${PYTHONPATH:+:$PYTHONPATH};; esac
# no wandb network calls from planTF/pluto/Diffusion-Planner code paths
export WANDB_MODE=${WANDB_MODE:-disabled}

if [ -n "${1:-}" ]; then
  source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
  conda activate "$1"
fi
