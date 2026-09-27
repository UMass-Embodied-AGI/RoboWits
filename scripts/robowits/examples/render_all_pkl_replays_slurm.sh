#!/bin/bash
#SBATCH --job-name=render_nyx_demos
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=65536M
#SBATCH --time=720
#SBATCH --partition=rtx-mid
# Single-element array on purpose: gs-srun only runs inside this allocation when
# SLURM_ARRAY_JOB_ID is set. Without it, it unsets SLURM_* and submits a *separate* job that
# inherits neither the GPU nor the partition -- which lands on an H100, where Nyx dies during
# scene build (no RT cores).
#SBATCH --array=0-0
#SBATCH --output=/mnt/home/%u/gs-gym/tmp/slurm_logs/%A_render_nyx.out
#SBATCH --error=/mnt/home/%u/gs-gym/tmp/slurm_logs/%A_render_nyx.err
#SBATCH --open-mode=append

# Path-traced replay of every task's first recording. Env vars are forwarded to the inner script.

. /mnt/data/shared/config/env.sh

GS_GYM_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

gs-srun bash -lc "
set -eo pipefail
cd \"${GS_GYM_DIR}\"
source \"${GS_GYM_DIR}/.venv/bin/activate\"

export TASKS=\"\${TASKS:-}\"
export OUT_DIR=\"\${OUT_DIR:-tmp/replay_nyx_demos}\"
export SPP=\"\${SPP:-}\"
export MAX_STEPS=\"\${MAX_STEPS:-}\"
export GALLERY=\"\${GALLERY:-}\"
export FORCE=\"\${FORCE:-0}\"

bash \"${GS_GYM_DIR}/scripts/robowits/examples/render_all_pkl_replays.sh\"
"
