#!/bin/bash
#SBATCH --job-name=lerobot_val
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=81920M
#SBATCH --time=60
#SBATCH --partition=rtx-mid
#SBATCH --array=0-0
#SBATCH --output=%x_%A_%a_val_log.out
#SBATCH --error=%x_%A_%a_val_log.err
#SBATCH --open-mode=append

. /mnt/data/shared/config/env.sh

GS_GYM_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

# Do not use gs-srun — this job is submitted from within a training container
# and gs-srun does not support nested Slurm jobs.
set -eox pipefail
cd "${GS_GYM_DIR}"
source "${GS_GYM_DIR}/.venv/bin/activate"

: "${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set}"
: "${VAL_STEP:?VAL_STEP must be set}"
: "${WANDB_RUN_ID:?WANDB_RUN_ID must be set}"

python -m lerobot.scripts.lerobot_val \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --step "${VAL_STEP}" \
    --wandb_run_id "${WANDB_RUN_ID}"
