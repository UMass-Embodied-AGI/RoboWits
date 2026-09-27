#!/bin/bash

# Usage: TASK_IDS="01 02" bash scripts/robowits/eval/eval.sh
set -e

CONFIG=${CONFIG:-"EE_ABS"}
OBSERVATION_MODE=${OBSERVATION_MODE:-"EE"}
ENV_DEVICE=${ENV_DEVICE:-"auto"}  # auto = each task's own backend; cpu/cuda forces one
EVAL_JSON_DIR=${EVAL_JSON_DIR:-"dataset/robowits/eval_dataset_seed"}
N_EPISODES=${N_EPISODES:-50}
SPLIT=${SPLIT:-""}
_step=$(basename "$(dirname "${CHECKPOINT_PATH}")")
_run=$(basename "$(dirname "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")")
_suffix=${SPLIT:+_${SPLIT}}
OUTPUT_DIR=${OUTPUT_DIR:-"tmp/gs-gym-evals/${_run}_${_step}${_suffix}"}
TASK_IDS=${TASK_IDS:-"01 02 04 05"}

for TASK_ID in $TASK_IDS; do
    # A short id ("02") or a full registered name ("robowits/02_01-v0"). Mutation variants
    # must be passed by name: draccus parses a bare "02_01" as octal (0o201). The stem names
    # the eval dataset file and the results directory: 02 | 02_01 | 06-dominos -> 06.
    _stem="${TASK_ID##*/}"
    _stem="${_stem%-v0}"
    _stem="${_stem%%-*}"

    lerobot-eval \
        --env.type=gs_gym \
        --env.task="${TASK_ID}" \
        --env.max_episode_steps=1000 \
        --env.headless=true \
        --env.video=false \
        --env.control_mode="${CONFIG}" \
        --env.observation_mode="${OBSERVATION_MODE}" \
        --env.device="${ENV_DEVICE}" \
        --env.eval_dataset_json="${EVAL_JSON_DIR}/${_stem}.json" \
        --policy.path="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to policy checkpoint path}" \
        --policy.device=cuda \
        --eval.batch_size=1 \
        --eval.n_episodes="${N_EPISODES}" \
        --eval.recording=true \
        --output_dir="${OUTPUT_DIR}/${_stem}" \
        --trust_remote_code=true
done
