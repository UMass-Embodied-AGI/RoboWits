#!/bin/bash

# Usage: TASK_IDS="01 02" bash scripts/robowits/eval/eval.sh
set -e

CONFIG=${CONFIG:-"EE_ABS"}
OBSERVATION_MODE=${OBSERVATION_MODE:-"EE"}
ENV_DEVICE=${ENV_DEVICE:-"cpu"}
EVAL_JSON_DIR=${EVAL_JSON_DIR:-"dataset/robowits/eval_dataset_50"}
N_EPISODES=${N_EPISODES:-50}
SPLIT=${SPLIT:-""}
_step=$(basename "$(dirname "${CHECKPOINT_PATH}")")
_run=$(basename "$(dirname "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")")
_suffix=${SPLIT:+_${SPLIT}}
OUTPUT_DIR=${OUTPUT_DIR:-"tmp/gs-gym-evals/${_run}_${_step}${_suffix}"}
TASK_IDS=${TASK_IDS:-"01 02 04 05"}

for TASK_ID in $TASK_IDS; do
    lerobot-eval \
        --env.type=gs_gym \
        --env.task="${TASK_ID}" \
        --env.max_episode_steps=1000 \
        --env.headless=true \
        --env.video=false \
        --env.control_mode="${CONFIG}" \
        --env.observation_mode="${OBSERVATION_MODE}" \
        --env.device="${ENV_DEVICE}" \
        --env.eval_dataset_json="${EVAL_JSON_DIR}/${TASK_ID}.json" \
        --policy.path="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to policy checkpoint path}" \
        --policy.device=cuda \
        --eval.batch_size=1 \
        --eval.n_episodes="${N_EPISODES}" \
        --eval.recording=true \
        --output_dir="${OUTPUT_DIR}/${TASK_ID}" \
        --trust_remote_code=true
done
