#!/bin/bash

# Usage: TASK_IDS="01 02" bash scripts/robowits/eval/eval_mutation.sh
set -e

CONFIG=${CONFIG:-"EE_ABS"}
OBSERVATION_MODE=${OBSERVATION_MODE:-"EE"}
EVAL_JSON_DIR=${EVAL_JSON_DIR:-"dataset/robowits/eval_dataset_mutation_10"}
N_EPISODES=${N_EPISODES:-5}
SPLIT=${SPLIT:-""}
_step=$(basename "$(dirname "${CHECKPOINT_PATH}")")
_run=$(basename "$(dirname "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")")
_suffix=${SPLIT:+_${SPLIT}}
OUTPUT_DIR=${OUTPUT_DIR:-"tmp/gs-gym-evals/${_run}_${_step}${_suffix}"}
TASK_IDS=${TASK_IDS:-"01 02"}
MUTATION_IDS=${MUTATION_IDS:-""}  # optional index filter, e.g. "[0,3]" to select specific variants

for TASK_ID in $TASK_IDS; do
    lerobot-eval \
        --env.type=gs_gym \
        --env.task="robowits_mutation_${TASK_ID}" \
        --env.max_episode_steps=1000 \
        --env.headless=true \
        --env.video=false \
        --env.control_mode="${CONFIG}" \
        --env.observation_mode="${OBSERVATION_MODE}" \
        --env.eval_dataset_json="${EVAL_JSON_DIR}" \
        ${MUTATION_IDS:+--env.task_ids="${MUTATION_IDS}"} \
        --policy.path="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to policy checkpoint path}" \
        --policy.device=cuda \
        --eval.batch_size=1 \
        --eval.n_episodes="${N_EPISODES}" \
        --eval.recording=true \
        --output_dir="${OUTPUT_DIR}/mutation_${TASK_ID}" \
        --trust_remote_code=true
done
