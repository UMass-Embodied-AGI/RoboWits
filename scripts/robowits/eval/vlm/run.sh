#!/bin/bash
# Evaluate the VLM baselines on one RoboWits task (or the whole robowits_10 suite).
#
# Runs the baselines in MODES back to back for each task:
#   - vlm_planner  (closed-loop oracle primitives)
#   - vlm_controller (closed-loop, drives the arms through IK)
#   - vlm_policy (move_eef only, the VLA observation/action interface)
#
# All three by default; finished episodes are skipped. Set MODES to change which are
# run, e.g. MODES="policy", or MODES="planner planner_open controller policy" to
# include the open-loop ablation.
#
# Usage:
#   TASK_ID=01 bash scripts/robowits/eval/vlm/run.sh
#   N_EPISODES=5 bash scripts/robowits/eval/vlm/run.sh   # all 10 tasks
#   TASK_ID=01 EPISODES="1" MODES=policy bash scripts/robowits/eval/vlm/run.sh   # eval-dataset index 1 only
set -eo pipefail

ROBOWITS_10=(01 02 03 04 06 09 13 16 17 25)

EVAL_JSON_DIR=${EVAL_JSON_DIR:-"dataset/robowits/eval_dataset_seed"}
N_EPISODES=${N_EPISODES:-5}
# MODEL applies to every mode when set; unset, each uses its own default (default_model).
MODEL=${MODEL:-}
MODES=${MODES:-"planner controller policy"}
OUTPUT_DIR=${OUTPUT_DIR:-"tmp/gs-gym-evals/vlm_baselines"}
# Planner: max primitives per episode. Controller: max env steps / VLM calls.
MAX_STEPS=${MAX_STEPS:-20}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-1000}
MAX_VLM_CALLS=${MAX_VLM_CALLS:-20}
# Policy: its own reasoning effort and budgets (control steps, LLM calls).
# HARNESS=ours (default) or roboprobe; results go to policy_<model>_${HARNESS}/.
# EPISODES (policy only): space-separated eval-dataset indices, overriding N_EPISODES.
POLICY_REASONING_EFFORT=${POLICY_REASONING_EFFORT:-medium}
POLICY_MAX_EPISODE_STEPS=${POLICY_MAX_EPISODE_STEPS:-1000}
POLICY_MAX_CALLS=${POLICY_MAX_CALLS:-100}
HARNESS=${HARNESS:-ours}
EPISODES=${EPISODES:-}

# A claude-* MODEL (policy only, e.g. claude-fable-5-2) calls the Claude API; every other model calls OpenAI.
case "${MODEL}" in
    claude-*) API_KEY_VAR=ANTHROPIC_API_KEY ;;
    *) API_KEY_VAR=OPENAI_API_KEY ;;
esac
if [ -z "${!API_KEY_VAR}" ]; then
    echo "${API_KEY_VAR} is not set; the VLM baselines cannot run." >&2
    exit 1
fi

# One task if TASK_ID is set, otherwise the whole suite.
if [ -n "${TASK_ID}" ]; then
    TASKS=("${TASK_ID}")
else
    TASKS=("${ROBOWITS_10[@]}")
fi

# The model a mode runs: MODEL if set, else gpt-6-astra for the policy and gpt-4o for the rest.
model_for() {
    if [ -n "${MODEL}" ]; then echo "${MODEL}"; elif [ "$1" = "policy" ]; then echo gpt-6-astra; else echo gpt-4o; fi
}

# Where a mode writes its results: <mode>_<model>, plus _<harness> for the policy.
mode_dir() {
    if [ "$1" = "policy" ]; then
        echo "${OUTPUT_DIR}/policy_$(model_for policy)_${HARNESS}"
    else
        echo "${OUTPUT_DIR}/$1_$(model_for "$1")"
    fi
}

echo "=== VLM baseline evaluation ==="
echo "Tasks:            ${TASKS[*]}"
echo "Modes:            ${MODES}"
echo "Episodes per task: ${N_EPISODES}"
echo "Model:            ${MODEL:-per mode: gpt-4o (planner, controller), gpt-6-astra (policy)}"
if [[ " ${MODES} " == *" policy "* ]]; then
    echo "Policy:           reasoning ${POLICY_REASONING_EFFORT}, harness ${HARNESS}," \
        "${POLICY_MAX_CALLS} calls / ${POLICY_MAX_EPISODE_STEPS} control steps${EPISODES:+, episodes ${EPISODES}}"
fi
echo "Eval dataset:     ${EVAL_JSON_DIR}"
echo "Output:           ${OUTPUT_DIR}"
echo ""

# Keep going if a single task/mode fails so one bad task cannot lose the sweep.
failures=()

for task in "${TASKS[@]}"; do
    for mode in ${MODES}; do
        echo "============================================================"
        echo "task ${task} | ${mode}"
        echo "============================================================"

        case "${mode}" in
            planner|planner_open)
                planner_mode="closed"
                [ "${mode}" = "planner_open" ] && planner_mode="open"
                python scripts/robowits/eval/vlm/planner.py \
                    --task "${task}" \
                    --mode "${planner_mode}" \
                    --model "$(model_for "${mode}")" \
                    --num-episodes "${N_EPISODES}" \
                    --max-steps "${MAX_STEPS}" \
                    --eval-dataset-json "${EVAL_JSON_DIR}" \
                    --output-dir "$(mode_dir "${mode}")" \
                    || failures+=("${task}:${mode}")
                ;;
            policy)
                # shellcheck disable=SC2086  # EPISODES is a space-separated index list
                python scripts/robowits/eval/vlm/policy.py \
                    --harness "${HARNESS}" \
                    --task "${task}" \
                    --model "$(model_for "${mode}")" \
                    --reasoning-effort "${POLICY_REASONING_EFFORT}" \
                    --num-episodes "${N_EPISODES}" \
                    ${EPISODES:+--episodes ${EPISODES}} \
                    --max-episode-steps "${POLICY_MAX_EPISODE_STEPS}" \
                    --max-calls "${POLICY_MAX_CALLS}" \
                    --eval-dataset-json "${EVAL_JSON_DIR}" \
                    --output-dir "$(mode_dir "${mode}")" \
                    || failures+=("${task}:${mode}")
                ;;
            controller)
                python scripts/robowits/eval/vlm/controller.py \
                    --task "${task}" \
                    --model "$(model_for "${mode}")" \
                    --num-episodes "${N_EPISODES}" \
                    --max-episode-steps "${MAX_EPISODE_STEPS}" \
                    --max-vlm-calls "${MAX_VLM_CALLS}" \
                    --eval-dataset-json "${EVAL_JSON_DIR}" \
                    --output-dir "$(mode_dir "${mode}")" \
                    || failures+=("${task}:${mode}")
                ;;
            *)
                echo "Unknown mode: ${mode}" >&2
                exit 2
                ;;
        esac
    done
done

echo ""
echo "=== Done ==="
if [ ${#failures[@]} -gt 0 ]; then
    echo "Failed runs: ${failures[*]}"
    exit 1
fi
echo "Summarize with:"
for mode in ${MODES}; do
    echo "  python scripts/robowits/eval/summarize_results.py $(mode_dir "${mode}")"
done
