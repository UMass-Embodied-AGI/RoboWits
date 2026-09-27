#!/bin/bash
set -e

HF_DATASET=${HF_DATASET:-"XHRlyb2001/RoboWits_lerobot_dataset"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/pi0_robowits"}
NUM_PROCESSES=${NUM_PROCESSES:-8}
EPISODES=${EPISODES:-""}  # optional episode subset, e.g. "[0,1,2]" (default: all episodes)
CHUNK_SIZE=${CHUNK_SIZE:-""}  # action horizon (default: pi0's 50)
IMAGE_TRANSFORMS=${IMAGE_TRANSFORMS:-0}  # 1 = enable the ColorJitter augmentation below
WANDB_PROJECT=${WANDB_PROJECT:-robowits}

# ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.08), all four always applied
COLOR_JITTER_TFS='{
  "brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.6, 1.4]}},
  "contrast":   {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.6, 1.4]}},
  "saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}},
  "hue":        {"weight": 1.0, "type": "ColorJitter", "kwargs": {"hue": [-0.08, 0.08]}}
}'

EXTRA_ARGS=()
if [ -n "${CHUNK_SIZE}" ]; then
    EXTRA_ARGS+=(--policy.chunk_size="${CHUNK_SIZE}" --policy.n_action_steps="${CHUNK_SIZE}")
fi
if [ "${IMAGE_TRANSFORMS}" = "1" ]; then
    EXTRA_ARGS+=(
        --dataset.image_transforms.enable=true
        --dataset.image_transforms.max_num_transforms=4
        --dataset.image_transforms.tfs="${COLOR_JITTER_TFS}"
    )
fi

LAST_CKPT="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"

if [ -f "${LAST_CKPT}" ]; then
    accelerate launch \
        --multi_gpu \
        --num_processes=${NUM_PROCESSES} \
        -m lerobot.scripts.lerobot_train \
        --resume=true \
        --config_path="${LAST_CKPT}" \
        --steps=${STEPS:-100000} \
        --batch_size=8 \
        --num_workers=4 \
        --wandb.enable=true \
        --wandb.project="${WANDB_PROJECT}"
else
    [ -d "${OUTPUT_DIR}" ] && rm -rf "${OUTPUT_DIR}"
    accelerate launch \
        --multi_gpu \
        --num_processes=${NUM_PROCESSES} \
        -m lerobot.scripts.lerobot_train \
        --dataset.repo_id="${HF_DATASET}" \
        ${EPISODES:+--dataset.episodes="${EPISODES}"} \
        --output_dir="${OUTPUT_DIR}" \
        --job_name="pi0_robowits" \
        --policy.type=pi0 \
        --policy.device=cuda \
        --policy.repo_id="local/pi0-robowits" \
        --policy.pretrained_path=lerobot/pi0_base \
        --policy.pretrained_revision=26b99b9439acb1e352439e34ee9c67af0d76efa3 \
        --policy.push_to_hub=false \
        --policy.gradient_checkpointing=false \
        --policy.dtype=bfloat16 \
        --policy.freeze_vision_encoder=false \
        --policy.train_expert_only=false \
        "${EXTRA_ARGS[@]}" \
        --steps=${STEPS:-100000} \
        --batch_size=8 \
        --num_workers=4 \
        --save_freq=${SAVE_FREQ:-5000} \
        --log_freq=100 \
        --eval_freq=0 \
        --val_freq=${VAL_FREQ:-500} \
        --val_max_samples=${VAL_MAX_SAMPLES:-5000} \
        --dataset.val_frac=${VAL_FRAC:-0.04} \
        --dataset.video_backend=pyav \
        --wandb.enable=true \
        --wandb.project="${WANDB_PROJECT}"
fi
