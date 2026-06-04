#!/bin/bash
set -e

HF_DATASET=${HF_DATASET:-"XHRlyb2001/RoboWits_lerobot_dataset"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/pi0_robowits"}
NUM_PROCESSES=${NUM_PROCESSES:-8}

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
        --wandb.project=robowits
else
    [ -d "${OUTPUT_DIR}" ] && rm -rf "${OUTPUT_DIR}"
    accelerate launch \
        --multi_gpu \
        --num_processes=${NUM_PROCESSES} \
        -m lerobot.scripts.lerobot_train \
        --dataset.repo_id="${HF_DATASET}" \
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
        --steps=${STEPS:-100000} \
        --batch_size=8 \
        --num_workers=4 \
        --save_freq=${SAVE_FREQ:-5000} \
        --log_freq=100 \
        --eval_freq=0 \
        --val_freq=${VAL_FREQ:-500} \
        --val_max_samples=5000 \
        --dataset.val_frac=0.04 \
        --dataset.video_backend=pyav \
        --wandb.enable=true \
        --wandb.project=robowits
fi
