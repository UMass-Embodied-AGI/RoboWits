#!/bin/bash
set -e

HF_DATASET=${HF_DATASET:-"XHRlyb2001/RoboWits_lerobot_dataset"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/act_robowits"}
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
        --batch_size=32 \
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
        --job_name="act_robowits" \
        --policy.type=act \
        --policy.device=cuda \
        --policy.repo_id="local/act-robowits" \
        --policy.push_to_hub=false \
        --policy.chunk_size=50 \
        --policy.n_action_steps=50 \
        --policy.vision_backbone=resnet18 \
        --policy.dim_model=512 \
        --policy.n_encoder_layers=4 \
        --policy.n_decoder_layers=1 \
        --policy.use_vae=true \
        --policy.latent_dim=32 \
        --policy.optimizer_lr=1e-5 \
        --policy.optimizer_lr_backbone=1e-5 \
        --steps=${STEPS:-100000} \
        --batch_size=32 \
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
