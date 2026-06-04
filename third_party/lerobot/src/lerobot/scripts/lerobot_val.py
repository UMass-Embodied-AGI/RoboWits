#!/usr/bin/env python
"""Standalone per-task MSE validation script. Intended to run on a single GPU as an async Slurm job."""

import argparse
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_val_datasets
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


def main():
    register_third_party_plugins()
    init_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to snapshot dir containing pretrained_model/")
    parser.add_argument("--step", type=int, required=True,
                        help="Training step this snapshot corresponds to")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="WandB run ID to resume. Falls back to WANDB_RUN_ID env var.")
    parser.add_argument("--no-delete-snapshot", dest="delete_snapshot", action="store_false",
                        help="Keep snapshot dir after logging (useful for debugging)")
    parser.set_defaults(delete_snapshot=True)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    pretrained_dir = checkpoint_dir / "pretrained_model"
    step = args.step
    wandb_run_id = args.wandb_run_id or os.environ.get("WANDB_RUN_ID")

    # Load train config — has dataset, policy type, val settings
    cfg = TrainPipelineConfig.from_pretrained(pretrained_dir)
    cfg.policy.pretrained_path = pretrained_dir

    # Build val dataset using the same deterministic split as training
    _train_dataset, val_dataset = make_train_val_datasets(cfg)
    if val_dataset is None:
        logging.warning("val_dataset is None (val_frac=0). Nothing to validate.")
        return

    val_task_names = {idx: name for idx, name in enumerate(val_dataset.meta.tasks.index)}
    num_tasks = len(val_task_names)

    # Load policy from snapshot weights
    policy = make_policy(cfg=cfg.policy, ds_meta=val_dataset.meta, rename_map=cfg.rename_map)
    device = next(policy.parameters()).device
    policy.eval()

    # Build preprocessor — reuse saved normalizer stats from snapshot
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": val_dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": val_dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        },
    )

    # Deterministic per-task selection: first (val_max_samples // num_tasks) samples per task
    samples_per_task = (cfg.val_max_samples // num_tasks) if cfg.val_max_samples > 0 else None
    task_idx_array = np.array(val_dataset.hf_dataset["task_index"])
    selected_indices: list[int] = []
    for t in range(num_tasks):
        task_frame_indices = np.where(task_idx_array == t)[0]
        if samples_per_task is not None:
            task_frame_indices = task_frame_indices[:samples_per_task]
        selected_indices.extend(task_frame_indices.tolist())

    val_subset = torch.utils.data.Subset(val_dataset, selected_indices)
    val_dataloader = torch.utils.data.DataLoader(
        val_subset,
        num_workers=4,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2,
    )

    # Per-task MSE loop
    task_mses: dict[int, list[float]] = {}
    with torch.no_grad():
        for val_batch in val_dataloader:
            batch_task_indices = val_batch["task_index"].tolist()
            val_batch = preprocessor(val_batch)
            gt_actions = val_batch["action"]  # (B, horizon, action_dim), normalized
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred_actions = policy.predict_action_chunk(val_batch)
            # generate_actions returns n_action_steps; gt has full horizon — align.
            gt_actions = gt_actions[:, : pred_actions.shape[1]]
            sample_mses = F.mse_loss(pred_actions, gt_actions, reduction="none").mean(
                dim=list(range(1, pred_actions.ndim))
            )
            for i, task_idx in enumerate(batch_task_indices):
                task_mses.setdefault(task_idx, []).append(sample_mses[i].item())

    per_task_mse = {t: sum(mses) / len(mses) for t, mses in task_mses.items()}
    val_mse = sum(per_task_mse.values()) / len(per_task_mse)
    val_r2 = 1.0 - val_mse
    per_task_lines = "\n".join(
        f"  {val_task_names.get(t, t)}: {mse:.4f}" for t, mse in sorted(per_task_mse.items())
    )
    logging.info(f"Val MSE at step {step}: {val_mse:.4f}, R²: {val_r2:.4f}\nPer-task MSE:\n{per_task_lines}")

    # Log to wandb by resuming the training run
    if cfg.wandb.enable and cfg.wandb.project:
        if not wandb_run_id:
            logging.warning("wandb enabled but no run_id; skipping wandb logging.")
        else:
            import wandb
            os.environ["WANDB_SILENT"] = "True"
            wandb.init(
                id=wandb_run_id,
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                resume="must",
            )
            # Use a custom x-axis so eval metrics display against training step.
            # wandb.log(step=N) silently drops data when N < current run step,
            # so we log at the current commit and pin all eval/* to "eval/train_step".
            wandb.define_metric("eval/*", step_metric="eval/train_step")
            log_dict = {"eval/train_step": step, "eval/val_mse": val_mse, "eval/val_r2": val_r2}
            for task_idx, mse in per_task_mse.items():
                task_label = val_task_names.get(task_idx, str(task_idx))
                log_dict[f"eval/val_mse/{task_label}"] = mse
            wandb.log(data=log_dict)
            wandb.finish()
            logging.info(f"Logged to wandb run {wandb_run_id} at step {step}")

    if args.delete_snapshot:
        if checkpoint_dir.is_symlink():
            checkpoint_dir.unlink()
        else:
            shutil.rmtree(checkpoint_dir)
        logging.info(f"Deleted snapshot: {checkpoint_dir}")


if __name__ == "__main__":
    main()
