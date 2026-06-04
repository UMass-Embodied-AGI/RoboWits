#!/usr/bin/env python
# Run a RoboWits task for a fixed number of steps and report timing.
# With --mut, iterate over all mutation variants of --task-id.
#
# Usage:
#   python scripts/robowits/examples/run_env.py
#   python scripts/robowits/examples/run_env.py --task-id 06 --steps 50
#   python scripts/robowits/examples/run_env.py --n_envs 4 --viewer
#   python scripts/robowits/examples/run_env.py --mut
#   python scripts/robowits/examples/run_env.py --mut --task-id 06

import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np

import gs_gym
from gs_gym.envs.robowits.utils import resolve_task


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default="06")
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--camera", default="ego", choices=["ego", "wrist_right", "wrist_left"])
    parser.add_argument("--mut", action="store_true", help="run all mutation variants of --task-id")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing videos (--mut mode)")
    parser.add_argument(
        "--episode-id", type=int, default=None, metavar="N", help="episode index in the eval dataset to start from"
    )
    return parser.parse_args()


def run_env(args, task: str, dataset_dir: str) -> None:
    from lerobot.datasets.video_utils import encode_video_frames
    from PIL import Image

    stem = task.split("/")[-1].split("-")[0]  # e.g. "06_01" or "06"
    make_kwargs = {"eval_dataset_json": str(Path(dataset_dir) / f"{stem}.json")}

    env = gs_gym.make(
        task,
        n_envs=args.n_envs,
        show_viewer=args.viewer,
        genesis_backend='gpu' if args.device == 'mps' else args.device,
        device=args.device,
        **make_kwargs,
    )
    print(f"task:         {task}")
    print(f"dataset:      {make_kwargs['eval_dataset_json']}")
    print(f"n_envs:       {args.n_envs}")

    video_path = f"tmp/examples/{task.replace('/', '_')}.mp4"
    imgs_dir = Path("tmp/examples") / f"frames_{task.replace('/', '_')}"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = 0

    def save_frame(arr):
        nonlocal frame_idx
        Image.fromarray(arr).save(imgs_dir / f"frame-{frame_idx:06d}.png")
        frame_idx += 1

    if args.episode_id is not None:
        env.unwrapped.envs[0].set_eval_dataset_idx(args.episode_id)

    obs, _ = env.reset()
    save_frame(obs["pixels"][args.camera][0])

    noop = obs["agent_pos"]
    start = time.perf_counter()
    for _ in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(noop)
        save_frame(obs["pixels"][args.camera][0])
        if np.any(terminated) or np.any(truncated):
            break
    elapsed = time.perf_counter() - start

    total_steps = args.steps * args.n_envs
    print(f"steps:        {args.steps}  ({total_steps} env-steps)")
    print(f"time:         {elapsed:.2f}s  ({total_steps / elapsed:.1f} env-steps/s)")

    saved_stderr = os.dup(2)
    with open(os.devnull, "wb") as devnull:
        os.dup2(devnull.fileno(), 2)
        try:
            encode_video_frames(imgs_dir, video_path, fps=30, overwrite=True)
        finally:
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
    shutil.rmtree(imgs_dir)
    print(f"video:        {video_path}  ({frame_idx} frames @ 30 fps)")

    env.close()


def main():
    args = parse_args()

    if args.mut:
        from gs_gym.envs.robowits import mutation  # noqa: F401 — triggers registration

        dataset_dir = "dataset/robowits/eval_dataset_mutation_10"
        prefix = args.task_id.zfill(2) + "_"
        mut_tasks = sorted(k for k in gs_gym.TASK_REGISTRY if "robowits/" in k and k.split("/")[-1].startswith(prefix))
        if not mut_tasks:
            raise ValueError(f"No mutation tasks found for task-id '{args.task_id}'.")
        print(f"Running {len(mut_tasks)} mutation tasks for task {args.task_id.zfill(2)}")
        for i, task in enumerate(mut_tasks, 1):
            video_path = f"tmp/examples/{task.replace('/', '_')}.mp4"
            if Path(video_path).exists() and not args.overwrite:
                print(f"[{i}/{len(mut_tasks)}] {task} ... SKIP")
                continue
            print(f"[{i}/{len(mut_tasks)}] {task}")
            run_env(args, task, dataset_dir)
    else:
        run_env(args, resolve_task(args.task_id), "dataset/robowits/eval_dataset_50")


if __name__ == "__main__":
    main()
