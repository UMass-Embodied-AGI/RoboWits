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
#   python scripts/robowits/examples/run_env.py --camera-type nyx --device cuda  # path traced

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
    parser.add_argument("--task-id", default="02")
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--save-video",
        metavar="PATH",
        default=None,
        help="save rendered frames to this .mp4 (default: tmp/examples/{task}.mp4)",
    )
    parser.add_argument("--camera", default="ego", choices=["ego", "wrist_right", "wrist_left", "gallery"])
    parser.add_argument(
        "--camera-type",
        default="rasterizer",
        choices=["rasterizer", "batch", "nyx"],
        help=(
            "renderer for the cameras; 'nyx' is a path tracer (needs --device cuda). "
            "~2.5x slower end-to-end than the rasterizer, near-independent of --spp"
        ),
    )
    parser.add_argument("--spp", type=int, default=None, help="Nyx samples per pixel (default: 32)")
    parser.add_argument("--gallery", action="store_true", help="add a wide third-person gallery camera")
    parser.add_argument(
        "--walls", action="store_true", help="add visual-only room walls (hides the env map; usually not wanted)"
    )
    parser.add_argument(
        "--no-ground",
        action="store_true",
        help="drop the ground plane so the environment map is the visible floor/backdrop",
    )
    parser.add_argument(
        "--env-map", default=None, metavar="PATH", help="environment map for Nyx (default: the standard one)"
    )
    parser.add_argument(
        "--render-mode",
        default=None,
        choices=["FastPathTracer", "RefPathTracer", "Forward", "Debug"],
        help="Nyx render mode (default: FastPathTracer)",
    )
    parser.add_argument("--mut", action="store_true", help="run all mutation variants of --task-id")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing videos (--mut mode)")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        metavar="DIR",
        help="directory with per-task eval JSON files; passes {stem}.json as eval_dataset_json",
    )
    parser.add_argument(
        "--episode-id", type=int, default=None, metavar="N", help="episode index in the eval dataset to start from"
    )
    return parser.parse_args()


def run_env(args, task: str) -> None:
    from lerobot.datasets.video_utils import encode_video_frames
    from PIL import Image

    make_kwargs = {}
    if args.dataset_dir:
        stem = task.split("/")[-1].split("-")[0]  # e.g. "06_01" or "06"
        make_kwargs["eval_dataset_json"] = str(Path(args.dataset_dir) / f"{stem}.json")

    if args.camera_type != "rasterizer":
        from gs_gym.sensors.camera_factory import env_map_for_task

        # Each task gets its own backdrop unless one was named explicitly.
        env_map = args.env_map or env_map_for_task(task.split("/")[-1].split("-")[0].split("_")[0])
        make_kwargs["camera_type"] = args.camera_type
        make_kwargs["camera_kwargs"] = {
            k: v
            for k, v in (("spp", args.spp), ("render_mode", args.render_mode), ("env_map", env_map))
            if v is not None
        }
    if args.gallery:
        make_kwargs["gallery_camera"] = True
    if args.walls:
        make_kwargs["add_walls"] = True
    if args.no_ground:
        make_kwargs["show_ground"] = False

    env = gs_gym.make(
        task,
        n_envs=args.n_envs,
        show_viewer=args.viewer,
        genesis_backend=args.device,
        device=args.device,
        **make_kwargs,
    )
    print(f"task:         {task}")
    print(f"n_envs:       {args.n_envs}")
    print(f"action space: {env.single_action_space if args.n_envs > 1 else env.action_space}")

    video_path = args.save_video or f"tmp/examples/{task.replace('/', '_')}.mp4"
    imgs_dir = Path("tmp/examples") / f"frames_{task.replace('/', '_')}"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = 0

    def save_frame(arr):
        nonlocal frame_idx
        Image.fromarray(arr).save(imgs_dir / f"frame-{frame_idx:06d}.png")
        frame_idx += 1

    if args.episode_id is not None:
        env.unwrapped.envs[0].set_eval_dataset_idx(args.episode_id)

    # The gallery camera is deliberately absent from the observation dict (it would change the
    # obs space that policies and eval expect), so read it from the env instead.
    inner_env = env.envs[0]

    def grab(obs):
        if args.camera in obs["pixels"]:
            return obs["pixels"][args.camera][0]
        return np.asarray(inner_env.render_frame(env_idx=0, camera_name=args.camera))

    obs, _ = env.reset()
    save_frame(grab(obs))

    noop = obs["agent_pos"]
    start = time.perf_counter()
    for _ in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(noop)
        save_frame(grab(obs))
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

    if args.camera_type == "nyx" and args.device == "cpu":
        raise SystemExit("--camera-type nyx needs a GPU; pass --device cuda")

    if args.mut:
        from gs_gym.envs.robowits import mutation  # noqa: F401 — triggers registration

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
            run_env(args, task)
    else:
        run_env(args, resolve_task(args.task_id))


if __name__ == "__main__":
    main()
