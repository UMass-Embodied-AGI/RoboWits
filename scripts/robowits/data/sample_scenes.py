#!/usr/bin/env python3
"""Sample random initial scenes for given tasks and save their objs_info.

For each given task ID (e.g. ``01_02,06_05``), launches a subprocess
that loads the env, randomly initializes the scene up to ``--max-attempts``
times, and records the ``objs_info`` of every initialization where
``_check_success()`` returns False, until ``--n-scenes`` valid scenes are
collected. Saves:

- ``./tmp/gs-gym-evals/sampled_scenes/{task_id}.json`` — JSON array of the collected
  ``objs_info`` dicts.
- ``./tmp/gs-gym-evals/sampled_scenes/{task_id}/{01..N}.png`` — ego-camera render of each
  collected scene, zero-padded to two digits.

Usage:
    python scripts/robowits/data/sample_scenes.py --tasks 01_02,06_05
    python scripts/robowits/data/sample_scenes.py --tasks 06_01 --n-scenes 10 --max-attempts 50
"""

import argparse
import subprocess
import sys
from pathlib import Path

from gs_gym.envs.robowits.utils import resolve_task

OUT_DIR = Path("tmp/gs-gym-evals/sampled_scenes")

# ── per-task worker (called as subprocess) ────────────────────────────────────
WORKER_CODE = r"""
import sys, os, json, traceback
task = sys.argv[1]
out_path = sys.argv[2]
images_dir = sys.argv[3]
n_scenes = int(sys.argv[4])
max_attempts = int(sys.argv[5])
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
n_steps = int(sys.argv[7]) if len(sys.argv) > 7 else 0
skip_random_placement = bool(int(sys.argv[8])) if len(sys.argv) > 8 else False

try:
    import numpy as np
    from PIL import Image
    import gs_gym

    def to_jsonable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_jsonable(v) for v in obj]
        return obj

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    vec_env = gs_gym.make(task, n_envs=1)
    inner_env = vec_env.envs[0]

    if skip_random_placement:
        # Bypass random placement so reset uses entities' initial poses defined
        # in ``_add_custom_entities``. We still need eval-particle restoration
        # in case the env loaded a particles .npz alongside.
        inner_env._place_objects = lambda: inner_env._restore_eval_particles()

    def first_placeable_pos():
        # Pos of the first non-table entity, for diagnostic logging.
        for name, info in inner_env._entities.items():
            if name == "table":
                continue
            try:
                p = info["entity"].get_pos()[0].detach().cpu().numpy()
                return name, tuple(round(float(v), 3) for v in p)
            except Exception:
                continue
        return None, None

    scenes = []
    particle_data = {}  # entity_name -> np.ndarray (n_particles, 3), same across all scenes
    attempts = 0
    while len(scenes) < n_scenes and attempts < max_attempts:
        attempts += 1
        # Force a different random state per attempt. ``vec_env.reset(seed=...)``
        # only re-seeds gym's ``self.np_random``; the numpy RNG used by the
        # placement sampler is ``inner_env._random``, so we must seed it too.
        inner_env.seed(seed + attempts)
        vec_env.reset(seed=seed + attempts)

        success = bool(inner_env._check_success().squeeze().item())
        probe_name, probe_pos = first_placeable_pos()
        if success:
            print(
                f"  attempt {attempts}: scene already success, re-init "
                f"[{probe_name}@{probe_pos}]",
                flush=True,
            )
            continue
        objs_info = inner_env.collect_objs_info()

        # Capture canonical particle positions from first valid scene (same across
        # all scenes since particle containers are not randomized).
        # Run extra steps beyond _preprocess_steps to let fluid fully settle.
        if not particle_data:
            PARTICLE_SETTLE_STEPS = 2
            settle_frames_dir = os.path.join(images_dir, "_settle")
            os.makedirs(settle_frames_dir, exist_ok=True)
            # Frame 0000 = pre-settle state; 0001..N = after each settle step.
            frame = inner_env.render()
            Image.fromarray(np.asarray(frame).astype(np.uint8)).save(
                os.path.join(settle_frames_dir, f"{0:04d}.png")
            )
            for step_i in range(1, PARTICLE_SETTLE_STEPS + 1):
                inner_env._scene.scene.step()
                frame = inner_env.render()
                Image.fromarray(np.asarray(frame).astype(np.uint8)).save(
                    os.path.join(settle_frames_dir, f"{step_i:04d}.png")
                )
            print(
                f"  saved {PARTICLE_SETTLE_STEPS + 1} settle frames -> {settle_frames_dir}",
                flush=True,
            )
            settled_info = inner_env.collect_objs_info()
            for name, info in settled_info.items():
                if info.get("material") == "particle" and "pos" in info:
                    particle_data[name] = np.array(info["pos"])

        # Strip particle entries (stored separately) and the static table.
        rigid_objs_info = {k: v for k, v in objs_info.items() if v.get("material") != "particle" and k != "table"}
        scenes.append(to_jsonable(rigid_objs_info))

        # Print each rigid entity's pose + AABB for diagnostic / scripting use.
        print(f"  Rigid poses for attempt {attempts}:", flush=True)
        for name, info in rigid_objs_info.items():
            pos = info.get("pos")
            euler = info.get("euler")
            bounds = info.get("bounds")

            pos_str = "n/a"
            if pos is not None:
                p = np.asarray(pos, dtype=float).reshape(-1)
                if p.size >= 3:
                    pos_str = f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"

            euler_str = "n/a"
            if euler is not None:
                e = np.asarray(euler, dtype=float).reshape(-1)
                if e.size >= 3:
                    euler_str = f"({e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f})"

            aabb_str = "n/a"
            if bounds is not None:
                b = np.asarray(bounds, dtype=float)
                if b.shape == (2, 3):
                    (xmin, ymin, zmin), (xmax, ymax, zmax) = b[0], b[1]
                    aabb_str = (
                        f"min=({xmin:.3f}, {ymin:.3f}, {zmin:.3f}) "
                        f"max=({xmax:.3f}, {ymax:.3f}, {zmax:.3f}) "
                        f"size=({xmax - xmin:.3f}, {ymax - ymin:.3f}, {zmax - zmin:.3f})"
                    )

            print(
                f"    {name}: pos={pos_str} euler={euler_str} aabb={aabb_str}",
                flush=True,
            )

        # Force the camera buffer to refresh with the post-reset scene state
        # before reading; without this the render path can return a stale frame
        # across consecutive resets.
        inner_env._scene.scene.step()

        scene_idx = len(scenes)
        if n_steps > 0:
            # Save one frame per simulation step under a per-scene subdirectory.
            scene_frames_dir = os.path.join(images_dir, f"{scene_idx:02d}")
            os.makedirs(scene_frames_dir, exist_ok=True)
            frame = inner_env.render()
            Image.fromarray(np.asarray(frame).astype(np.uint8)).save(
                os.path.join(scene_frames_dir, f"{0:04d}.png")
            )
            for step_i in range(1, n_steps + 1):
                inner_env._scene.scene.step()
                frame = inner_env.render()
                Image.fromarray(np.asarray(frame).astype(np.uint8)).save(
                    os.path.join(scene_frames_dir, f"{step_i:04d}.png")
                )
            img_path = scene_frames_dir
            extra = f" + {n_steps} step frames"
        else:
            frame = inner_env.render()
            img_path = os.path.join(images_dir, f"{scene_idx:02d}.png")
            Image.fromarray(np.asarray(frame).astype(np.uint8)).save(img_path)
            extra = ""
        print(
            f"  attempt {attempts}: collected scene {scene_idx}/{n_scenes} "
            f"[{probe_name}@{probe_pos}] (img -> {img_path}{extra})",
            flush=True,
        )

    vec_env.close()

    if len(scenes) < n_scenes:
        print(
            f"WARN: only collected {len(scenes)}/{n_scenes} scenes after {attempts} attempts",
            flush=True,
        )

    with open(out_path, "w") as f:
        json.dump(scenes, f, indent=2)
    print(f"OK saved {len(scenes)} scenes -> {out_path} (+ images in {images_dir})")

    if particle_data:
        particles_path = out_path.replace(".json", "_particles.npz")
        np.savez(particles_path, **particle_data)
        sizes = {k: v.shape for k, v in particle_data.items()}
        print(f"OK saved particle data {sizes} -> {particles_path}")
except Exception:
    traceback.print_exc()
    sys.exit(1)
"""


def sample_task(
    task: str,
    out_path: Path,
    images_dir: Path,
    n_scenes: int,
    max_attempts: int,
    seed: int,
    python: str,
    n_steps: int,
    skip_random_placement: bool,
) -> bool:
    result = subprocess.run(
        [
            python,
            "-c",
            WORKER_CODE,
            task,
            str(out_path),
            str(images_dir),
            str(n_scenes),
            str(max_attempts),
            str(seed),
            str(n_steps),
            "1" if skip_random_placement else "0",
        ],
    )
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", required=True, help="Comma-separated mutation task IDs (e.g., '01_02,06_05')")
    parser.add_argument("--out", default=str(OUT_DIR), help="Output directory")
    parser.add_argument("--n-scenes", type=int, default=10, help="Number of valid scenes to collect per task")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=50,
        help="Max random initializations to try per task before giving up",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output JSONs")
    parser.add_argument(
        "--n-steps",
        type=int,
        default=0,
        help=(
            "If > 0, after each collected scene step the simulator this many times "
            "and save one PNG per step under {images_dir}/{scene:02d}/{step:04d}.png. "
            "If 0 (default), save a single PNG per scene as before."
        ),
    )
    parser.add_argument(
        "--skip-random-placement",
        action="store_true",
        help=(
            "Skip the env's random placement pass and use each entity's initial "
            "pose as defined in the env's ``_add_custom_entities``. Useful for "
            "inspecting the canonical (non-randomized) scene layout."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not task_ids:
        print("No tasks given.", file=sys.stderr)
        sys.exit(2)

    print(f"Sampling {args.n_scenes} scenes for {len(task_ids)} task(s) -> {out_dir}/")

    failed = []
    for i, tid in enumerate(task_ids, 1):
        full = resolve_task(tid.strip())
        # Output filename uses the short id as requested
        short = tid.split("/")[-1].replace("-v0", "")
        out_path = out_dir / f"{short}.json"
        images_dir = out_dir / short

        if out_path.exists() and not args.overwrite:
            print(f"[{i:3d}/{len(task_ids)}] {full} ... SKIP (exists)")
            continue

        images_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{i:3d}/{len(task_ids)}] {full} ...")
        ok = sample_task(
            full,
            out_path,
            images_dir,
            args.n_scenes,
            args.max_attempts,
            args.seed,
            args.python,
            args.n_steps,
            args.skip_random_placement,
        )
        if not ok:
            failed.append(tid)

    print(f"\nDone. {len(task_ids) - len(failed)}/{len(task_ids)} succeeded.")
    if failed:
        print("Failed tasks:")
        for t in failed:
            print(f"  {t}")
        sys.exit(1)


if __name__ == "__main__":
    main()
