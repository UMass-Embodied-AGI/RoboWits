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
# Particle envs (MPM dough/sand, deformables) are unusably slow on the cpu
# backend -- a single reset of task 15 took 49 min. Pass "cuda" for those.
backend = sys.argv[9] if len(sys.argv) > 9 else "cpu"
vel_tol = float(sys.argv[10]) if len(sys.argv) > 10 else 0.30
move_tol = float(sys.argv[11]) if len(sys.argv) > 11 else 0.02
settle_steps = int(sys.argv[12]) if len(sys.argv) > 12 else 20
# Fluids and granular media keep creeping long after the rigid bodies have stopped, and
# the stored cloud is reused by every scene of the task, so it has to be a settled one.
particle_settle_steps = int(sys.argv[13]) if len(sys.argv) > 13 else 60
particle_vel_tol = float(sys.argv[14]) if len(sys.argv) > 14 else 0.05
particle_move_tol = float(sys.argv[15]) if len(sys.argv) > 15 else 0.015
particle_shape_tol = float(sys.argv[16]) if len(sys.argv) > 16 else 5e-4
# Path to a cloud captured elsewhere. A family whose vessel and contents are identical (the
# 25 tasks all hold the same water in the same pitcher at the same yaw) settles the same
# cloud, so re-capturing it per task only costs time and adds a way for sampling to fail.
particles_from = sys.argv[17] if len(sys.argv) > 17 else ""

try:
    import numpy as np
    from PIL import Image
    import gs_gym
    from gs_gym.envs.robowits import utils as _utils
    from gs_gym.envs.robowits.placement import particle_frame_from_anchor, particle_transform_to_local

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

    make_kwargs = {} if backend == "cpu" else {"genesis_backend": backend, "device": backend}
    vec_env = gs_gym.make(task, n_envs=1, **make_kwargs)
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

    def rigid_only(objs_info):
        return {k: v for k, v in objs_info.items() if v.get("material") != "particle" and k != "table"}

    def moving_objects(objs_info):
        # Anything still travelling at episode start is not a usable initial state.
        out = []
        for name, info in rigid_only(objs_info).items():
            vel = np.abs(np.asarray(info.get("vel", (0.0, 0.0, 0.0)), dtype=float)).max()
            if vel > vel_tol:
                out.append(f"{name} moving (|v|={vel:.2f} > {vel_tol})")
        return out

    # Physical-validity screen. Rather than guessing what supports what from
    # AABBs (which misreads containers, shelves and leaning objects), just let
    # the scene run briefly: anything dropped in mid-air or resting on an
    # unstable pile will move. Those are exactly the scenes that look wrong and
    # that later trigger "Invalid constraint forces causing 'nan'".
    def settle_issues(objs_info):
        # Measure drift over the *second half* of the window only. Round objects
        # (marbles, funnels, a rubber duck) legitimately roll a few centimetres
        # while settling and then stop, whereas an object dropped in mid-air or
        # sliding off the table is still travelling at the end. Judging the whole
        # window conflates the two and rejects otherwise fine scenes.
        half = max(1, settle_steps // 2)
        for _ in range(half):
            inner_env._scene.scene.step()
        mid_info = inner_env.collect_objs_info()
        mid = {k: np.asarray(v["pos"], dtype=float) for k, v in rigid_only(mid_info).items() if "pos" in v}
        for _ in range(settle_steps - half):
            inner_env._scene.scene.step()
        after_info = inner_env.collect_objs_info()

        reasons = []
        for name, pos_mid in mid.items():
            info = after_info.get(name)
            if info is None or "pos" not in info:
                continue
            drift = float(np.abs(np.asarray(info["pos"], dtype=float) - pos_mid).max())
            if drift > move_tol:
                reasons.append(f"{name} still drifting {drift:.3f}m in last {settle_steps - half} steps (> {move_tol})")
        reasons.extend(moving_objects(after_info))
        return reasons, after_info

    # An object resting on the table has its AABB bottom at the table top; one that
    # fell off is a whole object-height lower, so 5 cm of slack separates the two
    # without flagging normal contact or penetration. Same threshold as
    # validate_eval_scenes.py, which screens already-saved scenes for this.
    FALL_SLACK = 0.05

    # Being at rest is not the same as being on the table: a chopstick lying on the
    # floor is perfectly settled and passes every check above. Task 26's mutation
    # scenes were captured exactly that way -- 95 of 100 had sticks off the front
    # edge -- and because the progress metric scales by the on-table fraction, those
    # scenes scored 0 before the episode began, so the eval measured nothing.
    def fallen_objects(objs_info):
        table = objs_info.get("table")
        if table is None or table.get("bounds") is None:
            return []
        top = float(np.asarray(table["bounds"], dtype=float)[1, 2])
        floor = top - FALL_SLACK
        out = []
        for name, info in rigid_only(objs_info).items():
            b = info.get("bounds")
            if b is None:
                continue
            b = np.asarray(b, dtype=float)
            if b.shape != (2, 3) or not np.isfinite(b).all():
                continue
            bottom = float(b[0, 2])
            if bottom < floor:
                out.append(f"{name} off the table (bottom z={bottom:.3f} < {floor:.3f})")
        return out

    scenes = []
    particle_data = {}  # entity_name -> np.ndarray (n_particles, 3), same across all scenes
    attempts = 0
    # Opt-in, per env class: names whose 2D hull must lie wholly inside the table
    # footprint. Default empty, so no other task's sampling changes -- several stage an
    # object overhanging the edge deliberately. Task 26 sets it because its
    # _check_success requires every chopstick within table bounds, so a scene that opens
    # with one hanging over the front edge can never be solved, and its progress score is
    # capped at on_table/6 however well the policy aligns them.
    must_stay_on = tuple(getattr(inner_env, "MUST_STAY_ON_TABLE", ()) or ())

    def uncontained_objects(objs_info):
        if not must_stay_on:
            return []
        table = objs_info.get("table")
        if table is None:
            return []
        hull = table.get("convex_hull_2d")
        tb = np.asarray(table["bounds"], dtype=float)
        if hull is not None and len(hull) >= 3:
            th = np.asarray(hull, dtype=float)
            x_min, x_max = float(th[:, 0].min()), float(th[:, 0].max())
            y_min, y_max = float(th[:, 1].min()), float(th[:, 1].max())
        else:
            x_min, x_max = float(tb[0, 0]), float(tb[1, 0])
            y_min, y_max = float(tb[0, 1]), float(tb[1, 1])
        out = []
        for name in must_stay_on:
            info = objs_info.get(name)
            if info is None:
                continue
            h = info.get("convex_hull_2d")
            b = info.get("bounds")
            if h is not None and len(h) >= 3:
                a = np.asarray(h, dtype=float)
            elif b is not None:
                a = np.asarray(b, dtype=float)[:, :2]
            else:
                continue
            if (
                a[:, 0].min() < x_min - 1e-3
                or a[:, 0].max() > x_max + 1e-3
                or a[:, 1].min() < y_min - 1e-3
                or a[:, 1].max() > y_max + 1e-3
            ):
                out.append(f"{name} overhangs the table edge (hull x {a[:, 0].min():.3f}..{a[:, 0].max():.3f})")
        return out

    rejected = {
        "already success": 0,
        "not settled": 0,
        "placement fell back": 0,
        "off the table": 0,
        "overhanging the edge": 0,
    }
    while len(scenes) < n_scenes and attempts < max_attempts:
        attempts += 1
        # Force a different random state per attempt. ``vec_env.reset(seed=...)``
        # only re-seeds gym's ``self.np_random``; the numpy RNG used by the
        # placement sampler is ``inner_env._random``, so we must seed it too.
        inner_env.seed(seed + attempts)
        vec_env.reset(seed=seed + attempts)

        # A reset whose random placement gave up yields the canonical layout, so
        # collecting it just stores the same scene N times.
        if getattr(inner_env, "placement_fell_back", False):
            rejected["placement fell back"] += 1
            print(f"  attempt {attempts}: rejected -- random placement failed, canonical layout", flush=True)
            continue

        success = bool(inner_env._check_success().squeeze().item())
        probe_name, probe_pos = first_placeable_pos()
        if success:
            rejected["already success"] += 1
            print(
                f"  attempt {attempts}: scene already success, re-init "
                f"[{probe_name}@{probe_pos}]",
                flush=True,
            )
            continue
        objs_info = inner_env.collect_objs_info()

        # Screen for physical validity, and keep the settled poses: the stored
        # scene should be the state the episode actually starts from.
        issues, objs_info = settle_issues(objs_info)
        if issues:
            rejected["not settled"] += 1
            print(f"  attempt {attempts}: rejected -- {'; '.join(issues)}", flush=True)
            continue

        fallen = fallen_objects(objs_info)
        if fallen:
            rejected["off the table"] += 1
            print(f"  attempt {attempts}: rejected -- {'; '.join(fallen)}", flush=True)
            continue

        uncontained = uncontained_objects(objs_info)
        if uncontained:
            rejected["overhanging the edge"] += 1
            print(f"  attempt {attempts}: rejected -- {'; '.join(uncontained)}", flush=True)
            continue

        # Particle clouds are stored once per task, in the cloud's own placement frame,
        # and each scene records only the rigid transform that puts it back in the world.
        # Capturing world coordinates instead (the old format) pinned the cloud to the one
        # scene it came from: every other scene started with its sand outside the jar.
        # Derive each cloud's frame from its group anchor's *settled* pose, not from the
        # transform placement applied. The stored rigid poses are post-settle, and a dynamic
        # container moves in between -- task 15's jar is nudged by the sand pouring onto it,
        # which left the cloud offset from the jar in ~15% of scenes.
        cloud_anchor = {}
        for _group in inner_env.placement_groups or ():
            if not isinstance(_group, tuple):
                continue
            _rigid = [
                n for n in _group
                if n in inner_env._entities and not _utils.is_particle_entity(inner_env._entities[n]["entity"])
            ]
            for n in _group:
                if n in inner_env._entities and _utils.is_particle_entity(inner_env._entities[n]["entity"]) and _rigid:
                    cloud_anchor[n] = _rigid[0]

        transforms = {}
        for _c in [n for n, i in inner_env._entities.items() if _utils.is_particle_entity(i["entity"])]:
            _a = cloud_anchor.get(_c)
            _c0 = inner_env._entities[_c].get("initial_pos")
            if _c0 is None:
                continue
            _placed = (inner_env._particle_transforms or {}).get(_c)
            if _a is None and _placed is not None:
                # A standalone cloud (task 08's dough, task 20's foam ball) is a placement
                # target with no group anchor, and its transform is the *only* record of
                # where it went -- falling back to the authored centroid here silently
                # pinned every scene to the same spot.
                _fp, _fq = np.asarray(_placed["pos"], dtype=float), np.asarray(_placed["quat"], dtype=float)
            elif _a is None or _a not in objs_info:
                _fp, _fq = np.asarray(_c0, dtype=float), np.array([1.0, 0.0, 0.0, 0.0])
            else:
                _fp, _fq = particle_frame_from_anchor(
                    _c0,
                    inner_env._entities[_a]["initial_pos"],
                    inner_env._entities[_a]["initial_quat"],
                    objs_info[_a]["pos"],
                    objs_info[_a]["quat"],
                )
            transforms[_c] = {"pos": [float(v) for v in _fp], "quat": [float(v) for v in _fq]}
        if particles_from and not particle_data:
            _shared = np.load(particles_from)
            for _k in _shared.files:
                if _k != "_frame":
                    particle_data[_k] = _shared[_k]
            print(f"  reusing cloud {os.path.basename(particles_from)} "
                  f"({ {k: v.shape for k, v in particle_data.items()} })", flush=True)
        if not particle_data:
            # Settle in chunks and take the first snapshot that is genuinely at rest.
            # A fixed budget cannot serve every task: 22_01 is still collapsing at step 20
            # and settles by 30, while base 22 is at rest from step 10 but goes unstable
            # after step 80 (|v| 0.02 -> 0.25). Stopping at first rest lands in both windows.
            CHUNK = 10
            captured, diverged, series = {}, {}, {}
            prev_centroid, prev_r99 = {}, {}
            for step in range(0, particle_settle_steps + 1, CHUNK):
                if step:
                    for _ in range(CHUNK):
                        inner_env._scene.scene.step()
                snap = inner_env.collect_objs_info()
                for name, info in snap.items():
                    if info.get("material") != "particle" or name in captured:
                        continue
                    w = np.asarray(info["pos"], dtype=float)
                    vv = np.asarray(info.get("vel", 0.0), dtype=float).reshape(len(w), -1)
                    if not (np.isfinite(w).all() and np.isfinite(vv).all()):
                        diverged[name] = f"non-finite state at step {step}"
                        continue
                    # Rest is a stable *shape*, not a stationary object: a fluid at rest keeps
                    # jittering (task 21 holds r99 to 1e-4 while particles run at 0.1-0.29 m/s)
                    # and an elastic ball rolls without deforming. Max speed can never pass
                    # those; the 95th percentile plus a shape test does.
                    speed = float(np.percentile(np.linalg.norm(vv, axis=1), 95))
                    cen = w.mean(axis=0)
                    r99 = float(np.percentile(np.linalg.norm(w - cen, axis=1), 99))
                    drift = float(np.abs(cen - prev_centroid[name]).max()) if name in prev_centroid else float("inf")
                    d_r99 = abs(r99 - prev_r99[name]) if name in prev_r99 else float("inf")
                    prev_centroid[name], prev_r99[name] = cen, r99
                    series.setdefault(name, []).append((step, speed, drift, d_r99))
                    if speed <= particle_vel_tol and drift <= particle_move_tol and d_r99 <= particle_shape_tol:
                        captured[name] = (w, step)
                if prev_centroid and len(captured) + len(diverged) == len(prev_centroid):
                    break

            unsettled = []
            for name, hist in series.items():
                if name in diverged:
                    unsettled.append(f"{name} {diverged[name]} -- the simulation diverged")
                    continue
                if name not in captured:
                    tail = " ".join(f"{st}:v95={sp:.3f},drift={dr:.4f},dr99={d:.5f}" for st, sp, dr, d in hist[-3:])
                    unsettled.append(f"{name} never came to rest ({tail})")
                    continue
                world, at_step = captured[name]
                extent = world.max(axis=0) - world.min(axis=0)
                # A cloud collapsed to a single point passes every rest check but is unusable;
                # three shipped 21_* files are exactly that.
                if float(extent.max()) < 1e-3:
                    unsettled.append(f"{name} collapsed to a point (extent={np.round(extent, 4).tolist()})")
                    continue
                # ...and the other end: 08_04's dough comes to rest smeared 1.5 m across the
                # scene. A stable shape is not by itself an intact one.
                table_half = np.asarray(inner_env.TABLE_SIZE[:2], dtype=float) * 0.5
                if (extent[:2] > table_half).any():
                    unsettled.append(f"{name} spread across the scene (extent={np.round(extent, 3).tolist()})")
                    continue
                t = transforms.get(name)
                if t is None:
                    unsettled.append(f"{name} was never placed, so it has no frame to store it in")
                    continue
                particle_data[name] = particle_transform_to_local(world, t["pos"], t["quat"])
                print(
                    f"  captured '{name}': {len(world)} particles, at rest by step {at_step}, "
                    f"extent={np.round(extent, 3).tolist()}",
                    flush=True,
                )
            if unsettled:
                particle_data.clear()
                rejected["particles unsettled"] = rejected.get("particles unsettled", 0) + 1
                print(f"  attempt {attempts}: rejected -- {'; '.join(unsettled)}", flush=True)
                continue

        # Rigid poses, plus a pointer + transform for each cloud. The table is static.
        rigid_objs_info = {k: v for k, v in objs_info.items() if v.get("material") != "particle" and k != "table"}
        for name in particle_data:
            if name not in transforms:
                continue
            rigid_objs_info[name] = {
                "material": "particle",
                "particles_file": (
                    os.path.relpath(particles_from, os.path.dirname(out_path))
                    if particles_from
                    else os.path.basename(out_path).replace(".json", "_particles.npz")
                ),
                "transform": transforms[name],
            }
        scenes.append(to_jsonable(rigid_objs_info))

        # Flush after every scene: an unstable placement can kill Genesis mid-run
        # (e.g. "Invalid constraint forces causing 'nan'"), and scenes collected
        # before the crash would otherwise be lost with the process.
        with open(out_path, "w") as f:
            json.dump(scenes, f, indent=2)
            f.write("\n")
        if particle_data and not particles_from:
            np.savez(out_path.replace(".json", "_particles.npz"), _frame="local", **particle_data)

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

    print(f"  rejected: {rejected} (attempts={attempts}/{max_attempts})", flush=True)
    if len(scenes) < n_scenes:
        print(
            f"WARN: only collected {len(scenes)}/{n_scenes} scenes after {attempts} attempts",
            flush=True,
        )

    with open(out_path, "w") as f:
        json.dump(scenes, f, indent=2)
        f.write("\n")
    print(f"OK saved {len(scenes)} scenes -> {out_path} (+ images in {images_dir})")

    if particle_data and not particles_from:
        particles_path = out_path.replace(".json", "_particles.npz")
        np.savez(particles_path, _frame="local", **particle_data)
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
    backend: str,
    vel_tol: float,
    move_tol: float,
    settle_steps: int,
    particle_settle_steps: int,
    particle_vel_tol: float,
    particle_move_tol: float,
    particle_shape_tol: float,
    particles_from: str,
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
            backend,
            str(vel_tol),
            str(move_tol),
            str(settle_steps),
            str(particle_settle_steps),
            str(particle_vel_tol),
            str(particle_move_tol),
            str(particle_shape_tol),
            particles_from,
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
        help="Hard cap on random initializations per task, so a task that can never "
        "produce a usable scene cannot loop forever",
    )
    parser.add_argument(
        "--vel-tol",
        type=float,
        default=0.30,
        help="Reject a scene if any rigid object moves faster than this at episode start (m/s)",
    )
    parser.add_argument(
        "--move-tol",
        type=float,
        default=0.02,
        help="Reject a scene if any rigid object drifts more than this while settling (m)",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=20,
        help="Simulation steps used to test that the scene is at rest",
    )
    parser.add_argument(
        "--particle-settle-steps",
        type=int,
        default=120,
        help="Hard cap on settling steps. Capture stops at the first at-rest snapshot; the "
        "cloud is reused by every scene of the task, so it must be at rest",
    )
    parser.add_argument(
        "--particle-vel-tol",
        type=float,
        default=0.15,
        help="Max 95th-percentile particle speed for 'at rest' (m/s)",
    )
    parser.add_argument(
        "--particle-move-tol",
        type=float,
        default=0.015,
        help="Max particle-cloud centroid drift per 10-step chunk for 'at rest' (m)",
    )
    parser.add_argument(
        "--particle-shape-tol",
        type=float,
        default=5e-4,
        help="Max change in the cloud's 99th-percentile radius per chunk (m)",
    )
    parser.add_argument(
        "--particles-from",
        default="",
        help="Reuse this .npz instead of capturing a cloud per task. Valid when the tasks build "
        "identical contents in an identical vessel, which is what makes a family shareable",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument(
        "--backend",
        default="cpu",
        help="Genesis backend. Particle envs (08, 15, 20, 21, 22, 30) need 'cuda' to run in reasonable time",
    )
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
            args.backend,
            args.vel_tol,
            args.move_tol,
            args.settle_steps,
            args.particle_settle_steps,
            args.particle_vel_tol,
            args.particle_move_tol,
            args.particle_shape_tol,
            args.particles_from,
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
