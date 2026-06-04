"""WaterIntoMug environment for RoboWits.

Task: Collect water in the mug without moving the pitcher.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv


def mug_pos_relative_to_pitcher(
    px: float,
    py: float,
    pz: float,
    pitcher_yaw_deg: float,
    *,
    body_dx: float = -0.18,
    body_dy: float = 0.0,
    dz: float = -0.0695,
) -> tuple[float, float, float]:
    """World-frame mug position from pitcher center and yaw (morph euler Z, degrees).

    The canonical scene uses body-frame offset ``(body_dx, body_dy)=(-0.18, 0)``
    (meters in the pitcher XY plane); rotated into world by ``pitcher_yaw_deg``
    so the mug stays on the same geometric side of the pitcher when the mesh yaw
    changes. Vertical offset ``dz`` is applied along world Z.

    At ``pitcher_yaw_deg=-90`` this matches hand-authored ``(px, py+d, pz+dz)`` with
    ``d≈0.18`` in world +Y for this task's default pitcher placement.
    """
    psi = np.deg2rad(float(pitcher_yaw_deg))
    c, s = np.cos(psi), np.sin(psi)
    wx = body_dx * c - body_dy * s
    wy = body_dx * s + body_dy * c
    return (float(px + wx), float(py + wy), float(pz + dz))


@register_task("robowits/25-water-into-mug-v0")
class WaterIntoMugEnv(RoboWitsEnv):
    """Collect water in the mug using displacement.

    The pitcher is fixed and cannot be tilted to pour water directly.
    By dropping the heavy large object into the pitcher, the robot
    exploits Archimedes' principle of water displacement to raise the
    water level and cause overflow into the mug.

    Success criteria:
    - At least 1 water particle is inside the mug's interior (circular proxy, z within mug bounds)
    - Mug remains within table XY bounds
    - Mug is resting on the table (not floating)
    - At least 70% of water particles remain over the table
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Collect water in the mug without moving the pitcher."

    TABLE_Z = 0.76

    def __init__(
        self,
        config_name: str = "robowits_default",
        n_envs: int = 1,
        show_viewer: bool = False,
        max_episode_steps: int = 200,
        control_mode: str = "EE_ABS",
        **kwargs,
    ):
        # Initialize per-object reachable areas
        self._object_reachable_areas: dict[str, dict[str, float]] = {}

        super().__init__(
            config_name=config_name,
            n_envs=n_envs,
            show_viewer=show_viewer,
            max_episode_steps=max_episode_steps,
            control_mode=control_mode,
            **kwargs,
        )

        # Set reachable areas for objects
        big_area = {"x_min": 0.0, "x_max": 0.75, "y_min": -0.7, "y_max": 0.7}
        self._object_reachable_areas["heavy_large_object"] = big_area

    def _config_to_env_args(self, config):
        args = super()._config_to_env_args(config)
        args = args.model_copy(update={"scene_args": args.scene_args.model_copy(update={"substeps": 100})})
        return args

    @property
    def placement_groups(self) -> PlacementGroups:
        """Only heavy object is randomized."""
        return ("heavy_large_object",)

    def _add_custom_entities(self) -> None:
        """Add pitcher, mug, and heavy large object."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Pitcher (mesh, fixed) — mug XY offset uses same yaw (morph euler Z, degrees)
        _pitcher_yaw_deg = -90.0
        pitcher = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/pitcher.glb", pattern_is_dir=False),
                scale=1.525,
                pos=(0.50, -0.03, 0.9515 - 0.08),
                euler=(0.0, 0.0, _pitcher_yaw_deg),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=800.0, sdf_min_res=128, sdf_max_res=256),
            surface=gs.surfaces.Smooth(color=(0.8, 0.8, 0.85), double_sided=True, vis_mode="visual"),
        )
        self._entities["pitcher"] = {
            "entity": pitcher,
        }
        # Pitcher-centered blockers (aligned with mutation/25_04.py geometry).
        _px, _py = 0.50, -0.03
        self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(_px, _py + 0.02, 0.82),
                radius=0.11,
                height=0.05,
                fixed=True,
                visualization=False,
                collision=True,
            ),
        )
        for rot in [0, 30, 60, 120, 150, 180, 210, 240, 270, 300, 330]:
            sin_rot, cos_rot = np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot))
            self._scene.scene.add_entity(
                gs.morphs.Box(
                    pos=(_px + 0.09 * cos_rot, _py + 0.015 + 0.09 * sin_rot, 0.97 - 0.08),
                    euler=(0.0, 0.0, rot),
                    size=(0.01, 0.05, 0.15),
                    fixed=True,
                    visualization=False,
                    collision=True,
                ),
            )
        sin_rot, cos_rot = np.sin(np.deg2rad(90)), np.cos(np.deg2rad(90))
        self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(_px + 0.09 * cos_rot, _py + 0.015 + 0.09 * sin_rot, 0.95 - 0.08),
                euler=(0.0, 0.0, 90.0),
                size=(0.01, 0.05, 0.1),
                fixed=True,
                visualization=False,
                collision=True,
            ),
        )

        # Mug (mesh) — XY offset from pitcher rotates with pitcher yaw (euler Z)
        mug = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/94134b02-c73e-4f2c-ad1d-a00a78160d98/obj.glb", pattern_is_dir=False),
                scale=0.85,
                pos=mug_pos_relative_to_pitcher(0.50, -0.03, 0.8715, _pitcher_yaw_deg),
                euler=(0.0, 0.0, 180.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=1200.0),
            surface=gs.surfaces.Smooth(color=(0.9, 0.75, 0.6), double_sided=True, vis_mode="visual"),
        )
        self._entities["mug"] = {
            "entity": mug,
        }

        # Heavy large object (mesh)
        heavy = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/rockman.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.68, -0.32, 0.803),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500.0, friction=0.8),
            surface=gs.surfaces.Rough(color=(0.5, 0.5, 0.5), double_sided=True, vis_mode="visual"),
        )
        self._entities["heavy_large_object"] = {
            "entity": heavy,
        }

        # Water (SPH liquid inside pitcher)
        water = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.50, -0.02, 0.8715 + 0.2 - 0.08),
                euler=(0.0, 0.0, 0.0),
                radius=0.06,
                height=0.3,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.SPH.Liquid(
                rho=100.0, stiffness=5000.0, exponent=7.0, mu=0.01, gamma=0.02, sampler="pbs"
            ),
            surface=gs.surfaces.Glass(color=(0.6, 0.85, 1.0), double_sided=True, vis_mode="recon"),
        )
        self._entities["water"] = {
            "entity": water,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: mug contains water (particle-based)."""
        objs_info = self.collect_objs_info()

        def _is_completed(objs_info) -> bool:
            """Checks if the task is completed: The mug contains water (particle-based).
            Also ensures involved objects (mug and water) have not fallen off the table.
            """

            # Workspace/table bounds
            TABLE_X_MIN, TABLE_X_MAX = 0.21, 1.00
            TABLE_Y_MIN, TABLE_Y_MAX = -0.69, 0.69
            EDGE_TOL = 0.02  # small tolerance for boundary checks

            # Required objects
            if "mug" not in objs_info or "water" not in objs_info:
                return False

            mug = objs_info["mug"]
            water = objs_info["water"]

            # Basic validity checks
            if mug.get("bounds") is None or water.get("pos") is None:
                return False

            # Derive a conservative approximation of the mug's interior:
            # - Use mug AABB to estimate a circular opening centered at the mug's xy center
            # - Radius is a fraction of the smallest xy extent to avoid counting handle/outside
            try:
                (xmin, ymin, zmin), (xmax, ymax, zmax) = mug["bounds"]
            except Exception:
                return False

            cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
            dx, dy = max(0.0, xmax - xmin), max(0.0, ymax - ymin)
            if dx <= 0.0 or dy <= 0.0:
                return False

            # Circular interior proxy to avoid including handle/outer hull
            r = 0.4 * min(dx, dy)
            if r <= 0:
                return False

            # Vertical bounds for being inside the mug (avoid counting below base or above rim)
            z_low = zmin + 0.01
            z_high = zmax - 0.005
            if z_high <= z_low:
                return False

            # Water particle positions
            wpos = np.array(water["pos"])
            if wpos.ndim != 2 or wpos.shape[1] != 3:
                return False
            N = wpos.shape[0]

            # Count water particles within the mug interior proxy
            if N > 0:
                dxv = wpos[:, 0] - cx
                dyv = wpos[:, 1] - cy
                radial_ok = (dxv * dxv + dyv * dyv) <= (r * r)
                vertical_ok = (wpos[:, 2] >= z_low) & (wpos[:, 2] <= z_high)
                in_mug = radial_ok & vertical_ok
                collected = int(np.count_nonzero(in_mug))
            else:
                collected = 0

            # Success threshold: some amount of water is in the mug
            threshold = 1
            collected_ok = collected >= threshold

            # Mug must remain within table xy bounds (via convex hull if available)
            hull = mug.get("convex_hull_2d", None)
            if hull is not None:
                hull = np.array(hull)
                if hull.ndim == 2 and hull.shape[1] == 2 and hull.shape[0] > 0:
                    mug_on_table = (
                        np.all(hull[:, 0] >= (TABLE_X_MIN - EDGE_TOL))
                        and np.all(hull[:, 0] <= (TABLE_X_MAX + EDGE_TOL))
                        and np.all(hull[:, 1] >= (TABLE_Y_MIN - EDGE_TOL))
                        and np.all(hull[:, 1] <= (TABLE_Y_MAX + EDGE_TOL))
                    )
                else:
                    mug_on_table = (
                        xmin >= (TABLE_X_MIN - EDGE_TOL)
                        and xmax <= (TABLE_X_MAX + EDGE_TOL)
                        and ymin >= (TABLE_Y_MIN - EDGE_TOL)
                        and ymax <= (TABLE_Y_MAX + EDGE_TOL)
                    )
            else:
                mug_on_table = (
                    xmin >= (TABLE_X_MIN - EDGE_TOL)
                    and xmax <= (TABLE_X_MAX + EDGE_TOL)
                    and ymin >= (TABLE_Y_MIN - EDGE_TOL)
                    and ymax <= (TABLE_Y_MAX + EDGE_TOL)
                )

            # Mug must be resting on (or near) the table surface, not floating
            TABLE_Z = 0.76
            MUG_FLOAT_TOL = 0.05  # allow up to 5 cm above table for physics settling
            mug_on_surface = zmin <= TABLE_Z + MUG_FLOAT_TOL

            # Majority of water particles should remain over the table xy region
            if N > 0:
                inside_x = (wpos[:, 0] >= (TABLE_X_MIN - EDGE_TOL)) & (wpos[:, 0] <= (TABLE_X_MAX + EDGE_TOL))
                inside_y = (wpos[:, 1] >= (TABLE_Y_MIN - EDGE_TOL)) & (wpos[:, 1] <= (TABLE_Y_MAX + EDGE_TOL))
                frac_on_table = float(np.count_nonzero(inside_x & inside_y)) / N
            else:
                frac_on_table = 1.0
            water_not_fallen = frac_on_table >= 0.7  # allow some spill but not majority off-table

            return bool(collected_ok and mug_on_table and mug_on_surface and water_not_fallen)

        return torch.tensor([_is_completed(objs_info)], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute dense reward based on milestone-based progress score."""
        objs_info = self.collect_objs_info()

        HEAVY_OBJ = "heavy_large_object"
        PITCHER = "pitcher"
        MUG = "mug"

        REACH_THRESHOLD = 0.15
        TOUCH_THRESHOLD = 0.05
        APPROACH_THRESHOLD = 0.4

        INITIAL_HEAVY_POS = np.array([0.68, -0.32, 0.793])
        PITCHER_POS = np.array([0.50, -0.03, 0.8715])

        if MUG not in objs_info:
            score_t = torch.tensor(0.0, dtype=torch.float32, device=self._device)
            return score_t, {"score": score_t}

        def get_pos(name):
            obj = objs_info.get(name)
            if obj is None:
                return None
            p = obj.get("pos")
            if p is None:
                return None
            if isinstance(p, np.ndarray) and p.ndim == 2:
                return np.mean(p, axis=0)
            return np.array([p[0], p[1], p[2]], dtype=float)

        def get_bounds(name):
            obj = objs_info.get(name)
            if obj is None:
                return None
            b = obj.get("bounds")
            if b is None:
                return None
            return np.array(b, dtype=float)

        def aabb_intersection_ratio(bounds_a, bounds_b):
            if bounds_a is None or bounds_b is None:
                return 0.0
            try:
                a = np.array(bounds_a, dtype=float)
                b = np.array(bounds_b, dtype=float)
            except Exception:
                return 0.0
            if a.shape != (2, 3) or b.shape != (2, 3):
                return 0.0
            ixmin = max(a[0, 0], b[0, 0])
            iymin = max(a[0, 1], b[0, 1])
            izmin = max(a[0, 2], b[0, 2])
            ixmax = min(a[1, 0], b[1, 0])
            iymax = min(a[1, 1], b[1, 1])
            izmax = min(a[1, 2], b[1, 2])
            if ixmax <= ixmin or iymax <= iymin or izmax <= izmin:
                return 0.0
            inter_vol = (ixmax - ixmin) * (iymax - iymin) * (izmax - izmin)
            a_vol = max(0.0, a[1, 0] - a[0, 0]) * max(0.0, a[1, 1] - a[0, 1]) * max(0.0, a[1, 2] - a[0, 2])
            if a_vol <= 0.0:
                return 0.0
            return float(inter_vol / a_vol)

        right_ee_pos = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee_pos = self._robot.left_ee_pose[0, :3].cpu().numpy()

        heavy_score = 0.0
        if HEAVY_OBJ in objs_info:
            heavy_pos = get_pos(HEAVY_OBJ)
            heavy_bounds = get_bounds(HEAVY_OBJ)
            pitcher_bounds = get_bounds(PITCHER)

            if heavy_pos is not None:
                dist_right = float(np.linalg.norm(right_ee_pos - heavy_pos))
                dist_left = float(np.linalg.norm(left_ee_pos - heavy_pos))
                min_dist_to_heavy = min(dist_right, dist_left)

                intersection_ratio = aabb_intersection_ratio(heavy_bounds, pitcher_bounds)

                if intersection_ratio >= 0.50:
                    heavy_score = 1.0
                else:
                    height_gain = max(0.0, heavy_pos[2] - INITIAL_HEAVY_POS[2])
                    dist_to_pitcher_xy = float(np.linalg.norm(heavy_pos[:2] - PITCHER_POS[:2]))
                    moving_toward_pitcher = dist_to_pitcher_xy < 0.15

                    if height_gain > 0.1 or moving_toward_pitcher:
                        height_progress = min(1.0, height_gain / 0.2)
                        position_progress = max(0.0, 1.0 - dist_to_pitcher_xy / 0.3)
                        combined = max(height_progress, position_progress)
                        heavy_score = 0.4 + combined * 0.4
                    elif min_dist_to_heavy <= TOUCH_THRESHOLD:
                        height_progress = min(1.0, height_gain / 0.05) if height_gain > 0 else 0.0
                        heavy_score = 0.4 + height_progress * 0.2
                    elif min_dist_to_heavy <= REACH_THRESHOLD:
                        reach_progress = 1.0 - (min_dist_to_heavy - TOUCH_THRESHOLD) / (
                            REACH_THRESHOLD - TOUCH_THRESHOLD
                        )
                        reach_progress = max(0.0, min(1.0, reach_progress))
                        heavy_score = 0.2 + reach_progress * 0.2
                    elif min_dist_to_heavy < APPROACH_THRESHOLD:
                        approach_progress = 1.0 - (min_dist_to_heavy - REACH_THRESHOLD) / (
                            APPROACH_THRESHOLD - REACH_THRESHOLD
                        )
                        approach_progress = max(0.0, min(1.0, approach_progress))
                        heavy_score = approach_progress * 0.2

        mug_score = 0.0
        mug_pos = get_pos(MUG)
        if mug_pos is not None:
            dist_right_to_mug = float(np.linalg.norm(right_ee_pos - mug_pos))
            dist_left_to_mug = float(np.linalg.norm(left_ee_pos - mug_pos))
            min_dist_to_mug = min(dist_right_to_mug, dist_left_to_mug)

            if min_dist_to_mug <= TOUCH_THRESHOLD:
                mug_score = 0.3
            elif min_dist_to_mug <= REACH_THRESHOLD:
                reach_progress = 1.0 - (min_dist_to_mug - TOUCH_THRESHOLD) / (REACH_THRESHOLD - TOUCH_THRESHOLD)
                reach_progress = max(0.0, min(1.0, reach_progress))
                mug_score = 0.15 + reach_progress * 0.15
            elif min_dist_to_mug < APPROACH_THRESHOLD:
                approach_progress = 1.0 - (min_dist_to_mug - REACH_THRESHOLD) / (APPROACH_THRESHOLD - REACH_THRESHOLD)
                approach_progress = max(0.0, min(1.0, approach_progress))
                mug_score = approach_progress * 0.15

        score = max(heavy_score, mug_score)
        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"score": score_t}
