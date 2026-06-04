"""RetrieveRoll environment for RoboWits.

Task: Use the long rod to retrieve the out-of-reach roll and move it to the target area.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits import utils as _utils
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv


@register_task("robowits/27-retrieve-roll-v0")
class RetrieveRollEnv(RoboWitsEnv):
    """Retrieve an out-of-reach roll using a long rod.

    The roll is placed beyond the robot's direct reach. By using the rod as
    a tool extension, the robot can extend its effective reach. Inserting
    the rod into the hollow part of the roll creates a mechanical coupling.

    Success criteria:
    - Roll overlaps target area by at least 50%
    - Roll remains within table bounds
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Use the long rod to retrieve the out-of-reach roll and move it to the green target area."

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
        far_area = {"x_min": 0.8, "x_max": 1.0, "y_min": -0.4, "y_max": 0.4}
        self._object_reachable_areas["hollow roll"] = far_area

    @property
    def placement_groups(self) -> PlacementGroups:
        """All objects placed independently."""
        return ("green target area", "long rod", "hollow roll")

    def _add_custom_entities(self) -> None:
        """Add green target area, long rod, and hollow roll."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Green target area (box, fixed, no collision)
        target_area = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.50, 0.0, 0.76 + 0.0025),
                euler=(0.0, 0.0, 0.0),
                size=(0.18, 0.18, 0.005),
                fixed=True,
                collision=False,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.1, 0.8, 0.2), roughness=0.6),
        )
        self._entities["green target area"] = {
            "entity": target_area,
        }

        # Long rod (box)
        long_rod = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.52, -0.15, 0.76 + 0.01),
                euler=(0.0, 90.0, 0.0),
                size=(0.02, 0.02, 0.40),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=50.0, friction=1.0),
            surface=gs.surfaces.Smooth(color=(0.8, 0.7, 0.5), double_sided=False),
        )
        self._entities["long rod"] = {
            "entity": long_rod,
        }

        # Hollow roll (mesh)
        roll = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/a0c77eb5-d5b7-4754-8122-3badaf242b7e/obj.glb", pattern_is_dir=False),
                scale=1.2,
                pos=(0.88, 0.0, 0.76 + 0.0695 * 1.2),
                euler=(0.0, 90.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=10.0, friction=1.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["hollow roll"] = {
            "entity": roll,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: roll on target area."""
        objs_info = self.collect_objs_info()

        TABLE_X_MIN, TABLE_X_MAX = 0.21, 1.00
        TABLE_Y_MIN, TABLE_Y_MAX = -0.69, 0.69

        def get_hull_xy(bounds, hull_2d):
            if hull_2d is not None and len(hull_2d) >= 3:
                return [(float(p[0]), float(p[1])) for p in hull_2d]
            if bounds is None:
                return None
            xmin, ymin = float(bounds[0][0]), float(bounds[0][1])
            xmax, ymax = float(bounds[1][0]), float(bounds[1][1])
            return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

        def is_within_table_bounds(poly):
            if poly is None:
                return False
            for x, y in poly:
                if x < TABLE_X_MIN - 1e-3 or x > TABLE_X_MAX + 1e-3:
                    return False
                if y < TABLE_Y_MIN - 1e-3 or y > TABLE_Y_MAX + 1e-3:
                    return False
            return True

        def polygon_area(poly):
            if poly is None or len(poly) < 3:
                return 0.0
            n = len(poly)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += poly[i][0] * poly[j][1]
                area -= poly[j][0] * poly[i][1]
            return abs(area) / 2.0

        def compute_overlap_ratio(poly_a, poly_b):
            inter_area = _utils.intersection_area(poly_a, poly_b)
            a_area = polygon_area(poly_a)
            if a_area <= 1e-9:
                return 0.0
            return inter_area / a_area

        roll = objs_info.get("hollow roll")
        target = objs_info.get("green target area")

        if roll is None or target is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        roll_hull = get_hull_xy(roll.get("bounds"), roll.get("convex_hull_2d"))
        target_hull = get_hull_xy(target.get("bounds"), target.get("convex_hull_2d"))

        if roll_hull is None or target_hull is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        if not is_within_table_bounds(roll_hull):
            return torch.tensor([False], dtype=torch.bool, device=self._device)
        if not is_within_table_bounds(target_hull):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        overlap = compute_overlap_ratio(roll_hull, target_hull)
        return torch.tensor([overlap >= 0.5], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute reward: 1.0 on success, else milestone progress in [0, 1)."""
        success = self._check_success()
        if success.item():
            reward = torch.ones(1, dtype=torch.float32, device=self._device)
            return reward, {"success": reward}

        objs_info = self.collect_objs_info()
        roll = objs_info.get("hollow roll")
        target = objs_info.get("green target area")
        rod = objs_info.get("long rod")

        zero = torch.zeros(1, dtype=torch.float32, device=self._device)
        if roll is None or target is None or rod is None:
            return zero, {"success": zero}

        def get_pos(obj):
            p = obj.get("pos")
            return np.array(p, dtype=float) if p is not None else None

        roll_pos = get_pos(roll)
        target_pos = get_pos(target)
        rod_pos = get_pos(rod)
        if roll_pos is None or target_pos is None or rod_pos is None:
            return zero, {"success": zero}

        # EE proximity to rod (stage 1: 0 → 0.2)
        APPROACH, REACH, TOUCH = 0.3, 0.15, 0.05
        right_ee = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee = self._robot.left_ee_pose[0, :3].cpu().numpy()
        min_dist_rod = min(float(np.linalg.norm(right_ee - rod_pos)), float(np.linalg.norm(left_ee - rod_pos)))
        if min_dist_rod <= TOUCH:
            ee_score = 0.2
        elif min_dist_rod <= REACH:
            ee_score = 0.1 + 0.1 * (REACH - min_dist_rod) / (REACH - TOUCH)
        elif min_dist_rod < APPROACH:
            ee_score = 0.1 * (APPROACH - min_dist_rod) / (APPROACH - REACH)
        else:
            ee_score = 0.0

        # Roll XY distance to target center (stage 2: 0.2 → 0.8)
        MAX_DIST = 0.7
        dist_to_target = float(np.linalg.norm(roll_pos[:2] - target_pos[:2]))
        dist_frac = max(0.0, min(1.0, 1.0 - dist_to_target / MAX_DIST))
        dist_score = 0.2 + 0.6 * dist_frac

        # Overlap ratio with target (stage 3: 0.8 → 1.0)
        def get_hull_xy(obj):
            hull = obj.get("convex_hull_2d")
            if hull is not None and len(hull) >= 3:
                return [(float(p[0]), float(p[1])) for p in hull]
            b = obj.get("bounds")
            if b is None:
                return None
            xmin, ymin = float(b[0][0]), float(b[0][1])
            xmax, ymax = float(b[1][0]), float(b[1][1])
            return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

        roll_hull = get_hull_xy(roll)
        target_hull = get_hull_xy(target)
        overlap_ratio = 0.0
        if roll_hull is not None and target_hull is not None:
            inter = _utils.intersection_area(
                np.array(roll_hull, dtype=float),
                np.array(target_hull, dtype=float),
            )
            roll_area = _utils.polygon_area(np.array(roll_hull, dtype=float))
            if roll_area > 1e-9:
                overlap_ratio = inter / roll_area
        overlap_score = 0.8 + 0.2 * min(1.0, overlap_ratio / 0.5)

        score = max(ee_score, dist_score, overlap_score)
        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"success": zero}
