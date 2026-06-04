"""StackBowls environment for RoboWits.

Task: Nest the smaller bowl concentrically inside the larger one.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv


@register_task("robowits/19-stack-bowls-v0")
class StackBowlsEnv(RoboWitsEnv):
    """Nest the smaller bowl concentrically inside the larger one.

    Concentric alignment leverages circular geometry; matching rims
    provide a strong geometric signal for proper nesting.

    Success criteria:
    - Medium bowl's 2D center is within 1cm of large bowl's center
    - Both bowls remain within table bounds
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Nest the smaller bowl concentrically inside the larger one."

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
        smaller_area = {"x_min": 0.1, "x_max": 0.7, "y_min": -0.45, "y_max": 0.45}
        self._object_reachable_areas["large bowl"] = smaller_area
        self._object_reachable_areas["medium bowl"] = smaller_area

    @property
    def placement_groups(self) -> PlacementGroups:
        """Both bowls placed independently."""
        return ("large bowl", "medium bowl")

    def _add_custom_entities(self) -> None:
        """Add large and medium bowls."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Large bowl (mesh)
        large_bowl = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c652cf0f-d2eb-44bd-9e68-a2ceca698591/obj.glb", pattern_is_dir=False),
                scale=1.688,
                pos=(0.605, 0.0, 0.8204),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["large bowl"] = {
            "entity": large_bowl,
        }

        # Medium bowl (mesh)
        medium_bowl = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c652cf0f-d2eb-44bd-9e68-a2ceca698591/obj.glb", pattern_is_dir=False),
                scale=1.227,
                pos=(0.40, 0.10, 0.8042),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["medium bowl"] = {
            "entity": medium_bowl,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: bowls nested concentrically."""
        objs_info = self.collect_objs_info()

        TABLE_X_MIN, TABLE_X_MAX = 0.21, 1.00
        TABLE_Y_MIN, TABLE_Y_MAX = -0.69, 0.69
        TABLE_TOL = 0.01
        RADIAL_TOL = 0.01
        NUM_TOL = 1e-3

        def get_obj(name):
            obj = objs_info.get(name, None)
            if obj is None:
                return None
            return obj

        def get_center_2d(obj):
            hull = obj.get("convex_hull_2d", None)
            if hull is not None and isinstance(hull, np.ndarray) and hull.size >= 2:
                c = np.mean(hull, axis=0)
                if np.all(np.isfinite(c)):
                    return c
            bounds = obj.get("bounds", None)
            if bounds is not None and isinstance(bounds, np.ndarray) and bounds.shape == (2, 3):
                (xmin, ymin, _), (xmax, ymax, _) = bounds
                return np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0], dtype=float)
            pos = obj.get("pos", None)
            if pos is not None and isinstance(pos, np.ndarray) and pos.size >= 2:
                return pos[:2].astype(float)
            return None

        def is_on_table(obj):
            hull = obj.get("convex_hull_2d", None)
            if hull is not None and isinstance(hull, np.ndarray) and hull.size >= 2:
                xs = hull[:, 0]
                ys = hull[:, 1]
                if np.any(xs < TABLE_X_MIN - TABLE_TOL) or np.any(xs > TABLE_X_MAX + TABLE_TOL):
                    return False
                return not (np.any(ys < TABLE_Y_MIN - TABLE_TOL) or np.any(ys > TABLE_Y_MAX + TABLE_TOL))
            bounds = obj.get("bounds", None)
            if bounds is not None and isinstance(bounds, np.ndarray) and bounds.shape == (2, 3):
                (xmin, ymin, _), (xmax, ymax, _) = bounds
                if xmin < TABLE_X_MIN - TABLE_TOL or xmax > TABLE_X_MAX + TABLE_TOL:
                    return False
                return not (ymin < TABLE_Y_MIN - TABLE_TOL or ymax > TABLE_Y_MAX + TABLE_TOL)
            pos = obj.get("pos", None)
            if pos is not None and isinstance(pos, np.ndarray) and pos.size >= 2:
                x, y = float(pos[0]), float(pos[1])
                if x < TABLE_X_MIN - TABLE_TOL or x > TABLE_X_MAX + TABLE_TOL:
                    return False
                return not (y < TABLE_Y_MIN - TABLE_TOL or y > TABLE_Y_MAX + TABLE_TOL)
            return False

        large = get_obj("large bowl")
        medium = get_obj("medium bowl")
        if large is None or medium is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        if not (is_on_table(large) and is_on_table(medium)):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        c_large = get_center_2d(large)
        c_medium = get_center_2d(medium)
        if c_large is None or c_medium is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        radial_offset = np.linalg.norm(c_large - c_medium)

        return torch.tensor([bool(radial_offset <= (RADIAL_TOL + NUM_TOL))], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute reward: 1.0 on success, else milestone progress in [0, 1)."""
        success = self._check_success()
        if success.item():
            reward = torch.ones(1, dtype=torch.float32, device=self._device)
            return reward, {"success": reward}

        objs_info = self.collect_objs_info()
        large = objs_info.get("large bowl")
        medium = objs_info.get("medium bowl")

        zero = torch.zeros(1, dtype=torch.float32, device=self._device)
        if large is None or medium is None:
            return zero, {"success": zero}

        def get_center_2d(obj):
            hull = obj.get("convex_hull_2d")
            if hull is not None and isinstance(hull, np.ndarray) and hull.size >= 2:
                c = np.mean(hull, axis=0)
                if np.all(np.isfinite(c)):
                    return c[:2]
            b = obj.get("bounds")
            if b is not None:
                return np.array([(b[0][0] + b[1][0]) / 2.0, (b[0][1] + b[1][1]) / 2.0])
            p = obj.get("pos")
            return np.array(p[:2], dtype=float) if p is not None else None

        medium_pos = medium.get("pos")
        medium_pos = np.array(medium_pos, dtype=float) if medium_pos is not None else None

        # EE proximity to medium bowl (stage 1: 0 → 0.2)
        APPROACH, REACH, TOUCH = 0.3, 0.15, 0.05
        right_ee = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee = self._robot.left_ee_pose[0, :3].cpu().numpy()
        if medium_pos is not None:
            min_dist = min(
                float(np.linalg.norm(right_ee - medium_pos[:3])), float(np.linalg.norm(left_ee - medium_pos[:3]))
            )
            if min_dist <= TOUCH:
                ee_score = 0.2
            elif min_dist <= REACH:
                ee_score = 0.1 + 0.1 * (REACH - min_dist) / (REACH - TOUCH)
            elif min_dist < APPROACH:
                ee_score = 0.1 * (APPROACH - min_dist) / (APPROACH - REACH)
            else:
                ee_score = 0.0
        else:
            ee_score = 0.0

        # XY distance between bowl centers (stage 2: 0.2 → 1.0)
        c_large = get_center_2d(large)
        c_medium = get_center_2d(medium)
        if c_large is not None and c_medium is not None:
            dist = float(np.linalg.norm(c_large - c_medium))
            dist_frac = max(0.0, min(1.0, 1.0 - dist / 0.3))
            dist_score = 0.2 + 0.8 * dist_frac
        else:
            dist_score = 0.0

        score = max(ee_score, dist_score)
        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"success": zero}
