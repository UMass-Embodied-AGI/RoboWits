"""RetrieveRoll environment for RoboWits.

Task: Place the out-of-reach roll on the target area, using the long rod to reach it.
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
    """Place an out-of-reach roll on the target area using a long rod.

    The roll is placed beyond the robot's direct reach. By using the rod as
    a tool extension, the robot can extend its effective reach. Inserting
    the rod into the hollow part of the roll creates a mechanical coupling.

    Success criteria:
    - Roll overlaps target area by at least 50%
    - Roll rests on the target area (its lowest point within CONTACT_Z_TOL of the pad's top), not above it
    - Roll is released (utils.check_released) and still (utils.check_still)
    - Roll remains within table bounds
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Place the roll on the green target area."

    TABLE_Z = 0.76
    # How far the roll's lowest collision point may sit above the target pad's top and still count as resting on it.
    # The pad has no collision geometry, so this stands in for contact.
    CONTACT_Z_TOL = 0.01
    OVERLAP_MIN = 0.5  # share of the roll's footprint that must lie over the target
    TABLE_XY = (0.21, 1.00, -0.69, 0.69)  # x_min, x_max, y_min, y_max

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
        self._object_reachable_areas["hollow_roll"] = far_area

    @property
    def placement_groups(self) -> PlacementGroups:
        """All objects placed independently."""
        return ("green_target_area", "long_rod", "hollow_roll")

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
        self._entities["green_target_area"] = {
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
        self._entities["long_rod"] = {
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
        self._entities["hollow_roll"] = {
            "entity": roll,
        }

    def _footprint(self, obj: dict) -> np.ndarray | None:
        """An object's outline in the table plane: its convex hull, else its box's footprint."""
        hull = obj.get("convex_hull_2d")
        if hull is not None and len(hull) >= 3:
            return np.asarray(hull, dtype=float)[:, :2]
        bounds = obj.get("bounds")
        if bounds is None:
            return None
        (xmin, ymin, _), (xmax, ymax, _) = np.asarray(bounds, dtype=float)
        return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])

    def _roll_overlap(self, roll: dict, target: dict) -> float:
        """Share of the roll's footprint over the target; 0 if either footprint leaves the table."""
        roll_fp, target_fp = self._footprint(roll), self._footprint(target)
        if roll_fp is None or target_fp is None:
            return 0.0
        x_min, x_max, y_min, y_max = self.TABLE_XY
        for fp in (roll_fp, target_fp):
            if (
                fp[:, 0].min() < x_min - 1e-3
                or fp[:, 0].max() > x_max + 1e-3
                or fp[:, 1].min() < y_min - 1e-3
                or fp[:, 1].max() > y_max + 1e-3
            ):
                return 0.0
        roll_area = _utils.polygon_area(roll_fp)
        if roll_area <= 1e-9:
            return 0.0
        return _utils.intersection_area(roll_fp, target_fp) / roll_area

    def _roll_on_pad(self, roll: dict, target: dict) -> bool:
        """Whether the roll's lowest point is at the pad's top (the pad has no collision geometry)."""
        roll_b, target_b = roll.get("bounds"), target.get("bounds")
        if roll_b is None or target_b is None:
            return False
        return float(roll_b[0][2]) <= float(target_b[1][2]) + self.CONTACT_Z_TOL

    def _roll_at_goal(self, roll: dict, target: dict) -> bool:
        """The goal geometry: the roll lies on the pad, at least OVERLAP_MIN of its footprint over it."""
        return self._roll_overlap(roll, target) >= self.OVERLAP_MIN and self._roll_on_pad(roll, target)

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: roll lies on the target area, released and still."""
        objs_info = self.collect_objs_info(["hollow_roll", "green_target_area"])
        roll, target = objs_info.get("hollow_roll"), objs_info.get("green_target_area")
        if roll is None or target is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)
        success = (
            self._roll_at_goal(roll, target) and _utils.check_released(roll, self._robot) and _utils.check_still(roll)
        )
        return torch.tensor([success], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Dense reward on the shared ladder (utils.STAGE_*), each stage a condition of the success check.

        0 -> 0.2     a gripper approaches the rod or the roll
        0.2 -> 0.5   (added to the approach score) the roll moves toward the target, from where it spawned
        0.5 -> 0.65  the roll's footprint covers the target, up to OVERLAP_MIN (at any height)
        0.7          the goal geometry is met (lying on the pad, overlapping) but the roll is still held
        0.85         the goal geometry is met and the roll released, but still moving
        1.0          success
        """
        if self._check_success().item():
            reward = torch.ones(1, dtype=torch.float32, device=self._device)
            return reward, {"success": reward}

        objs_info = self.collect_objs_info()
        roll, target, rod = objs_info.get("hollow_roll"), objs_info.get("green_target_area"), objs_info.get("long_rod")
        zero = torch.zeros(1, dtype=torch.float32, device=self._device)
        if roll is None or target is None or roll.get("pos") is None or target.get("pos") is None:
            return zero, {"success": zero}

        score = _utils.approach_score(self._robot, [roll["pos"], rod.get("pos") if rod is not None else None])

        # Progress ends once the roll's center is within half the target's width of the target's center
        target_fp = self._footprint(target)
        done_dist = 0.5 * float(min(np.ptp(target_fp[:, 0]), np.ptp(target_fp[:, 1]))) if target_fp is not None else 0.0
        spawn = self._entities.get("hollow_roll", {}).get("spawn_pos")
        frac = _utils.progress_frac(roll["pos"], spawn, target["pos"], done_dist=done_dist)
        score += 0.3 * frac  # on top of the approach score

        overlap = self._roll_overlap(roll, target)
        if overlap > 0.0:
            score = max(score, 0.5 + (_utils.STAGE_PROGRESS - 0.5) * min(1.0, overlap / self.OVERLAP_MIN))
        if self._roll_at_goal(roll, target):
            score = max(score, _utils.at_goal_score(_utils.check_released(roll, self._robot)))

        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"success": zero}
