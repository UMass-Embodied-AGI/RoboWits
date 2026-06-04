"""StackCubes environment for RoboWits.

Task: Build a stable two-layer stack so the red cube's center is above the green marker.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv
from gs_gym.envs.robowits.utils import check_being_held


@register_task("robowits/14-stack-cubes-v0")
class StackCubesEnv(RoboWitsEnv):
    """Build a stable two-layer stack so the red cube is above the green marker.

    Two base cubes must be placed touching each other, and the red apex cube
    must be placed on top of them, centered over the green target marker.

    Success criteria:
    - Base cubes are on the table and touching with >= 50% face overlap
    - Apex cube sits on top of base cubes
    - Apex cube center is within tolerance of target marker
    - Apex-target overlap >= 20%
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Build a stable two-layer stack so the red cube's center is above the green colored dot."

    TABLE_Z = 0.76
    CUBE_SIZE = (0.05, 0.05, 0.05)

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
        target_area = {"x_min": 0.4, "x_max": 0.5, "y_min": -0.2, "y_max": 0.2}
        smaller_reachable_area = {"x_min": 0.3, "x_max": 0.6, "y_min": -0.3, "y_max": 0.3}
        self._object_reachable_areas["base cube 1"] = smaller_reachable_area
        self._object_reachable_areas["base cube 2"] = smaller_reachable_area
        self._object_reachable_areas["apex cube"] = smaller_reachable_area
        self._object_reachable_areas["apex target"] = target_area

    @property
    def placement_groups(self) -> PlacementGroups:
        """All objects placed independently."""
        return ("base cube 1", "base cube 2", "apex cube", "apex target")

    def _add_custom_entities(self) -> None:
        """Add cubes and target marker."""
        # Apex target (cylinder marker)
        target = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.605, 0.0, 0.761),
                euler=(0.0, 0.0, 0.0),
                radius=0.015,
                height=0.002,
                fixed=True,
                collision=False,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.8),
            surface=gs.surfaces.Default(color=(0.1, 0.8, 0.2), roughness=0.4),
        )
        self._entities["apex target"] = {
            "entity": target,
        }

        # Base cube 1
        base1 = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.52, -0.05, 0.785),
                euler=(0.0, 0.0, 0.0),
                size=self.CUBE_SIZE,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1.0),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5),
        )
        self._entities["base cube 1"] = {
            "entity": base1,
        }

        # Base cube 2
        base2 = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.68, 0.05, 0.785),
                euler=(0.0, 0.0, 0.0),
                size=self.CUBE_SIZE,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1.0),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5),
        )
        self._entities["base cube 2"] = {
            "entity": base2,
        }

        # Apex cube (red)
        apex = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.58, 0.12, 0.785),
                euler=(0.0, 0.0, 0.0),
                size=self.CUBE_SIZE,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1.0),
            surface=gs.surfaces.Default(color=(0.9, 0.1, 0.1), roughness=0.4),
        )
        self._entities["apex cube"] = {
            "entity": apex,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful."""
        objs_info = self.collect_objs_info()

        b1 = objs_info.get("base cube 1")
        b2 = objs_info.get("base cube 2")
        apex = objs_info.get("apex cube")
        target = objs_info.get("apex target")

        if any(x is None for x in [b1, b2, apex, target]):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        def is_on_table_surface(bounds, table_z, tolerance=0.015):
            z_min = float(bounds[0][2])
            return abs(z_min - table_z) <= tolerance

        def objects_touching(b1, b2, tolerance=0.01):
            return all(not (b1[1][i] + tolerance < b2[0][i] or b2[1][i] + tolerance < b1[0][i]) for i in range(3))

        if not is_on_table_surface(b1["bounds"], self.TABLE_Z):
            return torch.tensor([False], dtype=torch.bool, device=self._device)
        if not is_on_table_surface(b2["bounds"], self.TABLE_Z):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        if not objects_touching(b1["bounds"], b2["bounds"], tolerance=0.01):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        apex_bottom = float(apex["bounds"][0, 2])
        b1_top = float(b1["bounds"][1, 2])
        if apex_bottom < b1_top - 0.01:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        apex_pos = apex["pos"][:2]
        target_pos = target["pos"][:2]
        dist = np.sqrt((apex_pos[0] - target_pos[0]) ** 2 + (apex_pos[1] - target_pos[1]) ** 2)
        if dist > 0.025:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        if check_being_held(apex, self._robot):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        return torch.tensor([True], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute reward: 1.0 on success, else milestone progress in [0, 1)."""
        success = self._check_success()
        if success.item():
            reward = torch.ones(1, dtype=torch.float32, device=self._device)
            return reward, {"success": reward}

        objs_info = self.collect_objs_info()
        b1 = objs_info.get("base cube 1")
        b2 = objs_info.get("base cube 2")
        apex = objs_info.get("apex cube")
        target = objs_info.get("apex target")

        zero = torch.zeros(1, dtype=torch.float32, device=self._device)
        if any(x is None for x in [b1, b2, apex, target]):
            return zero, {"success": zero}
        if any(x.get("bounds") is None for x in [b1, b2, apex]):
            return zero, {"success": zero}

        b1_bounds = np.array(b1["bounds"], dtype=float)
        b2_bounds = np.array(b2["bounds"], dtype=float)
        apex_bounds = np.array(apex["bounds"], dtype=float)
        b1_pos = np.array(b1["pos"][:3], dtype=float)
        b2_pos = np.array(b2["pos"][:3], dtype=float)
        apex_pos = np.array(apex["pos"][:3], dtype=float)
        target_pos = np.array(target["pos"][:3], dtype=float)

        # EE proximity to any cube (0 → 0.2)
        APPROACH, REACH, TOUCH = 0.3, 0.15, 0.05
        right_ee = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee = self._robot.left_ee_pose[0, :3].cpu().numpy()
        min_dist = min(
            min(float(np.linalg.norm(right_ee - p)), float(np.linalg.norm(left_ee - p)))
            for p in [b1_pos, b2_pos, apex_pos]
        )
        if min_dist <= TOUCH:
            ee_score = 0.2
        elif min_dist <= REACH:
            ee_score = 0.1 + 0.1 * (REACH - min_dist) / (REACH - TOUCH)
        elif min_dist < APPROACH:
            ee_score = 0.1 * (APPROACH - min_dist) / (APPROACH - REACH)
        else:
            ee_score = 0.0

        # Base cubes approaching each other (0.2 → 0.5)
        base_dist = float(np.linalg.norm(b1_pos[:2] - b2_pos[:2]))
        approach_frac = max(0.0, min(1.0, 1.0 - base_dist / 0.3))
        base_approach_score = 0.2 + 0.3 * approach_frac

        # Base cubes touching and near target (0.5 → 0.7)
        bases_touching = all(
            not (b1_bounds[1, i] + 0.01 < b2_bounds[0, i] or b2_bounds[1, i] + 0.01 < b1_bounds[0, i]) for i in range(3)
        )
        midpoint = (b1_pos[:2] + b2_pos[:2]) / 2.0
        target_dist = float(np.linalg.norm(midpoint - target_pos[:2]))
        target_frac = max(0.0, min(1.0, 1.0 - target_dist / 0.2))
        base_placed_score = 0.5 + 0.2 * target_frac if bases_touching else 0.0

        # Apex on top of base cubes and aligned over target (0.7 → 1.0)
        b1_top = b1_bounds[1, 2]
        apex_bottom = apex_bounds[0, 2]
        apex_on_top = apex_bottom >= b1_top - 0.01 and bases_touching
        apex_xy_dist = float(np.linalg.norm(apex_pos[:2] - target_pos[:2]))
        apex_align_frac = max(0.0, min(1.0, 1.0 - apex_xy_dist / 0.1))
        apex_score = (0.7 + 0.3 * apex_align_frac) if apex_on_top else 0.0

        score = max(ee_score, base_approach_score, base_placed_score, apex_score)
        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"success": zero}
