"""StackCubes environment for RoboWits.

Task: Build a stable two-layer stack so the red cube's center is above the green marker.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv
from gs_gym.envs.robowits.utils import (
    STAGE_PROGRESS,
    approach_score,
    at_goal_score,
    check_released,
    check_still,
    progress_frac,
)


@register_task("robowits/14-stack-cubes-v0")
class StackCubesEnv(RoboWitsEnv):
    """Build a stable two-layer stack so the red cube is above the green marker.

    The check tests only the goal, not how it is reached: the gray cubes are there to raise the red cube, but
    any support that holds it above the marker counts (one gray cube, both side by side, ...).

    Success criteria:
    - Red cube's bottom is at least RAISE_MIN above the marker's top (raised off the table, not resting on the marker)
    - Red cube's center is within TARGET_XY_TOL of the marker's center in the table plane
    - Red cube is not held by either gripper
    - Red cube is still (utils.check_still)
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Build a stable two-layer stack so the red cube's center is above the green colored dot."

    TABLE_Z = 0.76
    CUBE_SIZE = (0.05, 0.05, 0.05)

    # The red cube counts as raised when its bottom is this far above the marker's top. Resting on a gray cube
    # (4.5 cm in 14_01, 5 cm elsewhere) clears it with margin; resting on the table, the marker or a thin
    # distractor (coaster 8 mm, sunglasses 1.5 cm) does not.
    RAISE_MIN = 0.035
    # Red cube center to marker center, in the table plane.
    TARGET_XY_TOL = 0.025

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
        self._object_reachable_areas["base_cube_1"] = smaller_reachable_area
        self._object_reachable_areas["base_cube_2"] = smaller_reachable_area
        self._object_reachable_areas["apex_cube"] = smaller_reachable_area
        self._object_reachable_areas["apex_target"] = target_area

    @property
    def placement_groups(self) -> PlacementGroups:
        """All objects placed independently."""
        return ("base_cube_1", "base_cube_2", "apex_cube", "apex_target")

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
        self._entities["apex_target"] = {
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
        self._entities["base_cube_1"] = {
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
        self._entities["base_cube_2"] = {
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
        self._entities["apex_cube"] = {
            "entity": apex,
        }

    def _apex_raised(self, apex: dict, target: dict) -> bool:
        """Whether the red cube's bottom is at least RAISE_MIN above the marker's top."""
        return float(apex["bounds"][0][2]) >= float(target["bounds"][1][2]) + self.RAISE_MIN

    def _apex_xy_dist(self, apex: dict, target: dict) -> float:
        """Red cube center to marker center, in the table plane."""
        return float(np.linalg.norm(np.asarray(apex["pos"][:2], dtype=float) - np.asarray(target["pos"][:2])))

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: the red cube rests, raised and released, above the green marker."""
        objs_info = self.collect_objs_info(["apex_cube", "apex_target"])
        apex = objs_info.get("apex_cube")
        target = objs_info.get("apex_target")
        if apex is None or target is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        success = (
            self._apex_raised(apex, target)
            and self._apex_xy_dist(apex, target) <= self.TARGET_XY_TOL
            and check_released(apex, self._robot)
            and check_still(apex)
        )
        return torch.tensor([success], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Dense reward on the shared ladder (utils.STAGE_*), each stage a condition of the success check.

        Like the success check, the stages reward the goal, not a particular way of building the stack.
        0 -> 0.2     a gripper approaches a cube
        0.2 -> 0.4   (added to the approach score) the red cube moves toward the marker, from where it spawned
        0.45         a support stands under the marker (any object whose top clears RAISE_MIN, its footprint over
                     the marker's center)
        0.5 -> 0.65  the red cube is raised but not yet centered, scored by its distance to the marker
        0.7          the goal geometry is met (raised and centered) but the red cube is still held
        0.85         the goal geometry is met and the red cube released, but still moving
        1.0          success
        """
        if self._check_success().item():
            reward = torch.ones(1, dtype=torch.float32, device=self._device)
            return reward, {"success": reward}

        objs_info = self.collect_objs_info()
        apex = objs_info.get("apex_cube")
        target = objs_info.get("apex_target")

        zero = torch.zeros(1, dtype=torch.float32, device=self._device)
        if apex is None or target is None or apex.get("bounds") is None or target.get("bounds") is None:
            return zero, {"success": zero}

        cubes = [objs_info[n]["pos"] for n in ("base_cube_1", "base_cube_2", "apex_cube") if n in objs_info]
        score = approach_score(self._robot, cubes)

        spawn = self._entities.get("apex_cube", {}).get("spawn_pos")
        frac = progress_frac(apex["pos"], spawn, target["pos"], done_dist=self.TARGET_XY_TOL)
        score += 0.2 * frac  # on top of the approach score

        # A support under the marker, whatever it is built from (within 1 cm, so two cubes meeting over it count)
        target_top = float(target["bounds"][1][2])
        tx, ty = (float(v) for v in target["pos"][:2])
        for name, info in objs_info.items():
            if name in ("apex_cube", "apex_target", "table") or info.get("bounds") is None:
                continue
            b = np.asarray(info["bounds"], dtype=float)
            if b.shape != (2, 3) or b[1, 2] < target_top + self.RAISE_MIN:
                continue
            if b[0, 0] - 0.01 <= tx <= b[1, 0] + 0.01 and b[0, 1] - 0.01 <= ty <= b[1, 1] + 0.01:
                score = max(score, 0.45)
                break

        if self._apex_raised(apex, target):
            dist = self._apex_xy_dist(apex, target)
            if dist <= self.TARGET_XY_TOL:
                score = max(score, at_goal_score(check_released(apex, self._robot)))
            else:
                align = max(0.0, 1.0 - (dist - self.TARGET_XY_TOL) / 0.1)
                score = max(score, 0.5 + (STAGE_PROGRESS - 0.5) * align)

        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"success": zero}
