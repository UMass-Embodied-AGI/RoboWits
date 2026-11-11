"""MoveCube environment for RoboWits.

Task: Move any cube beyond the goal line using the slopes.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv
from gs_gym.envs.robowits.utils import (
    STAGE_APPROACH,
    STAGE_PROGRESS,
    approach_score,
    at_goal_score,
    check_released,
    check_still,
)


@register_task("robowits/28-move-cube-v0")
class MoveCubeEnv(RoboWitsEnv):
    """Move a cube beyond the goal line using slopes with different friction.

    This task requires the robot to understand and exploit the friction
    properties of different surfaces. The smooth slope has very low friction,
    while the rough slope has very high friction.

    Success criteria, for at least one cube:
    - Its center is past the goal line (x > GOAL_X)
    - It is over the table: its outline within the table and its bottom not below the tabletop, so a cube that
      fell off the far edge does not count (resting on another object past the line does)
    - It is released (utils.check_released) and still (utils.check_still)
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Move any cube beyond the goal line."

    TABLE_Z = 0.76
    GOAL_X = 0.9  # goal line, in every variant
    START_X = 0.4  # cubes spawn around x = 0.4-0.5; progress runs from this when a cube's spawn is unknown
    TABLE_XY = (0.21, 1.00, -0.69, 0.69)  # x_min, x_max, y_min, y_max
    TABLE_TOL = 1e-3
    BELOW_TABLE_TOL = 0.01  # a cube's bottom this far under the tabletop has left the table

    def __init__(
        self,
        config_name: str = "robowits_default",
        n_envs: int = 1,
        show_viewer: bool = False,
        max_episode_steps: int = 200,
        control_mode: str = "EE_ABS",
        **kwargs,
    ):
        super().__init__(
            config_name=config_name,
            n_envs=n_envs,
            show_viewer=show_viewer,
            max_episode_steps=max_episode_steps,
            control_mode=control_mode,
            **kwargs,
        )

    @property
    def placement_groups(self) -> PlacementGroups:
        """Only cubes are randomized."""
        return ("cube_1", "cube_2", "cube_3")

    def _add_custom_entities(self) -> None:
        """Add three cubes, goal line, and two slopes."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Cube 1 (red)
        cube1 = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.40, -0.10, 0.78),
                euler=(0.0, 0.0, 0.0),
                size=(0.04, 0.04, 0.04),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.01),
            surface=gs.surfaces.Default(color=(0.9, 0.2, 0.2), roughness=0.5),
        )
        self._entities["cube_1"] = {
            "entity": cube1,
        }

        # Cube 2 (green)
        cube2 = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.44, 0.00, 0.78),
                euler=(0.0, 0.0, 0.0),
                size=(0.04, 0.04, 0.04),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.01),
            surface=gs.surfaces.Default(color=(0.2, 0.9, 0.2), roughness=0.5),
        )
        self._entities["cube_2"] = {
            "entity": cube2,
        }

        # Cube 3 (blue)
        cube3 = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.48, 0.10, 0.78),
                euler=(0.0, 0.0, 0.0),
                size=(0.04, 0.04, 0.04),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.01),
            surface=gs.surfaces.Default(color=(0.2, 0.2, 0.9), roughness=0.5),
        )
        self._entities["cube_3"] = {
            "entity": cube3,
        }

        # Goal line (box, fixed, no collision)
        goal_line = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.9, 0.0, 0.761),
                euler=(0.0, 0.0, 0.0),
                size=(0.01, 1.30, 0.002),
                fixed=True,
                collision=False,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.5),
            surface=gs.surfaces.Default(color=(1.0, 0.9, 0.1), roughness=0.3),
        )
        self._entities["goal_line"] = {
            "entity": goal_line,
        }

        # Smooth slope (mesh, fixed)
        smooth_slope = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/ab77c172-c69f-41b5-956a-600ea7b64c73/obj.glb", pattern_is_dir=False),
                scale=0.64,
                pos=(0.66, -0.16, 0.7934),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.01),
            surface=gs.surfaces.Smooth(color=(0.8, 0.8, 0.85), double_sided=True),
        )
        self._entities["smooth_slope"] = {
            "entity": smooth_slope,
        }

        # Rough slope (mesh, fixed)
        rough_slope = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/ab77c172-c69f-41b5-956a-600ea7b64c73/obj.glb", pattern_is_dir=False),
                scale=0.64,
                pos=(0.66, 0.16, 0.7934),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1.5),
            surface=gs.surfaces.Rough(color=(0.5, 0.4, 0.3), double_sided=True),
        )
        self._entities["rough_slope"] = {
            "entity": rough_slope,
        }

    def _over_table(self, cube: dict) -> bool:
        """Whether the cube is over the table: outline within its area and bottom not below the tabletop."""
        bounds = cube.get("bounds")
        if bounds is None:
            return False
        x_min, x_max, y_min, y_max = self.TABLE_XY
        hull = cube.get("convex_hull_2d")
        pts = np.asarray(hull, dtype=float) if hull is not None and len(hull) >= 3 else np.asarray(bounds)[:, :2]
        inside = (
            pts[:, 0].min() >= x_min - self.TABLE_TOL
            and pts[:, 0].max() <= x_max + self.TABLE_TOL
            and pts[:, 1].min() >= y_min - self.TABLE_TOL
            and pts[:, 1].max() <= y_max + self.TABLE_TOL
        )
        return bool(inside) and float(bounds[0][2]) >= self.TABLE_Z - self.BELOW_TABLE_TOL

    def _cubes(self, objs_info: dict) -> list[dict]:
        """The rigid cubes present this step."""
        cubes = [objs_info.get(n) for n in ("cube_1", "cube_2", "cube_3")]
        return [c for c in cubes if c is not None and c.get("material") == "rigid" and c.get("pos") is not None]

    def _cube_succeeds(self, cube: dict) -> bool:
        """Whether this cube alone satisfies the task: past the line, over the table, released and still."""
        return (
            float(cube["pos"][0]) > self.GOAL_X
            and self._over_table(cube)
            and check_released(cube, self._robot)
            and check_still(cube)
        )

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: some cube rests past the goal line, on the table, released and still."""
        objs_info = self.collect_objs_info(["cube_1", "cube_2", "cube_3"])
        success = any(self._cube_succeeds(c) for c in self._cubes(objs_info))
        return torch.tensor([success], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Dense reward on the shared ladder (utils.STAGE_*), each stage a condition of the success check.

        Scored per cube, the best cube counting:
        0 -> 0.2     a gripper approaches a cube
        0.2 -> 0.65  (added to the approach score) a cube still over the table moves toward the line, from where it spawned (a cube that fell
                     off scores nothing)
        0.7          past the line, over the table, but still held
        0.85         past the line, over the table, released, but still moving
        1.0          success
        """
        if self._check_success().item():
            reward = torch.ones(1, dtype=torch.float32, device=self._device)
            return reward, {"success": reward}

        objs_info = self.collect_objs_info()
        zero = torch.zeros(1, dtype=torch.float32, device=self._device)
        cubes = [
            (name, objs_info.get(name))
            for name in ("cube_1", "cube_2", "cube_3")
            if objs_info.get(name) is not None
            and objs_info[name].get("material") == "rigid"
            and objs_info[name].get("pos") is not None
        ]
        if not cubes:
            return zero, {"success": zero}

        approach = approach_score(self._robot, [c["pos"] for _, c in cubes])
        best = approach
        for name, cube in cubes:
            if not self._over_table(cube):
                continue  # fell off: no longer counts toward the goal
            x = float(cube["pos"][0])
            spawn = self._entities.get(name, {}).get("spawn_pos")
            x0 = min(float(spawn[0]), self.GOAL_X - 0.05) if spawn is not None else self.START_X
            frac = max(0.0, min(1.0, (x - x0) / (self.GOAL_X - x0)))
            best = max(best, approach + (STAGE_PROGRESS - STAGE_APPROACH) * frac)  # on top of the approach score
            if x > self.GOAL_X:
                best = max(best, at_goal_score(check_released(cube, self._robot)))

        score_t = torch.tensor(best, dtype=torch.float32, device=self._device)
        return score_t, {"success": zero}
