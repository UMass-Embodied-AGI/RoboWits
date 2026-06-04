"""MoveCube environment for RoboWits.

Task: Move any cube past the goal line using the slopes.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv


@register_task("robowits/28-move-cube-v0")
class MoveCubeEnv(RoboWitsEnv):
    """Move a cube past the goal line using slopes with different friction.

    This task requires the robot to understand and exploit the friction
    properties of different surfaces. The smooth slope has very low friction,
    while the rough slope has very high friction.

    Success criteria:
    - At least one cube has x > 0.9 (passes the goal line)
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Move any of the three cubes to the left of the goal line."

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
        return ("cube 1", "cube 2", "cube 3")

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
        self._entities["cube 1"] = {
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
        self._entities["cube 2"] = {
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
        self._entities["cube 3"] = {
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
        self._entities["goal line"] = {
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
        self._entities["smooth slope"] = {
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
        self._entities["rough slope"] = {
            "entity": rough_slope,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: any cube past goal line."""
        objs_info = self.collect_objs_info()

        cube_names = ["cube 1", "cube 2", "cube 3"]

        any_past_goal = False
        for cube_name in cube_names:
            cube = objs_info.get(cube_name)
            if cube is None:
                continue
            if cube.get("material") != "rigid":
                continue
            pos = cube.get("pos")
            if pos is not None:
                pos = np.asarray(pos)
                if pos.shape == (3,) and pos[0] > 0.9:
                    any_past_goal = True
                    break

        return torch.tensor([any_past_goal], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute reward: 1.0 on success, else milestone progress in [0, 1)."""
        success = self._check_success()
        if success.item():
            reward = torch.ones(1, dtype=torch.float32, device=self._device)
            return reward, {"success": reward}

        objs_info = self.collect_objs_info()
        zero = torch.zeros(1, dtype=torch.float32, device=self._device)

        cube_names = ["cube 1", "cube 2", "cube 3"]
        cube_infos = [objs_info.get(n) for n in cube_names]
        cube_poses = []
        for c in cube_infos:
            if c is None:
                continue
            p = c.get("pos")
            if p is not None:
                cube_poses.append(np.asarray(p, dtype=float))

        if not cube_poses:
            return zero, {"success": zero}

        # EE proximity to any cube (stage 1: 0 → 0.2)
        APPROACH, REACH, TOUCH = 0.3, 0.15, 0.05
        right_ee = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee = self._robot.left_ee_pose[0, :3].cpu().numpy()
        min_dist = min(
            min(float(np.linalg.norm(right_ee - p[:3])), float(np.linalg.norm(left_ee - p[:3]))) for p in cube_poses
        )
        if min_dist <= TOUCH:
            ee_score = 0.2
        elif min_dist <= REACH:
            ee_score = 0.1 + 0.1 * (REACH - min_dist) / (REACH - TOUCH)
        elif min_dist < APPROACH:
            ee_score = 0.1 * (APPROACH - min_dist) / (APPROACH - REACH)
        else:
            ee_score = 0.0

        # Best cube x-position toward goal line x=0.9 (stage 2: 0.2 → 1.0)
        # Cubes start around x=0.4–0.5; success at x > 0.9
        START_X, GOAL_X = 0.4, 0.9
        best_x = max(float(p[0]) for p in cube_poses)
        x_frac = max(0.0, min(1.0, (best_x - START_X) / (GOAL_X - START_X)))
        x_score = 0.2 + 0.8 * x_frac

        score = max(ee_score, x_score)
        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"success": zero}
