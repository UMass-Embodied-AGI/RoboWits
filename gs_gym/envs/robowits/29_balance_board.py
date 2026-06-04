"""BalanceBoard environment for RoboWits.

Task: Place the board onto the wall and keep it balanced.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv
from gs_gym.envs.robowits.utils import check_being_held


@register_task("robowits/29-balance-board-v0")
class BalanceBoardEnv(RoboWitsEnv):
    """Balance a board on a narrow wall support.

    The goal requires balancing a board on a narrow support. The key insight
    is to recognize that the board must be placed with its center of mass
    directly above the wall to maintain equilibrium.

    Success criteria:
    - Board is resting on top of the wall near its center
    - Board is balanced (nearly horizontal, not tipping/moving)
    - Objects remain within table bounds
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Place the board onto the wall and keep it balanced."

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
        """Small wall independent, board and support cube grouped."""
        return ("small wall", ("board", "support cube"))

    def _add_custom_entities(self) -> None:
        """Add small wall, board, and support cube."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Small wall (box, fixed)
        small_wall = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.31, 0.0, 0.76 + 0.12 / 2),
                euler=(0.0, 0.0, 0.0),
                size=(0.02, 0.30, 0.12),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=400.0),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.6),
        )
        self._entities["small wall"] = {
            "entity": small_wall,
        }

        # Board (mesh)
        board = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/f923466c-c37d-4744-b197-6089b2899715/obj.glb", pattern_is_dir=False),
                scale=0.8,
                pos=(0.535, 0.0, 0.76 + 0.0272 * 0.8 / 2 + 0.03),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=100),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["board"] = {
            "entity": board,
        }

        # Support cube (box, fixed)
        support_cube = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.515, 0.0, 0.78),
                euler=(0.0, 0.0, 0.0),
                size=(0.04, 0.04, 0.04),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1),
            surface=gs.surfaces.Default(color=(0.95, 0.95, 0.95), roughness=0.5),
        )
        self._entities["support cube"] = {
            "entity": support_cube,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: board balanced on wall."""
        objs_info = self.collect_objs_info()

        TABLE_X_MIN, TABLE_X_MAX = 0.21, 1.00
        TABLE_Y_MIN, TABLE_Y_MAX = -0.69, 0.69

        board = objs_info.get("board")
        wall = objs_info.get("small wall")

        if board is None or wall is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        if board.get("material") != "rigid" or wall.get("material") != "rigid":
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        if board.get("bounds") is None or wall.get("bounds") is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        def centroid_xy(hull, bounds):
            if hull is not None and isinstance(hull, np.ndarray) and hull.size >= 2:
                return np.mean(hull, axis=0)
            bmin, bmax = bounds
            return np.array([(bmin[0] + bmax[0]) / 2.0, (bmin[1] + bmax[1]) / 2.0])

        def horiz_extents(bounds):
            bmin, bmax = bounds
            return float(bmax[0] - bmin[0]), float(bmax[1] - bmin[1])

        def within_table(hull):
            if hull is None or not isinstance(hull, np.ndarray) or hull.size < 2:
                return False
            xs, ys = hull[:, 0], hull[:, 1]
            return (
                np.all(xs >= TABLE_X_MIN)
                and np.all(xs <= TABLE_X_MAX)
                and np.all(ys >= TABLE_Y_MIN)
                and np.all(ys <= TABLE_Y_MAX)
            )

        def angle_close_to_horizontal(euler, thr_deg=12.0):
            if euler is None:
                return True
            roll = float(euler[0]) * 180.0 / np.pi
            pitch = float(euler[1]) * 180.0 / np.pi

            def dist_to_0_or_180(a):
                a = a % 360.0
                return min(abs(a), abs(180.0 - a))

            return (dist_to_0_or_180(roll) <= thr_deg) and (dist_to_0_or_180(pitch) <= thr_deg)

        b_bounds = board["bounds"]
        w_bounds = wall["bounds"]
        bmin, bmax = b_bounds
        wmin, wmax = w_bounds

        board_hull = board.get("convex_hull_2d")
        wall_hull = wall.get("convex_hull_2d")

        if not (within_table(board_hull) and within_table(wall_hull)):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        wall_top_z = float(wmax[2])
        board_bottom_z = float(bmin[2])
        tol_z_contact = 0.02
        if abs(board_bottom_z - wall_top_z) > tol_z_contact:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        if (board_bottom_z - self.TABLE_Z) <= 0.03:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        board_xy = centroid_xy(board_hull, b_bounds)
        wall_xy = centroid_xy(wall_hull, w_bounds)
        center_dist = float(np.linalg.norm(board_xy - wall_xy))

        wx_ext, wy_ext = horiz_extents(w_bounds)
        wall_foot_min = max(1e-6, min(wx_ext, wy_ext))
        center_tol = max(0.03, 0.5 * wall_foot_min)
        center_tol = min(center_tol, 0.05)

        if center_dist > center_tol:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        euler = board.get("euler")
        if not angle_close_to_horizontal(euler, thr_deg=12.0):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        vel = board.get("vel")
        if vel is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        vel = np.array(vel).astype(float)
        if np.linalg.norm(vel) > 0.05:
            return torch.tensor([False], dtype=torch.bool, device=self._device)
        if abs(vel[2]) > 0.04:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        if check_being_held(board, self._robot):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        return torch.tensor([True], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute reward: 1.0 on success, else milestone progress in [0, 1)."""
        success = self._check_success()
        if success.item():
            reward = torch.ones(1, dtype=torch.float32, device=self._device)
            return reward, {"success": reward}

        objs_info = self.collect_objs_info()
        board = objs_info.get("board")
        wall = objs_info.get("small wall")

        zero = torch.zeros(1, dtype=torch.float32, device=self._device)
        if board is None or wall is None:
            return zero, {"success": zero}
        if board.get("bounds") is None or wall.get("bounds") is None:
            return zero, {"success": zero}

        b_bounds = np.asarray(board["bounds"], dtype=float)
        w_bounds = np.asarray(wall["bounds"], dtype=float)
        board_bottom_z = float(b_bounds[0, 2])
        wall_top_z = float(w_bounds[1, 2])

        # EE proximity to board (stage 1: 0 → 0.2)
        APPROACH, REACH, TOUCH = 0.3, 0.15, 0.05
        right_ee = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee = self._robot.left_ee_pose[0, :3].cpu().numpy()
        board_pos = board.get("pos")
        if board_pos is not None:
            ref = np.array(board_pos[:3], dtype=float)
            min_dist = min(float(np.linalg.norm(right_ee - ref)), float(np.linalg.norm(left_ee - ref)))
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

        # Board lift toward wall top (stage 2: 0.2 → 0.5)
        target_z = wall_top_z
        lift_frac = max(0.0, min(1.0, (board_bottom_z - self.TABLE_Z) / max(target_z - self.TABLE_Z, 1e-6)))
        lift_score = 0.2 + 0.3 * lift_frac

        # Board XY center near wall XY center (stage 3: 0.5 → 0.8), gated on lift
        wall_pos = wall.get("pos")
        align_score = 0.0
        align_frac = 0.0
        if board_pos is not None and wall_pos is not None and lift_frac > 0.5:
            b_xy = np.array(board_pos[:2], dtype=float)
            w_xy = np.array(wall_pos[:2], dtype=float)
            dist_xy = float(np.linalg.norm(b_xy - w_xy))
            align_frac = max(0.0, min(1.0, 1.0 - dist_xy / 0.3))
            align_score = 0.5 + 0.3 * align_frac

        # Board angle close to horizontal (stage 4: 0.8 → 1.0), gated on lift + align + not held
        angle_score = 0.0
        not_held = not check_being_held(board, self._robot)
        if lift_frac > 0.5 and align_frac > 0.3 and not_held:
            euler = board.get("euler")
            if euler is not None:
                roll = float(euler[0]) * 180.0 / np.pi
                pitch = float(euler[1]) * 180.0 / np.pi

                def dist_0_or_180(a):
                    a = a % 360.0
                    return min(abs(a), abs(180.0 - a))

                angle_err = max(dist_0_or_180(roll), dist_0_or_180(pitch))
                angle_frac = max(0.0, min(1.0, 1.0 - angle_err / 12.0))
                angle_score = 0.8 + 0.2 * angle_frac

        score = max(ee_score, lift_score, align_score, angle_score)
        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"success": zero}
