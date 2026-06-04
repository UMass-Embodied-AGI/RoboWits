"""RollUpBall environment for RoboWits.

Task: Move a box into a basket using an inclined plane.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv


@register_task("robowits/05-roll-up-ball-v0")
class RollUpBallEnv(RoboWitsEnv):
    """Move a box into a basket using an inclined plane.

    The box is too heavy to lift directly. The robot must use the inclined
    plane (a simple machine formed by the board and basket edge) to reduce
    the required force. This utilizes the box's sliding property to smoothly
    ascend the ramp surface.

    Success criteria:
    - Box is completely inside the basket's bounding box
    - Box and basket are on the table
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Move the box into the basket."

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
        """All objects placed independently."""
        return ("long board", "box", "high basket")

    @property
    def _object_reachable_areas(self) -> dict:
        # 'long board' AABB x-size slightly exceeds the default 0.400-wide x-bounds,
        # so widen just enough to allow valid placement sampling.
        return {"long board": {"x_min": 0.25, "x_max": 0.75}}

    def _add_custom_entities(self) -> None:
        """Add basket, board, and box."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # High basket (mesh) - fixed
        basket = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c6009731-c1d9-48f9-9486-1d5754c336d9/obj.glb", pattern_is_dir=False),
                scale=0.611,
                pos=(0.595, 0.04, 0.806),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["high basket"] = {
            "entity": basket,
        }

        # Long board (mesh)
        board = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/08893dc4-bfb1-49ca-9e47-4b3958a21e4b/obj.glb", pattern_is_dir=False),
                scale=(0.1567, 0.214, 0.151),
                pos=(0.50, -0.15, 0.77),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["long board"] = {
            "entity": board,
        }

        # Box
        box = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.395, 0.105, 0.76 + 0.05),
                euler=(0, 0, 0),
                size=(0.1, 0.1, 0.1),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1),
            surface=gs.surfaces.Smooth(color=(1.0, 0.5, 0.0), double_sided=True),
        )
        self._entities["box"] = {
            "entity": box,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: box inside basket."""
        objs_info = self.collect_objs_info()

        box = objs_info.get("box")
        basket = objs_info.get("high basket")

        if box is None or basket is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        basket_bounds = basket.get("bounds")
        if basket_bounds is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        box_pos = box.get("pos")
        if box_pos is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        box_bounds = box.get("bounds")
        if box_bounds is not None:
            r_x = (float(box_bounds[1][0]) - float(box_bounds[0][0])) / 2.0
            r_y = (float(box_bounds[1][1]) - float(box_bounds[0][1])) / 2.0
            r_z = (float(box_bounds[1][2]) - float(box_bounds[0][2])) / 2.0
            r = max(r_x, r_y, r_z)
        else:
            r = 0.05

        bmin = basket_bounds[0]
        bmax = basket_bounds[1]
        tol = 0.01

        x_ok = (box_pos[0] - r) >= (bmin[0] + tol) and (box_pos[0] + r) <= (bmax[0] - tol)
        y_ok = (box_pos[1] - r) >= (bmin[1] + tol) and (box_pos[1] + r) <= (bmax[1] - tol)
        z_ok = (box_pos[2] - r) >= (bmin[2] + tol) and (box_pos[2] + r) <= (bmax[2] - tol)

        return torch.tensor([bool(x_ok and y_ok and z_ok)], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute dense reward based on milestone-based progress score."""
        objs_info = self.collect_objs_info()

        REACH_THRESHOLD = 0.15
        TOUCH_THRESHOLD = 0.05
        TILT_THRESHOLD_DEG = 5.0
        LIFT_HEIGHT = 0.05
        APPROACH_THRESHOLD = 0.3

        BOARD = "long board"
        BASKET = "high basket"
        BOX = "box"

        INITIAL_BOARD_POS = np.array([0.50, -0.15, 0.77], dtype=float)
        INITIAL_BOX_POS = np.array([0.395, 0.105, 0.76 + 0.05], dtype=float)
        INITIAL_BOX_BOARD_DIST = float(np.linalg.norm(INITIAL_BOX_POS[:2] - INITIAL_BOARD_POS[:2]))

        def get_pos(name):
            obj = objs_info.get(name)
            if obj is None:
                return None
            p = obj.get("pos")
            if p is None:
                return None
            return np.array([p[0], p[1], p[2]], dtype=float)

        def get_bounds(name):
            obj = objs_info.get(name)
            if obj is None:
                return None
            b = obj.get("bounds")
            if b is None:
                return None
            return np.array(b, dtype=float)

        def get_euler(name):
            obj = objs_info.get(name)
            if obj is None:
                return None
            e = obj.get("euler")
            if e is None:
                return None
            return np.array([e[0], e[1], e[2]], dtype=float)

        def bounds_intersect_xy(b1, b2):
            if b1 is None or b2 is None:
                return False
            tol = 0.01
            return (
                b1[1, 0] >= b2[0, 0] - tol
                and b2[1, 0] >= b1[0, 0] - tol
                and b1[1, 1] >= b2[0, 1] - tol
                and b2[1, 1] >= b1[0, 1] - tol
            )

        def xy_overlap_progress(b1, b2):
            if b1 is None or b2 is None:
                return 0.0
            x_overlap = max(0.0, min(b1[1, 0], b2[1, 0]) - max(b1[0, 0], b2[0, 0]))
            y_overlap = max(0.0, min(b1[1, 1], b2[1, 1]) - max(b1[0, 1], b2[0, 1]))
            overlap_area = x_overlap * y_overlap
            board_area = max(1e-8, (b1[1, 0] - b1[0, 0]) * (b1[1, 1] - b1[0, 1]))
            if overlap_area > 0:
                return min(1.0, overlap_area / board_area)
            gap_x = max(0.0, max(b1[0, 0], b2[0, 0]) - min(b1[1, 0], b2[1, 0]))
            gap_y = max(0.0, max(b1[0, 1], b2[0, 1]) - min(b1[1, 1], b2[1, 1]))
            min_gap = max(gap_x, gap_y)
            return max(0.0, 1.0 - min_gap / 0.05)

        def bounds_intersect_3d(b1, b2):
            if b1 is None or b2 is None:
                return False
            tol = 0.01
            return (
                b1[1, 0] >= b2[0, 0] - tol
                and b2[1, 0] >= b1[0, 0] - tol
                and b1[1, 1] >= b2[0, 1] - tol
                and b2[1, 1] >= b1[0, 1] - tol
                and b1[1, 2] >= b2[0, 2] - tol
                and b2[1, 2] >= b1[0, 2] - tol
            )

        def box_inside_basket():
            box_bounds = get_bounds(BOX)
            basket_bounds = get_bounds(BASKET)
            if box_bounds is None or basket_bounds is None:
                return False
            tol = 0.01
            return (
                box_bounds[0, 0] >= basket_bounds[0, 0] + tol
                and box_bounds[1, 0] <= basket_bounds[1, 0] - tol
                and box_bounds[0, 1] >= basket_bounds[0, 1] + tol
                and box_bounds[1, 1] <= basket_bounds[1, 1] - tol
                and box_bounds[1, 2] <= basket_bounds[1, 2] - tol
            )

        board_bounds = get_bounds(BOARD)
        basket_bounds = get_bounds(BASKET)
        if board_bounds is None or basket_bounds is None:
            score_t = torch.tensor(0.0, dtype=torch.float32, device=self._device)
            return score_t, {"score": score_t}

        right_ee_pos = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee_pos = self._robot.left_ee_pose[0, :3].cpu().numpy()

        board_score = 0.0
        if BOARD in objs_info:
            board_pos = get_pos(BOARD)
            board_euler = get_euler(BOARD)
            if board_pos is not None:
                dist_right = float(np.linalg.norm(right_ee_pos - board_pos))
                dist_left = float(np.linalg.norm(left_ee_pos - board_pos))
                min_dist_board = min(dist_right, dist_left)
            else:
                min_dist_board = float("inf")

            board_height_gain = float(board_pos[2] - INITIAL_BOARD_POS[2]) if board_pos is not None else 0.0
            board_tilted = False
            if board_euler is not None:
                board_tilted = (
                    abs(float(board_euler[0])) > TILT_THRESHOLD_DEG or abs(float(board_euler[1])) > TILT_THRESHOLD_DEG
                )
            board_lifted = board_height_gain > LIFT_HEIGHT
            board_tilt_or_lift = board_tilted or board_lifted
            tilt_progress = min(
                1.0,
                max(
                    abs(float(board_euler[0])) / TILT_THRESHOLD_DEG if board_euler is not None else 0.0,
                    abs(float(board_euler[1])) / TILT_THRESHOLD_DEG if board_euler is not None else 0.0,
                    board_height_gain / LIFT_HEIGHT,
                ),
            )
            board_basket_xy_intersect = bounds_intersect_xy(board_bounds, basket_bounds)

            if min_dist_board <= REACH_THRESHOLD:
                reach_progress = 1.0 - (min_dist_board - TOUCH_THRESHOLD) / (REACH_THRESHOLD - TOUCH_THRESHOLD)
                reach_progress = max(0.0, min(1.0, reach_progress))
                ee_based = (
                    0.2 + reach_progress * 0.1 if min_dist_board <= TOUCH_THRESHOLD else 0.1 + reach_progress * 0.1
                )
            elif min_dist_board < APPROACH_THRESHOLD:
                approach_progress = 1.0 - (min_dist_board - REACH_THRESHOLD) / (APPROACH_THRESHOLD - REACH_THRESHOLD)
                approach_progress = max(0.0, min(1.0, approach_progress))
                ee_based = approach_progress * 0.1
            else:
                ee_based = 0.0

            overlap_progress = xy_overlap_progress(board_bounds, basket_bounds)
            if board_basket_xy_intersect:
                board_state_score = 0.4
            elif board_tilt_or_lift:
                board_state_score = 0.3 + overlap_progress * 0.1
            elif min_dist_board <= TOUCH_THRESHOLD:
                board_state_score = 0.2 + min(1.0, tilt_progress) * 0.1
            else:
                board_state_score = ee_based

            board_score = max(board_state_score, ee_based)

        box_score = 0.0
        if BOX in objs_info:
            box_pos = get_pos(BOX)
            box_bounds = get_bounds(BOX)
            board_pos = get_pos(BOARD)
            if box_pos is not None:
                dist_right = float(np.linalg.norm(right_ee_pos - box_pos))
                dist_left = float(np.linalg.norm(left_ee_pos - box_pos))
                min_dist_box = min(dist_right, dist_left)
            else:
                min_dist_box = float("inf")

            box_board_dist_2d = (
                float(np.linalg.norm(box_pos[:2] - board_pos[:2]))
                if box_pos is not None and board_pos is not None
                else float("inf")
            )
            closer_progress = max(0.0, 1.0 - box_board_dist_2d / max(INITIAL_BOX_BOARD_DIST, 1e-6))
            box_board_collide = bounds_intersect_3d(box_bounds, board_bounds)
            box_z = float(box_pos[2]) if box_pos is not None else 0.0
            box_pushed_up = box_board_collide and box_z > INITIAL_BOX_POS[2]

            if min_dist_box <= REACH_THRESHOLD:
                reach_progress = 1.0 - (min_dist_box - TOUCH_THRESHOLD) / (REACH_THRESHOLD - TOUCH_THRESHOLD)
                reach_progress = max(0.0, min(1.0, reach_progress))
                ee_based = 0.2 + reach_progress * 0.1 if min_dist_box <= TOUCH_THRESHOLD else 0.1 + reach_progress * 0.1
            elif min_dist_box < APPROACH_THRESHOLD:
                approach_progress = 1.0 - (min_dist_box - REACH_THRESHOLD) / (APPROACH_THRESHOLD - REACH_THRESHOLD)
                approach_progress = max(0.0, min(1.0, approach_progress))
                ee_based = approach_progress * 0.1
            else:
                ee_based = 0.0

            if box_inside_basket():
                box_state_score = 0.6
            elif box_pushed_up:
                progress_to_basket = 0.0
                if box_bounds is not None and basket_bounds is not None:
                    box_center_in_basket_xy = (
                        box_bounds[0, 0] >= basket_bounds[0, 0]
                        and box_bounds[1, 0] <= basket_bounds[1, 0]
                        and box_bounds[0, 1] >= basket_bounds[0, 1]
                        and box_bounds[1, 1] <= basket_bounds[1, 1]
                    )
                    progress_to_basket = 0.5 if box_center_in_basket_xy else 0.0
                    box_top_z = box_bounds[1, 2]
                    basket_top_z = basket_bounds[1, 2]
                    z_progress = min(
                        1.0,
                        max(0.0, (box_top_z - INITIAL_BOX_POS[2]) / (basket_top_z - INITIAL_BOX_POS[2] + 1e-6)),
                    )
                    progress_to_basket += 0.5 * z_progress
                box_state_score = 0.5 + min(1.0, progress_to_basket) * 0.1
            elif box_board_collide:
                push_progress = min(1.0, max(0.0, (box_z - INITIAL_BOX_POS[2]) / 0.05))
                box_state_score = 0.4 + push_progress * 0.1
            elif min_dist_box <= TOUCH_THRESHOLD:
                box_state_score = 0.2 + min(1.0, closer_progress) * 0.1
            else:
                box_state_score = ee_based

            box_score = max(box_state_score, ee_based)

        total_score = board_score + box_score
        score_t = torch.tensor(total_score, dtype=torch.float32, device=self._device)
        return score_t, {"score": score_t}
