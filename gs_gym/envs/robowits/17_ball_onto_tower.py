"""BallOntoTower environment for RoboWits.

Task: Place the ball stably on top of the tower.
"""

from __future__ import annotations

import math

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv
from gs_gym.envs.robowits.utils import check_being_held


@register_task("robowits/17-ball-onto-tower-v0")
class BallOntoTowerEnv(RoboWitsEnv):
    """Place the ball stably on top of the tower.

    The tower pole alone cannot support the ball as it would roll off.
    The base can only fit 2.5 rings inside its height, so stacking all
    three rings causes the top ring to protrude above the pole, creating
    a hollow cup-like surface where the ball can rest stably in the center.

    Success criteria:
    - Ball is stable (low velocity)
    - Ball is on top of the tower base (XY projection over base footprint)
    - Ball center is above the top of the tower base
    - Ball is not being held by either gripper
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Place the ball stably on top of the tower."

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
        tower_area = {"x_min": 0.40, "x_max": 0.6, "y_min": -0.2, "y_max": 0.2}
        self._object_reachable_areas["tower base"] = tower_area
        self._object_reachable_areas["large ring"] = tower_area
        self._object_reachable_areas["medium ring"] = tower_area
        small_area = {"x_min": 0.35, "x_max": 0.6, "y_min": -0.4, "y_max": 0.4}
        self._object_reachable_areas["small ring"] = small_area
        self._object_reachable_areas["ball"] = small_area

    @property
    def placement_groups(self) -> PlacementGroups:
        """Small ring and ball independent, tower components grouped."""
        return ("small ring", "ball", ("tower base", "large ring", "medium ring"))

    def _add_custom_entities(self) -> None:
        """Add tower base, rings, and ball."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Tower base (mesh, fixed)
        tower_base = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/tower_base.glb", pattern_is_dir=False),
                scale=(1.0, 1.0, 1.5),
                pos=(0.605, 0.0, 0.803),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
                decimate=True,
                decimate_face_num=100,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["tower base"] = {
            "entity": tower_base,
        }

        # Large ring (mesh, fixed)
        large_ring = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/large_ring.glb", pattern_is_dir=False),
                scale=(1.2, 1.2, 1.5),
                pos=(0.605, 0.0, 0.802),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
                decimate=True,
                decimate_face_num=100,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["large ring"] = {
            "entity": large_ring,
        }

        # Medium ring (mesh, fixed)
        medium_ring = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/medium_ring.glb", pattern_is_dir=False),
                scale=(1.2, 1.2, 1.5),
                pos=(0.605, 0.0, 0.835),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
                decimate=True,
                decimate_face_num=100,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["medium ring"] = {
            "entity": medium_ring,
        }

        # Small ring (mesh, movable)
        small_ring = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/small_ring.glb", pattern_is_dir=False),
                scale=(1.2, 1.2, 1.5),
                pos=(0.69, 0.08, 0.7704),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
                decimate=True,
                decimate_face_num=100,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["small ring"] = {
            "entity": small_ring,
        }

        # Ball (mesh)
        ball = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb", pattern_is_dir=False),
                scale=0.2,
                pos=(0.34, 0.0, 0.773),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["ball"] = {
            "entity": ball,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: ball on top of tower, stable, not held."""
        objs_info = self.collect_objs_info()

        TABLE_X_MIN, TABLE_X_MAX = 0.21, 1.00
        TABLE_Y_MIN, TABLE_Y_MAX = -0.69, 0.69
        TABLE_TOL = 1e-3

        base = objs_info.get("tower base")
        ball = objs_info.get("ball")

        if base is None or ball is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        def in_table_xy(obj):
            hull = obj.get("convex_hull_2d")
            if hull is not None:
                for pt in hull:
                    x, y = float(pt[0]), float(pt[1])
                    if not (TABLE_X_MIN - TABLE_TOL <= x <= TABLE_X_MAX + TABLE_TOL):
                        return False
                    if not (TABLE_Y_MIN - TABLE_TOL <= y <= TABLE_Y_MAX + TABLE_TOL):
                        return False
                return True
            pos = obj.get("pos")
            if pos is None:
                return False
            x, y = float(pos[0]), float(pos[1])
            return (TABLE_X_MIN - TABLE_TOL <= x <= TABLE_X_MAX + TABLE_TOL) and (
                TABLE_Y_MIN - TABLE_TOL <= y <= TABLE_Y_MAX + TABLE_TOL
            )

        if not in_table_xy(base) or not in_table_xy(ball):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        vel = ball.get("vel")
        if vel is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        speed = math.sqrt(float(vel[0]) ** 2 + float(vel[1]) ** 2 + float(vel[2]) ** 2)
        stable_linear = speed < 0.05
        stable_vertical = abs(float(vel[2])) < 0.03
        stability_ok = stable_linear and stable_vertical

        base_bounds = base.get("bounds")
        ball_pos = ball.get("pos")
        if base_bounds is None or ball_pos is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        base_top_z = float(base_bounds[1][2])
        ball_center_z = float(ball_pos[2])
        height_ok = ball_center_z > base_top_z + 0.005

        base_hull = base.get("convex_hull_2d")
        ball_xy = (float(ball_pos[0]), float(ball_pos[1]))

        def point_in_polygon(pt, poly):
            x, y = pt
            inside = False
            n = len(poly)
            if n < 3:
                return False
            for i in range(n):
                x1, y1 = float(poly[i][0]), float(poly[i][1])
                x2, y2 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
                if (y1 > y) != (y2 > y):
                    xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
                    if x < xinters + 1e-12:
                        inside = not inside
            return inside

        if base_hull is not None and len(base_hull) >= 3:
            on_top_ok = point_in_polygon(ball_xy, base_hull)
        else:
            x_min, y_min = float(base_bounds[0][0]) - 0.005, float(base_bounds[0][1]) - 0.005
            x_max, y_max = float(base_bounds[1][0]) + 0.005, float(base_bounds[1][1]) + 0.005
            on_top_ok = (x_min <= ball_xy[0] <= x_max) and (y_min <= ball_xy[1] <= y_max)

        # Check ball is not being held by either gripper
        not_held = not check_being_held(ball, self._robot)

        return torch.tensor(
            [stability_ok and height_ok and on_top_ok and not_held], dtype=torch.bool, device=self._device
        )

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute dense reward based on milestone-based progress score."""
        objs_info = self.collect_objs_info()

        SMALL_RING = "small ring"
        BALL = "ball"
        TOWER_BASE = "tower base"

        REACH_THRESHOLD = 0.15
        TOUCH_THRESHOLD = 0.05
        APPROACH_THRESHOLD = 0.4

        INITIAL_RING_POS = np.array([0.69, 0.08, 0.7704])
        INITIAL_BALL_POS = np.array([0.34, 0.0, 0.773])
        TOWER_BASE_POS = np.array([0.605, 0.0, 0.803])

        for name in [SMALL_RING, BALL, TOWER_BASE]:
            if name not in objs_info:
                score_t = torch.tensor(0.0, dtype=torch.float32, device=self._device)
                return score_t, {"score": score_t}

        def get_pos(name):
            p = objs_info[name]["pos"]
            return np.array([p[0], p[1], p[2]], dtype=float)

        def get_bounds(name):
            b = objs_info[name].get("bounds")
            if b is None:
                return None
            return np.array(b, dtype=float)

        def is_ring_on_tower():
            ring_pos = get_pos(SMALL_RING)
            base_pos = get_pos(TOWER_BASE)
            base_bounds = get_bounds(TOWER_BASE)
            if base_bounds is None:
                return False
            xy_dist = float(np.linalg.norm(ring_pos[:2] - base_pos[:2]))
            base_top_z = float(base_bounds[1, 2])
            ring_z = ring_pos[2]
            return xy_dist < 0.08 and ring_z >= base_top_z - 0.02

        def is_ball_on_tower():
            ball_pos = get_pos(BALL)
            base_pos = get_pos(TOWER_BASE)
            base_bounds = get_bounds(TOWER_BASE)
            if base_bounds is None:
                return False
            xy_dist = float(np.linalg.norm(ball_pos[:2] - base_pos[:2]))
            base_top_z = float(base_bounds[1, 2])
            ball_z = ball_pos[2]
            v = objs_info[BALL].get("vel")
            is_stable = True
            if v is not None:
                speed = float(np.linalg.norm(v))
                is_stable = speed < 0.1
            return xy_dist < 0.05 and ball_z > base_top_z + 0.01 and is_stable

        right_ee_pos = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee_pos = self._robot.left_ee_pose[0, :3].cpu().numpy()

        ring_pos = get_pos(SMALL_RING)
        dist_right_to_ring = float(np.linalg.norm(right_ee_pos - ring_pos))
        dist_left_to_ring = float(np.linalg.norm(left_ee_pos - ring_pos))
        min_dist_to_ring = min(dist_right_to_ring, dist_left_to_ring)

        ring_score = 0.0
        if is_ring_on_tower():
            ring_score = 0.6
        elif min_dist_to_ring <= TOUCH_THRESHOLD:
            ring_displacement = float(np.linalg.norm(ring_pos - INITIAL_RING_POS))
            ring_toward_tower = max(0.0, 1.0 - float(np.linalg.norm(ring_pos[:2] - TOWER_BASE_POS[:2])) / 0.3)
            progress = min(1.0, (ring_displacement * 0.5 + ring_toward_tower * 0.5))
            ring_score = 0.4 + progress * 0.2
        elif min_dist_to_ring <= REACH_THRESHOLD:
            reach_progress = 1.0 - (min_dist_to_ring - TOUCH_THRESHOLD) / (REACH_THRESHOLD - TOUCH_THRESHOLD)
            reach_progress = max(0.0, min(1.0, reach_progress))
            ring_score = 0.2 + reach_progress * 0.2
        elif min_dist_to_ring < APPROACH_THRESHOLD:
            approach_progress = 1.0 - (min_dist_to_ring - REACH_THRESHOLD) / (APPROACH_THRESHOLD - REACH_THRESHOLD)
            approach_progress = max(0.0, min(1.0, approach_progress))
            ring_score = approach_progress * 0.2
        ring_score = min(0.6, ring_score)

        ball_pos = get_pos(BALL)
        dist_right_to_ball = float(np.linalg.norm(right_ee_pos - ball_pos))
        dist_left_to_ball = float(np.linalg.norm(left_ee_pos - ball_pos))
        min_dist_to_ball = min(dist_right_to_ball, dist_left_to_ball)

        ball_score = 0.0
        if is_ball_on_tower():
            ball_score = 0.4
        else:
            ball_height_gain = max(0.0, ball_pos[2] - INITIAL_BALL_POS[2])
            ball_toward_tower = max(0.0, 1.0 - float(np.linalg.norm(ball_pos[:2] - TOWER_BASE_POS[:2])) / 0.3)

            if ball_height_gain > 0.02 or ball_toward_tower > 0.5:
                height_progress = min(1.0, ball_height_gain / 0.08)
                position_progress = ball_toward_tower
                combined_progress = max(height_progress, position_progress)
                ball_score = 0.2 + combined_progress * 0.2
            elif min_dist_to_ball <= TOUCH_THRESHOLD:
                ball_score = 0.2
            elif min_dist_to_ball <= REACH_THRESHOLD:
                reach_progress = 1.0 - (min_dist_to_ball - TOUCH_THRESHOLD) / (REACH_THRESHOLD - TOUCH_THRESHOLD)
                reach_progress = max(0.0, min(1.0, reach_progress))
                ball_score = reach_progress * 0.2
            elif min_dist_to_ball < APPROACH_THRESHOLD:
                approach_progress = 1.0 - (min_dist_to_ball - REACH_THRESHOLD) / (APPROACH_THRESHOLD - REACH_THRESHOLD)
                approach_progress = max(0.0, min(1.0, approach_progress))
                ball_score = approach_progress * 0.1

        total_score = ring_score + ball_score
        score_t = torch.tensor(total_score, dtype=torch.float32, device=self._device)
        return score_t, {"score": score_t}
