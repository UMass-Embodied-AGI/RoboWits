"""BallIntoBottle environment for RoboWits.

Task: Transfer a ball from a container into a bottle using a funnel.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv


@register_task("robowits/12-ball-into-bottle-v0")
class BallIntoBottleEnv(RoboWitsEnv):
    """Transfer a ball from a container into a bottle using a funnel.

    The funnel provides a wide capture entrance. Gravity guides the ball
    through the spout into the bottle. The ball container is graspable
    and can be poured into the funnel opening.

    Success criteria:
    - Ball is fully inside the bottle's bounding box
    - Ball is not touching the table
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Transfer the ball from the blue container into the red bottle."

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
        small_area = {"x_min": 0.25, "x_max": 0.65, "y_min": -0.4, "y_max": 0.4}
        self._object_reachable_areas["small ball"] = small_area
        self._object_reachable_areas["red bottle"] = small_area
        self._object_reachable_areas["funnel"] = small_area
        self._object_reachable_areas["ball container"] = small_area

    @property
    def placement_groups(self) -> PlacementGroups:
        """Bottle and funnel independent, ball container grouped with ball."""
        return ("red bottle", "funnel", ("ball container", "small ball"))

    def placement_entity_extra_hulls(self, name: str) -> list:
        """Force arm exclusion hulls for the bottle even though it is fixed."""
        if name == "red bottle":
            import numpy as np

            pad = 0.05

            def _hull(bbox):
                x0, x1 = bbox["x"][0] - pad, bbox["x"][1] + pad
                y0, y1 = bbox["y"][0] - pad, bbox["y"][1] + pad
                return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)

            return [_hull(self.ARM_FOREARM_BBOX_RIGHT), _hull(self.ARM_FOREARM_BBOX_LEFT)]
        return super().placement_entity_extra_hulls(name)

    def _add_custom_entities(self) -> None:
        """Add bottle, funnel, ball container, and ball."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Red bottle (mesh) - fixed
        bottle = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=gs.options.CoacdOptions(
                    threshold=0.01, preprocess_resolution=150, max_convex_hull=50, decimate=True
                ),
                file=get_asset_path("blender_kit/ffb3fbe7-1355-465f-8750-475210d8c949/obj.glb", pattern_is_dir=False),
                scale=(1.4, 1.4, 1.0),
                pos=(0.50, 0.00, 0.88026),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Glass(double_sided=True, color=(0.8, 0.2, 0.2)),
        )
        self._entities["red bottle"] = {
            "entity": bottle,
        }

        # Funnel (mesh)
        funnel = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/funnel.glb", pattern_is_dir=False),
                scale=(0.7, 0.7, 1.1),
                pos=(0.64, -0.08, 0.825),
                euler=(0.0, -90.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["funnel"] = {
            "entity": funnel,
        }

        # Ball container (mesh)
        container = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/3d998505-6bbb-4cc2-8359-c147ac531430/obj.glb", pattern_is_dir=False),
                scale=1.1,
                pos=(0.50, 0.14, 0.76 + 0.0459 * 1.1),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Glass(color=(0.8, 0.9, 1.0), opacity=0.3, double_sided=True),
        )
        self._entities["ball container"] = {
            "entity": container,
        }

        # Small ball (sphere)
        ball = self._scene.scene.add_entity(
            gs.morphs.Sphere(
                pos=(0.50, 0.14, 0.85),
                radius=0.005,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(color=(0.2, 0.6, 0.9), double_sided=True),
        )
        self._entities["small ball"] = {
            "entity": ball,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: ball inside bottle."""
        objs_info = self.collect_objs_info()

        if not isinstance(objs_info, dict):
            return torch.tensor([False], dtype=torch.bool, device=self._device)
        ball = objs_info.get("small ball")
        bottle = objs_info.get("red bottle")
        if ball is None or bottle is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        table_z = 0.76
        safe_margin = 0.002
        top_margin = 0.11

        def f(v):
            try:
                return float(v)
            except Exception:
                return None

        def point_in_polygon(px, py, poly):
            n = len(poly)
            if n < 3:
                return False
            inside = False
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                if (y1 > py) != (y2 > py):
                    x_int = (x2 - x1) * (py - y1) / (y2 - y1 + 1e-12) + x1
                    if px < x_int:
                        inside = not inside
            return inside

        def not_fallen(obj):
            b = obj.get("bounds")
            if b is not None:
                zmin = f(b[0][2])
            else:
                pos = obj.get("pos")
                zmin = f(pos[2]) if pos is not None else None
            if zmin is None:
                return False
            return zmin >= (table_z - 0.05)

        ball_pos = ball.get("pos")
        if ball_pos is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)
        bx, by, bz = f(ball_pos[0]), f(ball_pos[1]), f(ball_pos[2])
        if bx is None or by is None or bz is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        ball_bounds = ball.get("bounds")
        ball_radius = None
        if ball_bounds is not None:
            try:
                dx = f(ball_bounds[1][0]) - f(ball_bounds[0][0])
                dy = f(ball_bounds[1][1]) - f(ball_bounds[0][1])
                dz = f(ball_bounds[1][2]) - f(ball_bounds[0][2])
                ball_radius = 0.5 * min(abs(dx), abs(dy), abs(dz))
            except Exception:
                ball_radius = None
        if ball_radius is None or ball_radius <= 0:
            ball_radius = 0.01

        ball_bottom_z = bz - ball_radius
        if ball_bottom_z <= (table_z + safe_margin):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        bottle_poly = bottle.get("convex_hull_2d")
        if bottle_poly is not None:
            poly = [(f(p[0]), f(p[1])) for p in bottle_poly]
            poly = [(x, y) for (x, y) in poly if x is not None and y is not None]
        else:
            bnd = bottle.get("bounds")
            if bnd is None:
                return torch.tensor([False], dtype=torch.bool, device=self._device)
            x0, y0 = f(bnd[0][0]), f(bnd[0][1])
            x1, y1 = f(bnd[1][0]), f(bnd[1][1])
            if None in (x0, y0, x1, y1):
                return torch.tensor([False], dtype=torch.bool, device=self._device)
            poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

        bottle_bounds = bottle.get("bounds")
        if bottle_bounds is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)
        bottle_zmin = f(bottle_bounds[0][2])
        bottle_zmax = f(bottle_bounds[1][2])
        if bottle_zmin is None or bottle_zmax is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        inside_xy = point_in_polygon(bx, by, poly)

        ball_top_z = bz + ball_radius
        fully_below_mouth = ball_top_z < (bottle_zmax - top_margin)
        above_bottle_base = ball_bottom_z > (bottle_zmin + safe_margin)

        if not (not_fallen(ball) and not_fallen(bottle)):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        return torch.tensor(
            [inside_xy and fully_below_mouth and above_bottle_base], dtype=torch.bool, device=self._device
        )

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute dense reward based on milestone-based progress score."""
        objs_info = self.collect_objs_info()

        REACH_THRESHOLD = 0.15
        TOUCH_THRESHOLD = 0.05
        LIFT_HEIGHT = 0.1
        APPROACH_THRESHOLD = 0.3

        FUNNEL = "funnel"
        BALL_CONTAINER = "ball container"
        RED_BOTTLE = "red bottle"
        SMALL_BALL = "small ball"

        INITIAL_FUNNEL_POS = np.array([0.64, -0.08, 0.825], dtype=float)
        INITIAL_BALL_CONTAINER_POS = np.array([0.50, 0.14, 0.8601], dtype=float)

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

        def point_in_polygon(px, py, poly):
            if poly is None or len(poly) < 3:
                return False
            inside = False
            n = len(poly)
            for i in range(n):
                x1, y1 = float(poly[i, 0]), float(poly[i, 1])
                x2, y2 = float(poly[(i + 1) % n, 0]), float(poly[(i + 1) % n, 1])
                if (y1 > py) != (y2 > py):
                    denom = y2 - y1
                    if abs(denom) > 1e-12:
                        x_int = (x2 - x1) * (py - y1) / denom + x1
                        if px < x_int:
                            inside = not inside
            return inside

        def bounds_intersect(b1, b2):
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

        def get_footprint(name):
            obj = objs_info.get(name)
            if obj is None:
                return None
            hull = obj.get("convex_hull_2d")
            if hull is not None and len(hull) >= 3:
                return np.array([[float(p[0]), float(p[1])] for p in hull])
            b = obj.get("bounds")
            if b is None:
                return None
            b = np.array(b, dtype=float)
            return np.array(
                [[b[0, 0], b[0, 1]], [b[1, 0], b[0, 1]], [b[1, 0], b[1, 1]], [b[0, 0], b[1, 1]]], dtype=float
            )

        def ball_inside_bottle():
            ball = objs_info.get(SMALL_BALL)
            bottle = objs_info.get(RED_BOTTLE)
            if ball is None or bottle is None:
                return False
            ball_pos = get_pos(SMALL_BALL)
            if ball_pos is None:
                return False
            bottle_bounds = get_bounds(RED_BOTTLE)
            if bottle_bounds is None:
                return False
            bottle_poly = get_footprint(RED_BOTTLE)
            if bottle_poly is None:
                return False
            ball_radius = 0.005
            ball_bounds = get_bounds(SMALL_BALL)
            if ball_bounds is not None:
                sz = min(
                    abs(ball_bounds[1, 0] - ball_bounds[0, 0]),
                    abs(ball_bounds[1, 1] - ball_bounds[0, 1]),
                    abs(ball_bounds[1, 2] - ball_bounds[0, 2]),
                )
                ball_radius = 0.5 * sz
            bx, by, bz = float(ball_pos[0]), float(ball_pos[1]), float(ball_pos[2])
            table_z = 0.76
            bottle_zmin = float(bottle_bounds[0, 2])
            bottle_zmax = float(bottle_bounds[1, 2])
            top_margin = 0.11
            if not point_in_polygon(bx, by, bottle_poly):
                return False
            ball_bottom_z = bz - ball_radius
            ball_top_z = bz + ball_radius
            if ball_bottom_z <= table_z + 0.002:
                return False
            if ball_top_z >= bottle_zmax - top_margin:
                return False
            return not ball_bottom_z <= bottle_zmin + 0.002

        bottle_bounds = get_bounds(RED_BOTTLE)
        if bottle_bounds is None:
            score_t = torch.tensor(0.0, dtype=torch.float32, device=self._device)
            return score_t, {"score": score_t}

        right_ee_pos = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee_pos = self._robot.left_ee_pose[0, :3].cpu().numpy()
        bottle_zmax = float(bottle_bounds[1, 2])
        bottle_footprint = get_footprint(RED_BOTTLE)
        funnel_footprint = get_footprint(FUNNEL)

        funnel_score = 0.0
        if FUNNEL in objs_info:
            funnel_pos = get_pos(FUNNEL)
            funnel_bounds = get_bounds(FUNNEL)
            if funnel_pos is not None:
                dist_right = float(np.linalg.norm(right_ee_pos - funnel_pos))
                dist_left = float(np.linalg.norm(left_ee_pos - funnel_pos))
                min_dist_funnel = min(dist_right, dist_left)
            else:
                min_dist_funnel = float("inf")

            funnel_height_gain = float(funnel_pos[2] - INITIAL_FUNNEL_POS[2]) if funnel_pos is not None else 0.0
            funnel_inside_bottle = bounds_intersect(funnel_bounds, bottle_bounds)
            funnel_on_mouth = False
            if funnel_pos is not None and funnel_bounds is not None and bottle_footprint is not None:
                funnel_z_min = float(funnel_bounds[0, 2])
                mouth_tol = 0.05
                funnel_on_mouth = (
                    point_in_polygon(funnel_pos[0], funnel_pos[1], bottle_footprint)
                    and abs(funnel_z_min - bottle_zmax) < mouth_tol
                )

            if min_dist_funnel <= REACH_THRESHOLD:
                reach_progress = 1.0 - (min_dist_funnel - TOUCH_THRESHOLD) / (REACH_THRESHOLD - TOUCH_THRESHOLD)
                reach_progress = max(0.0, min(1.0, reach_progress))
                ee_based = (
                    0.2 + reach_progress * 0.1 if min_dist_funnel <= TOUCH_THRESHOLD else 0.1 + reach_progress * 0.1
                )
            elif min_dist_funnel < APPROACH_THRESHOLD:
                approach_progress = 1.0 - (min_dist_funnel - REACH_THRESHOLD) / (APPROACH_THRESHOLD - REACH_THRESHOLD)
                approach_progress = max(0.0, min(1.0, approach_progress))
                ee_based = approach_progress * 0.1
            else:
                ee_based = 0.0

            lift_progress = min(1.0, max(0.0, funnel_height_gain) / LIFT_HEIGHT)
            if funnel_on_mouth:
                funnel_state_score = 0.5
            elif funnel_inside_bottle:
                progress_to_mouth = 0.0
                if funnel_pos is not None and funnel_bounds is not None:
                    funnel_z_min = float(funnel_bounds[0, 2])
                    dist_to_mouth = abs(funnel_z_min - bottle_zmax)
                    progress_to_mouth = max(0.0, 1.0 - dist_to_mouth / 0.05)
                funnel_state_score = 0.4 + min(1.0, progress_to_mouth) * 0.1
            elif funnel_height_gain > LIFT_HEIGHT:
                progress_to_inside = 0.0
                if funnel_bounds is not None and bottle_bounds is not None:
                    overlap_x = (
                        funnel_bounds[1, 0] >= bottle_bounds[0, 0] and bottle_bounds[1, 0] >= funnel_bounds[0, 0]
                    )
                    overlap_y = (
                        funnel_bounds[1, 1] >= bottle_bounds[0, 1] and bottle_bounds[1, 1] >= funnel_bounds[0, 1]
                    )
                    progress_to_inside = 1.0 if (overlap_x and overlap_y) else 0.5
                funnel_state_score = 0.3 + min(1.0, progress_to_inside) * 0.1
            elif min_dist_funnel <= TOUCH_THRESHOLD:
                funnel_state_score = 0.2 + lift_progress * 0.1
            else:
                funnel_state_score = ee_based

            funnel_score = max(funnel_state_score, ee_based)

        ball_container_score = 0.0
        if BALL_CONTAINER in objs_info:
            container_pos = get_pos(BALL_CONTAINER)
            funnel_pos = get_pos(FUNNEL)
            if container_pos is not None:
                dist_right = float(np.linalg.norm(right_ee_pos - container_pos))
                dist_left = float(np.linalg.norm(left_ee_pos - container_pos))
                min_dist_container = min(dist_right, dist_left)
            else:
                min_dist_container = float("inf")

            container_height_gain = (
                float(container_pos[2] - INITIAL_BALL_CONTAINER_POS[2]) if container_pos is not None else 0.0
            )
            container_above_funnel = False
            if container_pos is not None and funnel_pos is not None and funnel_footprint is not None:
                container_above_funnel = (
                    point_in_polygon(container_pos[0], container_pos[1], funnel_footprint)
                    and container_pos[2] > funnel_pos[2]
                )

            if min_dist_container <= REACH_THRESHOLD:
                reach_progress = 1.0 - (min_dist_container - TOUCH_THRESHOLD) / (REACH_THRESHOLD - TOUCH_THRESHOLD)
                reach_progress = max(0.0, min(1.0, reach_progress))
                ee_based = (
                    0.2 + reach_progress * 0.1 if min_dist_container <= TOUCH_THRESHOLD else 0.1 + reach_progress * 0.1
                )
            elif min_dist_container < APPROACH_THRESHOLD:
                approach_progress = 1.0 - (min_dist_container - REACH_THRESHOLD) / (
                    APPROACH_THRESHOLD - REACH_THRESHOLD
                )
                approach_progress = max(0.0, min(1.0, approach_progress))
                ee_based = approach_progress * 0.1
            else:
                ee_based = 0.0

            lift_progress = min(1.0, max(0.0, container_height_gain) / LIFT_HEIGHT)
            if ball_inside_bottle():
                container_state_score = 0.5
            elif container_above_funnel:
                container_state_score = 0.4
            elif container_height_gain > LIFT_HEIGHT:
                progress_to_above = 0.0
                if container_pos is not None and funnel_pos is not None and funnel_footprint is not None:
                    xy_inside = point_in_polygon(container_pos[0], container_pos[1], funnel_footprint)
                    z_above = container_pos[2] > funnel_pos[2]
                    progress_to_above = (0.5 if xy_inside else 0.0) + (0.5 if z_above else 0.0)
                container_state_score = 0.3 + min(1.0, progress_to_above) * 0.1
            elif min_dist_container <= TOUCH_THRESHOLD:
                container_state_score = 0.2 + lift_progress * 0.1
            else:
                container_state_score = ee_based

            ball_container_score = max(container_state_score, ee_based)

            total_score = 1.0 if ball_inside_bottle() else funnel_score + ball_container_score
        score_t = torch.tensor(total_score, dtype=torch.float32, device=self._device)
        return score_t, {"score": score_t}
