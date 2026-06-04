"""HoldCup environment for RoboWits.

Task: Put a cup on a sloped target area without it sliding down.
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


@register_task("robowits/09-hold-cup-v0")
class HoldCupEnv(RoboWitsEnv):
    """Put a cup on a sloped target area without it sliding down.

    The cup and slope lack sufficient friction to remain stationary. The
    robot must utilize the surface friction of the coaster to stabilize
    the cup. The coaster acts as a high-friction interface.

    Success criteria:
    - Cup's 2D footprint overlaps target area by at least 50%
    - Cup is stable with near-zero linear velocity
    - Cup is upright (tilt from vertical <= 20 degrees)
    - Cup remains within table bounds
    - Cup is not being held by either gripper
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Put the cup on the green target area on the slope without sliding down."

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
        closer_area = {"x_min": 0.2, "x_max": 0.55, "y_min": -0.5, "y_max": 0.5}
        farther_area = {"x_min": 0.45, "x_max": 0.65, "y_min": -0.25, "y_max": 0.25}
        self._object_reachable_areas["target area"] = farther_area
        self._object_reachable_areas["coaster"] = closer_area
        self._object_reachable_areas["cup"] = closer_area

    @property
    def placement_groups(self) -> PlacementGroups:
        """Cup and coaster independent, target area grouped with slope."""
        return ("cup", "coaster", ("target area", "slope"))

    def _add_custom_entities(self) -> None:
        """Add cup, slope, coaster, and target area."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Cup (mesh)
        cup = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/3d998505-6bbb-4cc2-8359-c147ac531430/obj.glb", pattern_is_dir=False),
                scale=1.1,
                pos=(0.38, 0.00, 0.76 + 0.0459 * 1.1),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.2, coup_friction=0.05),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["cup"] = {
            "entity": cup,
        }

        # Slope (mesh) - fixed
        slope = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/ab77c172-c69f-41b5-956a-600ea7b64c73/obj.glb", pattern_is_dir=False),
                scale=(0.5, 1.5, 0.6),
                pos=(0.58, 0.0, 0.76 + 0.0586 * 0.5),
                euler=(0.0, 0.0, 90.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.2),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["slope"] = {
            "entity": slope,
        }

        # Coaster (cylinder)
        coaster = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.36, -0.12, 0.76 + 0.01),
                euler=(0.0, 0.0, 0.0),
                radius=0.04,
                height=0.02,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1.5),
            surface=gs.surfaces.Default(color=(0.6, 0.4, 0.2), roughness=0.7),
        )
        self._entities["coaster"] = {
            "entity": coaster,
        }

        # Target area (on slope, tilted)
        target_area = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.60, 0.0, 0.79988),
                euler=(-10.3, 0.0, 0.0),
                size=(0.13, 0.16, 0.002),
                fixed=True,
                collision=False,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.2, 0.8, 0.3), roughness=0.2),
        )
        self._entities["target area"] = {
            "entity": target_area,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: cup on target, stable, upright, within bounds."""

        objs_info = self.collect_objs_info()

        TBL_X_MIN, TBL_X_MAX = 0.21, 1.00
        TBL_Y_MIN, TBL_Y_MAX = -0.69, 0.69
        UPRIGHT_TOLERANCE_DEG = 20.0  # Max tilt from vertical

        cup = objs_info.get("cup")
        target = objs_info.get("target area")

        if cup is None or target is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        def is_upright(obj, tolerance_deg: float) -> bool:
            """Check if object's local up vector is close to world up using euler angles."""
            euler = obj.get("euler")
            if euler is None:
                return False
            # euler values are in radians; convert to degrees
            roll_deg = abs(math.degrees(float(euler[0])))
            pitch_deg = abs(math.degrees(float(euler[1])))
            # Normalize to [-180, 180] range
            if roll_deg > 180:
                roll_deg = 360 - roll_deg
            if pitch_deg > 180:
                pitch_deg = 360 - pitch_deg
            return roll_deg <= tolerance_deg and pitch_deg <= tolerance_deg

        def get_poly(obj):
            ch = obj.get("convex_hull_2d", None)
            try:
                if ch is not None and len(ch) >= 3:
                    return [(float(p[0]), float(p[1])) for p in ch]
            except Exception:
                pass
            b = obj.get("bounds", None)
            if b is None:
                return None
            try:
                (xmin, ymin, _), (xmax, ymax, _) = b
                return [
                    (float(xmin), float(ymin)),
                    (float(xmax), float(ymin)),
                    (float(xmax), float(ymax)),
                    (float(xmin), float(ymax)),
                ]
            except Exception:
                return None

        def poly_area(poly):
            if poly is None or len(poly) < 3:
                return 0.0
            area = 0.0
            for i in range(len(poly)):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % len(poly)]
                area += x1 * y2 - x2 * y1
            return abs(area) * 0.5

        def signed_area(poly):
            s = 0.0
            for i in range(len(poly)):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % len(poly)]
                s += x1 * y2 - x2 * y1
            return 0.5 * s

        def line_intersection(p1, p2, p3, p4):
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-12:
                return ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
            det1 = x1 * y2 - y1 * x2
            det2 = x3 * y4 - y3 * x4
            px = (det1 * (x3 - x4) - (x1 - x2) * det2) / denom
            py = (det1 * (y3 - y4) - (y1 - y2) * det2) / denom
            return (px, py)

        def clip_polygon(subject, clip):
            if subject is None or clip is None or len(subject) < 3 or len(clip) < 3:
                return []
            ori_sign = 1.0 if signed_area(clip) >= 0.0 else -1.0

            def inside(pt, a, b):
                cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
                return ori_sign * cross >= -1e-9

            output = [(float(x), float(y)) for x, y in subject]
            cl = [(float(x), float(y)) for x, y in clip]
            for i in range(len(cl)):
                a = cl[i]
                b = cl[(i + 1) % len(cl)]
                input_list = output
                output = []
                if not input_list:
                    break
                s = input_list[-1]
                for e in input_list:
                    if inside(e, a, b):
                        if not inside(s, a, b):
                            output.append(line_intersection(s, e, a, b))
                        output.append(e)
                    elif inside(s, a, b):
                        output.append(line_intersection(s, e, a, b))
                    s = e
            return output

        cup_poly = get_poly(cup)
        target_poly = get_poly(target)

        if cup_poly is None or target_poly is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        inter_poly = clip_polygon(cup_poly, target_poly)
        cup_area = poly_area(cup_poly)
        inter_area = poly_area(inter_poly)

        if cup_area <= 0.0:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        overlap_ok = (inter_area / cup_area) >= 0.5

        vel = cup.get("vel")
        if vel is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        vnorm = math.sqrt(float(vel[0]) ** 2 + float(vel[1]) ** 2 + float(vel[2]) ** 2)
        stable_ok = vnorm <= 0.005

        upright_ok = is_upright(cup, UPRIGHT_TOLERANCE_DEG)

        cup_ok = True
        for pt in cup_poly:
            x, y = float(pt[0]), float(pt[1])
            if not (TBL_X_MIN - 1e-6 <= x <= TBL_X_MAX + 1e-6):
                cup_ok = False
                break
            if not (TBL_Y_MIN - 1e-6 <= y <= TBL_Y_MAX + 1e-6):
                cup_ok = False
                break

        not_held = not check_being_held(cup, self._robot)

        return torch.tensor(
            [overlap_ok and stable_ok and upright_ok and cup_ok and not_held], dtype=torch.bool, device=self._device
        )

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute dense reward based on milestone-based progress score."""
        objs_info = self.collect_objs_info()

        CUP = "cup"
        COASTER = "coaster"
        SLOPE = "slope"

        REACH_THRESHOLD = 0.15
        TOUCH_THRESHOLD = 0.05
        APPROACH_THRESHOLD = 0.4

        if CUP not in objs_info or SLOPE not in objs_info:
            score_t = torch.tensor(0.0, dtype=torch.float32, device=self._device)
            return score_t, {"score": score_t}

        def get_pos(name):
            obj = objs_info.get(name)
            if obj is None:
                return None
            p = obj.get("pos")
            if p is None:
                return None
            return np.array([p[0], p[1], p[2]], dtype=float)

        def get_hull_2d(name):
            obj = objs_info.get(name)
            if obj is None:
                return None
            hull = obj.get("convex_hull_2d")
            if hull is not None:
                try:
                    hull = np.array(hull, dtype=float)
                    if hull.ndim == 2 and hull.shape[1] == 2 and len(hull) >= 3:
                        return hull
                except Exception:
                    pass
            b = obj.get("bounds")
            if b is None:
                return None
            b = np.array(b, dtype=float)
            if b.shape != (2, 3):
                return None
            x_min, y_min = b[0, 0], b[0, 1]
            x_max, y_max = b[1, 0], b[1, 1]
            return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=float)

        def point_in_polygon(point, poly):
            if poly is None or len(poly) < 3:
                return False
            x, y = float(point[0]), float(point[1])
            inside = False
            n = len(poly)
            for i in range(n):
                x1, y1 = float(poly[i][0]), float(poly[i][1])
                x2, y2 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
                if (y1 > y) != (y2 > y):
                    denom = y2 - y1
                    if abs(denom) > 1e-12:
                        xinters = (x2 - x1) * (y - y1) / denom + x1
                        if x < xinters:
                            inside = not inside
            return inside

        slope_pos = get_pos(SLOPE)

        def compute_object_score(obj_name, min_dist):
            obj_pos = get_pos(obj_name)
            slope_hull = get_hull_2d(SLOPE)

            if obj_pos is None:
                return 0.0

            if point_in_polygon(obj_pos[:2], slope_hull):
                return 0.5

            dist_to_slope_xy = (
                float(np.linalg.norm(obj_pos[:2] - slope_pos[:2])) if slope_pos is not None else float("inf")
            )
            position_progress = max(0.0, 1.0 - dist_to_slope_xy / 0.25)

            if dist_to_slope_xy < 0.25:
                return 0.3 + position_progress * 0.15
            elif min_dist <= TOUCH_THRESHOLD:
                return 0.3
            elif min_dist <= REACH_THRESHOLD:
                reach_progress = 1.0 - (min_dist - TOUCH_THRESHOLD) / (REACH_THRESHOLD - TOUCH_THRESHOLD)
                return 0.15 + max(0.0, min(1.0, reach_progress)) * 0.15
            elif min_dist < APPROACH_THRESHOLD:
                approach_progress = 1.0 - (min_dist - REACH_THRESHOLD) / (APPROACH_THRESHOLD - REACH_THRESHOLD)
                return max(0.0, min(1.0, approach_progress)) * 0.15
            else:
                return 0.0

        right_ee_pos = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee_pos = self._robot.left_ee_pose[0, :3].cpu().numpy()

        cup_pos = get_pos(CUP)
        if cup_pos is not None:
            dist_right_to_cup = float(np.linalg.norm(right_ee_pos - cup_pos))
            dist_left_to_cup = float(np.linalg.norm(left_ee_pos - cup_pos))
            min_dist_to_cup = min(dist_right_to_cup, dist_left_to_cup)
        else:
            min_dist_to_cup = float("inf")

        cup_score = compute_object_score(CUP, min_dist_to_cup)

        # If cup is on slope, also require uprightness for full score
        if cup_score >= 0.5:
            cup_obj = objs_info.get(CUP)
            if cup_obj is not None:
                euler = cup_obj.get("euler")
                if euler is not None:
                    roll_deg = abs(math.degrees(float(euler[0])))
                    pitch_deg = abs(math.degrees(float(euler[1])))
                    if roll_deg > 180:
                        roll_deg = 360 - roll_deg
                    if pitch_deg > 180:
                        pitch_deg = 360 - pitch_deg
                    if roll_deg > 20.0 or pitch_deg > 20.0:
                        cup_score = 0.4

        coaster_score = 0.0
        if COASTER in objs_info:
            coaster_pos = get_pos(COASTER)
            if coaster_pos is not None:
                dist_right_to_coaster = float(np.linalg.norm(right_ee_pos - coaster_pos))
                dist_left_to_coaster = float(np.linalg.norm(left_ee_pos - coaster_pos))
                min_dist_to_coaster = min(dist_right_to_coaster, dist_left_to_coaster)
            else:
                min_dist_to_coaster = float("inf")
            coaster_score = compute_object_score(COASTER, min_dist_to_coaster)

        total_score = cup_score + coaster_score
        score_t = torch.tensor(total_score, dtype=torch.float32, device=self._device)
        return score_t, {"score": score_t}
