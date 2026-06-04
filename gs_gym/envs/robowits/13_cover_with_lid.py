"""CoverWithLid environment for RoboWits.

Task: Cover a pot precisely using the correct sized lid.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits import utils as _utils
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv


@register_task("robowits/13-cover-with-lid-v0")
class CoverWithLidEnv(RoboWitsEnv):
    """Cover a pot precisely using the correct sized lid.

    This requires reasoning about cross-sectional fit. Only the correctly
    sized circular lid will cover the opening with minimal excess. The
    circular geometry and ring reference enable precise centering.

    Success criteria:
    - Exactly one lid is above the pot
    - Lid overlaps pot by at least 80%
    - Lid size is appropriate (85%-140% of pot area)
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Cover the pot precisely using one lid."

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
        return ("pot without lid", "small lid", "lid medium", "lid large")

    @property
    def _object_reachable_areas(self) -> dict:
        # 'lid large' starts at y=0.44 and has a large footprint that exceeds the default
        # y[-0.25, 0.25] bounds. Widen y for all lids so placement sampling succeeds.
        return {
            "small lid": {"y_min": -0.45, "y_max": 0.45},
            "lid medium": {"y_min": -0.45, "y_max": 0.45},
            "lid large": {"y_min": -0.45, "y_max": 0.45},
        }

    def _add_custom_entities(self) -> None:
        """Add pot and three lids of different sizes."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Pot without lid (mesh) - fixed
        pot = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/3aee9353-e21a-40d4-b160-e2a0af1fff7f/obj.glb", pattern_is_dir=False),
                scale=0.208,
                pos=(0.52, 0.0, 0.8292),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
                decimate=True,
                decimate_face_num=100,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["pot without lid"] = {
            "entity": pot,
        }

        # Small lid (mesh)
        small_lid = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c1a05823-5cb6-4e64-b3f3-5fd5a86cfc0b/obj.glb", pattern_is_dir=False),
                scale=(0.142, 0.142, 0.5),
                pos=(0.36, -0.12, 0.7668),
                euler=(0.0, 0.0, 90.0),
                fixed=False,
                collision=True,
                decimate=True,
                decimate_face_num=100,
            ),
            material=gs.materials.Rigid(rho=500, friction=1.5),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["small lid"] = {
            "entity": small_lid,
        }

        # Medium lid (mesh) - correct size
        medium_lid = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c1a05823-5cb6-4e64-b3f3-5fd5a86cfc0b/obj.glb", pattern_is_dir=False),
                scale=(0.221, 0.221, 0.5),
                pos=(0.70, 0.14, 0.7707),
                euler=(0.0, 0.0, 90.0),
                fixed=False,
                collision=True,
                decimate=True,
                decimate_face_num=100,
            ),
            material=gs.materials.Rigid(rho=500, friction=1.5),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["lid medium"] = {
            "entity": medium_lid,
        }

        # Large lid (mesh)
        large_lid = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c1a05823-5cb6-4e64-b3f3-5fd5a86cfc0b/obj.glb", pattern_is_dir=False),
                scale=(0.315, 0.315, 0.5),
                pos=(0.36, 0.44, 0.7752),
                euler=(0.0, 0.0, 90.0),
                fixed=False,
                collision=True,
                decimate=True,
                decimate_face_num=100,
            ),
            material=gs.materials.Rigid(rho=500, friction=1.5),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["lid large"] = {
            "entity": large_lid,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: exactly one lid covers pot correctly."""
        objs_info = self.collect_objs_info()

        TABLE_X_MIN, TABLE_X_MAX = 0.21, 1.00
        TABLE_Y_MIN, TABLE_Y_MAX = -0.69, 0.69
        OVERLAP_RATIO_MIN = 0.80
        SIZE_RATIO_MIN = 0.85
        SIZE_RATIO_MAX = 1.40
        Z_ABOVE_EPS = 0.01

        def get_hull(obj):
            hull = obj.get("convex_hull_2d", None)
            if hull is not None and isinstance(hull, np.ndarray) and hull.shape[0] >= 3:
                return hull.copy()
            b = obj.get("bounds", None)
            if b is None:
                return None
            (xmin, ymin, _), (xmax, ymax, _) = b
            return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float)

        def is_within_table_bounds(hull):
            if hull is None or len(hull) == 0:
                return False
            xs = hull[:, 0]
            ys = hull[:, 1]
            return (
                np.all(xs >= TABLE_X_MIN - 1e-3)
                and np.all(xs <= TABLE_X_MAX + 1e-3)
                and np.all(ys >= TABLE_Y_MIN - 1e-3)
                and np.all(ys <= TABLE_Y_MAX + 1e-3)
            )

        def compute_overlap_ratio(hull1, hull2):
            if hull1 is None or hull2 is None:
                return 0.0
            area1 = self._polygon_area(hull1)
            if area1 <= 1e-9:
                return 0.0
            inter_area = _utils.intersection_area(hull1, hull2)
            return inter_area / area1

        pot = objs_info.get("pot without lid")
        if pot is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        pot_hull = get_hull(pot)
        if pot_hull is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        if not is_within_table_bounds(pot_hull):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        pot_area = self._polygon_area(pot_hull)
        if pot_area <= 1e-6:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        pot_z = float(pot["pos"][2])

        lid_names = ["small lid", "lid medium", "lid large"]
        valid_candidates = []

        for lname in lid_names:
            lid = objs_info.get(lname)
            if lid is None:
                continue

            lid_hull = get_hull(lid)
            if lid_hull is None:
                continue

            if not is_within_table_bounds(lid_hull):
                continue

            lid_area = self._polygon_area(lid_hull)
            if lid_area <= 1e-6:
                continue

            lid_z = float(lid["pos"][2])
            if not (lid_z > pot_z + Z_ABOVE_EPS):
                continue

            size_ratio = lid_area / pot_area
            if size_ratio < SIZE_RATIO_MIN or size_ratio > SIZE_RATIO_MAX:
                continue

            overlap = compute_overlap_ratio(lid_hull, pot_hull)
            if overlap < OVERLAP_RATIO_MIN - 1e-6:
                continue

            valid_candidates.append(lname)

        return torch.tensor([len(valid_candidates) == 1], dtype=torch.bool, device=self._device)

    def _polygon_area(self, poly: np.ndarray) -> float:
        """Compute polygon area using shoelace formula."""
        if poly is None or len(poly) < 3:
            return 0.0
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def _intersection_area(self, poly1: np.ndarray, poly2: np.ndarray) -> float:
        """Compute intersection area using Sutherland-Hodgman."""
        if poly1 is None or poly2 is None or len(poly1) < 3 or len(poly2) < 3:
            return 0.0

        def ensure_ccw(poly):
            x, y = poly[:, 0], poly[:, 1]
            signed = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            return poly[::-1].copy() if signed < 0 else poly.copy()

        def line_intersection(p1, p2, p3, p4):
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-12:
                return p2
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            return np.array([px, py])

        def is_left(a, b, p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

        output = ensure_ccw(poly1)
        clip = ensure_ccw(poly2)

        for i in range(len(clip)):
            c1, c2 = clip[i], clip[(i + 1) % len(clip)]
            input_list = output
            output = []
            if len(input_list) == 0:
                return 0.0
            S = input_list[-1]
            for E in input_list:
                inside_E = is_left(c1, c2, E) >= 0
                inside_S = is_left(c1, c2, S) >= 0
                if inside_E:
                    if not inside_S:
                        output.append(line_intersection(S, E, c1, c2))
                    output.append(E)
                elif inside_S:
                    output.append(line_intersection(S, E, c1, c2))
                S = E
            if len(output) == 0:
                return 0.0
            output = np.array(output, dtype=float)

        return self._polygon_area(output) if len(output) >= 3 else 0.0

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute dense reward based on milestone-based progress score."""
        objs_info = self.collect_objs_info()

        REACH_THRESHOLD = 0.12
        TOUCH_THRESHOLD = 0.05
        GRASP_GRIPPER_WIDTH = 0.04
        HEIGHT_LIFT_THRESHOLD = 0.02

        POT = "pot without lid"
        LID_MEDIUM = "lid medium"
        LID_SMALL = "small lid"
        LID_LARGE = "lid large"

        INITIAL_HEIGHTS = {
            LID_SMALL: 0.7668,
            LID_MEDIUM: 0.7707,
            LID_LARGE: 0.7752,
        }
        MULTIPLIERS = {
            LID_MEDIUM: 1.0,
            LID_SMALL: 0.6,
            LID_LARGE: 0.6,
        }

        if POT not in objs_info:
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

        def get_hull_2d(name):
            obj = objs_info.get(name)
            if obj is None:
                return None
            hull = obj.get("convex_hull_2d")
            if hull is not None and len(hull) >= 3:
                return [(float(p[0]), float(p[1])) for p in np.asarray(hull)]
            b = obj.get("bounds")
            if b is None:
                return None
            b = np.array(b, dtype=float)
            if b.shape != (2, 3):
                return None
            return [(b[0, 0], b[0, 1]), (b[1, 0], b[0, 1]), (b[1, 0], b[1, 1]), (b[0, 0], b[1, 1])]

        def polygon_area(poly):
            if poly is None or len(poly) < 3:
                return 0.0
            area = 0.0
            for i in range(len(poly)):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % len(poly)]
                area += x1 * y2 - x2 * y1
            return abs(0.5 * area)

        def point_in_polygon(px, py, poly):
            n = len(poly)
            inside = False
            j = n - 1
            for i in range(n):
                xi, yi = poly[i]
                xj, yj = poly[j]
                if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
                    inside = not inside
                j = i
            return inside

        def bounds_within_hull(bounds, hull):
            if bounds is None or hull is None or len(hull) < 3:
                return False
            corners = [
                (bounds[0, 0], bounds[0, 1]),
                (bounds[1, 0], bounds[0, 1]),
                (bounds[1, 0], bounds[1, 1]),
                (bounds[0, 0], bounds[1, 1]),
            ]
            return all(point_in_polygon(cx, cy, hull) for cx, cy in corners)

        def compute_overlap_ratio(lid_hull, pot_hull):
            if lid_hull is None or pot_hull is None:
                return 0.0

            def polygon_area_signed(poly):
                if len(poly) < 3:
                    return 0.0
                area = 0.0
                for i in range(len(poly)):
                    x1, y1 = poly[i]
                    x2, y2 = poly[(i + 1) % len(poly)]
                    area += x1 * y2 - x2 * y1
                return 0.5 * area

            def ensure_ccw(poly):
                if polygon_area_signed(poly) < 0:
                    return poly[::-1]
                return poly

            def line_intersection(p1, p2, p3, p4):
                x1, y1 = p1
                x2, y2 = p2
                x3, y3 = p3
                x4, y4 = p4
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) < 1e-9:
                    return p2
                px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
                py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
                return (px, py)

            def is_inside_edge(p, a, b):
                return ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= -1e-9

            def polygon_clip(subject, clip):
                if len(subject) < 3 or len(clip) < 3:
                    return []
                output = subject[:]
                for i in range(len(clip)):
                    inp = output
                    output = []
                    if not inp:
                        break
                    A = clip[i]
                    B = clip[(i + 1) % len(clip)]
                    S = inp[-1]
                    for E in inp:
                        if is_inside_edge(E, A, B):
                            if not is_inside_edge(S, A, B):
                                output.append(line_intersection(S, E, A, B))
                            output.append(E)
                        elif is_inside_edge(S, A, B):
                            output.append(line_intersection(S, E, A, B))
                        S = E
                return output

            pot_area = polygon_area(pot_hull)
            if pot_area < 1e-8:
                return 0.0
            inter = polygon_clip(ensure_ccw(lid_hull), ensure_ccw(pot_hull))
            inter_area = polygon_area(inter)
            return inter_area / pot_area

        right_ee_pos = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee_pos = self._robot.left_ee_pose[0, :3].cpu().numpy()
        _dof_pos = self._robot.robot_entity.get_dofs_position()
        right_gripper_width = float(_dof_pos[0, self._robot.right_gripper_dofs[0]]) * 2
        left_gripper_width = float(_dof_pos[0, self._robot.left_gripper_dofs[0]]) * 2

        pot_hull = get_hull_2d(POT)
        pot_pos = get_pos(POT) if POT in objs_info else None

        def compute_lid_score(lid_name, multiplier):
            if lid_name not in objs_info:
                return 0.0

            lid_pos = get_pos(lid_name)
            lid_bounds = get_bounds(lid_name)
            lid_hull = get_hull_2d(lid_name)
            initial_height = INITIAL_HEIGHTS.get(lid_name, 0.77)

            dist_right = float(np.linalg.norm(right_ee_pos - lid_pos))
            dist_left = float(np.linalg.norm(left_ee_pos - lid_pos))
            min_dist = min(dist_right, dist_left)
            closest_gripper_width = right_gripper_width if dist_right < dist_left else left_gripper_width

            is_reaching = min_dist <= REACH_THRESHOLD
            is_touching = min_dist <= TOUCH_THRESHOLD
            is_grasping = is_touching and closest_gripper_width < GRASP_GRIPPER_WIDTH

            lid_z = float(lid_pos[2])
            is_lifted = lid_z > initial_height + HEIGHT_LIFT_THRESHOLD

            lid_within_pot = bounds_within_hull(lid_bounds, pot_hull)
            overlap_ratio = compute_overlap_ratio(lid_hull, pot_hull)
            pot_z = float(pot_pos[2]) if pot_pos is not None else 0.0
            lid_on_pot = overlap_ratio >= 0.8 and lid_z > pot_z + 0.01

            if lid_on_pot:
                raw_score = 1.0
            elif lid_within_pot:
                raw_score = 0.9 + overlap_ratio * 0.09
            elif is_lifted or is_grasping:
                raw_score = 0.7 + overlap_ratio * 0.19
            elif is_touching:
                grip_progress = max(0.0, 1.0 - closest_gripper_width / 0.095)
                raw_score = 0.5 + grip_progress * 0.2
            elif is_reaching:
                reach_progress = 1.0 - (min_dist - TOUCH_THRESHOLD) / (REACH_THRESHOLD - TOUCH_THRESHOLD)
                reach_progress = max(0.0, min(1.0, reach_progress))
                raw_score = 0.25 + reach_progress * 0.25
            else:
                if min_dist < 0.3:
                    approach_progress = 1.0 - (min_dist - REACH_THRESHOLD) / (0.3 - REACH_THRESHOLD)
                    approach_progress = max(0.0, min(1.0, approach_progress))
                    raw_score = approach_progress * 0.25
                else:
                    raw_score = 0.0

            return raw_score * multiplier

        score_medium = compute_lid_score(LID_MEDIUM, MULTIPLIERS[LID_MEDIUM])
        score_small = compute_lid_score(LID_SMALL, MULTIPLIERS[LID_SMALL])
        score_large = compute_lid_score(LID_LARGE, MULTIPLIERS[LID_LARGE])

        final_score = max(score_medium, score_small, score_large)
        score_t = torch.tensor(final_score, dtype=torch.float32, device=self._device)
        return score_t, {"score": score_t}
