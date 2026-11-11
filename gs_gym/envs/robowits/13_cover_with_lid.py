"""CoverWithLid environment for RoboWits.

Task: Cover a pot precisely using the correct sized lid.
"""

from __future__ import annotations

import os

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

    This requires reasoning about cross-sectional fit. Only the correctly sized circular lid will cover the
    opening with minimal excess. The circular geometry and ring reference enable precise centering without
    measurements.

    Success criteria, judged against the pot's opening (the hole inside its rolled lip), not its footprint.
    Exactly one object in COVER_CANDIDATES must satisfy all of:
    - It is above the pot
    - It covers at least 80% of the opening
    - It is larger than the opening but no wider than the lip: lid area / opening area in [1, (lip / hole)^2]
    - It is not being held by either gripper
    """

    # Objects judged as covers. The tool-substitution mutations add their plate.
    COVER_CANDIDATES: tuple[str, ...] = ("small_lid", "lid_medium", "lid_large")

    # Unit-scale radii (m) of each pot mesh's opening ("hole": inner edge of the lip) and of the lip's
    # outer edge, keyed by BlenderKit id. Measured as the min/max vertex radius within the top 5% of the
    # mesh's height (Z-up); both meshes are centred on their origin in xy.
    POT_OPENINGS = {
        "3aee9353-e21a-40d4-b160-e2a0af1fff7f": (0.2695, 0.3574),
        "2cf9e625-d115-46a9-98af-c771663006db": (0.1404, 0.2391),
    }

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
        return ("pot_without_lid", "small_lid", "lid_medium", "lid_large")

    @property
    def _object_reachable_areas(self) -> dict:
        # 'lid_large' starts at y=0.44 and has a large footprint that exceeds the default
        # y[-0.25, 0.25] bounds. Widen y for all lids so placement sampling succeeds.
        return {
            "small_lid": {"y_min": -0.45, "y_max": 0.45},
            "lid_medium": {"y_min": -0.45, "y_max": 0.45},
            "lid_large": {"y_min": -0.45, "y_max": 0.45},
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
        self._entities["pot_without_lid"] = {
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
        self._entities["small_lid"] = {
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
        self._entities["lid_medium"] = {
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
        self._entities["lid_large"] = {
            "entity": large_lid,
        }

    def _pot_opening(self, pot_info: dict) -> tuple[np.ndarray, float]:
        """Return the pot's opening as a (64, 2) xy circle and the max lid/opening area ratio (lip / hole)^2."""
        morph = self.get_entity("pot_without_lid").morph
        hole_r, lip_r = self.POT_OPENINGS[os.path.basename(os.path.dirname(morph.file))]
        scale = float(np.atleast_1d(morph.scale)[0])
        theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        circle = hole_r * scale * np.stack([np.cos(theta), np.sin(theta)], axis=1)
        return circle + np.asarray(pot_info["pos"][:2], dtype=float), (lip_r / hole_r) ** 2

    def _lid_fits(self, lid_hull, opening: np.ndarray, size_ratio_max: float) -> bool:
        """Whether a lid's xy hull is larger than the pot's opening but no wider than its lip (by area)."""
        if lid_hull is None or len(lid_hull) < 3:
            return False
        size_ratio = self._polygon_area(np.asarray(lid_hull, dtype=float)) / self._polygon_area(opening)
        return 1.0 <= size_ratio <= size_ratio_max

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: exactly one lid covers the pot's opening correctly."""
        objs_info = self.collect_objs_info()

        TABLE_X_MIN, TABLE_X_MAX = 0.21, 1.00
        TABLE_Y_MIN, TABLE_Y_MAX = -0.69, 0.69
        COVERAGE_MIN = 0.80
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

        pot = objs_info.get("pot_without_lid")
        if pot is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        pot_hull = get_hull(pot)
        if pot_hull is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        if not is_within_table_bounds(pot_hull):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        opening, size_ratio_max = self._pot_opening(pot)
        opening_area = self._polygon_area(opening)

        pot_z = float(pot["pos"][2])

        valid_candidates = []

        for lname in self.COVER_CANDIDATES:
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

            if not self._lid_fits(lid_hull, opening, size_ratio_max):
                continue

            coverage = _utils.intersection_area(lid_hull, opening) / opening_area
            if coverage < COVERAGE_MIN - 1e-6:
                continue

            if not _utils.check_released(lid, self._robot):
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

        POT = "pot_without_lid"

        FIT_MULTIPLIER = 1.0  # a movable cover that passes _check_success's size check
        MISFIT_MULTIPLIER = 0.6

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
        # Overlap is measured as coverage of the pot's opening, and the full multiplier goes to the movable
        # cover that passes the size check, both matching _check_success. A cover fixed to the table (the
        # intended lid in the tool-substitution mutations) can never be used, so it gets the misfit multiplier.
        opening_np, size_ratio_max = self._pot_opening(objs_info[POT])
        opening = [(float(x), float(y)) for x, y in opening_np]

        def multiplier(lid_name):
            fits = lid_name in objs_info and self._lid_fits(get_hull_2d(lid_name), opening_np, size_ratio_max)
            fixed = self.get_entity(lid_name).morph.fixed
            return FIT_MULTIPLIER if fits and not fixed else MISFIT_MULTIPLIER

        def compute_lid_score(lid_name, multiplier):
            if lid_name not in objs_info:
                return 0.0

            lid_pos = get_pos(lid_name)
            lid_bounds = get_bounds(lid_name)
            lid_hull = get_hull_2d(lid_name)
            initial_height = float(self.get_entity(lid_name).morph.pos[2])

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
            overlap_ratio = compute_overlap_ratio(lid_hull, opening)
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

        final_score = max(compute_lid_score(name, multiplier(name)) for name in self.COVER_CANDIDATES)
        score_t = torch.tensor(final_score, dtype=torch.float32, device=self._device)
        return score_t, {"score": score_t}
