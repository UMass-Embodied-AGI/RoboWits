"""StandBulb environment for RoboWits.

Task: Stand a tall object with curved bottom stably within a target area.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.placement import _is_fixed_entity
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv
from gs_gym.envs.robowits.utils import check_being_held


@register_task("robowits/16-stand-bulb-v0")
class StandBulbEnv(RoboWitsEnv):
    """Stand a tall object with curved bottom stably within a target area.

    A tall object with curved bottom is prone to tipping due to a curved base.
    A surrounding ring provides lateral support and increases the effective base,
    using geometry for stability.

    Success criteria:
    - Tall object overlaps target patch by at least 80%
    - Tall object is upright (Z extent dominates XY extents)
    - Tall object is stable (low velocity)
    - Objects remain within table bounds
    - Tall object is not being held
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Stand the tall object with curved bottom stably within the green target area."

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
        smaller_area = {"x_min": 0.3, "x_max": 0.8, "y_min": -0.4, "y_max": 0.4}
        self._object_reachable_areas["target patch"] = smaller_area
        self._object_reachable_areas["tall object with curved bottom"] = smaller_area
        # Ring has large footprint (0.25m), so needs wider bounds to reliably find placement
        self._object_reachable_areas["stabilizing ring"] = {"x_min": 0.25, "x_max": 0.85, "y_min": -0.45, "y_max": 0.45}

    @property
    def placement_groups(self) -> PlacementGroups:
        """All objects placed independently.

        Tall object placed before ring: it faces arm exclusion hulls, so placing
        it second (only target patch already placed) gives it more free space than
        placing it last when both target and ring are blocking.
        """
        return ("target patch", "tall object with curved bottom", "stabilizing ring")

    def placement_entity_extra_hulls(self, name: str) -> list:
        """Arm exclusion hulls with zero padding.

        The default 0.05m padding shrinks the inter-arm y-gap from 0.487m to
        0.387m, leaving only a 0.098m y-band for ring/tall-object centers.
        The ring's y-footprint (0.249m) is wider than the half-band (0.049m),
        making placement geometrically impossible. With zero padding the band
        widens to 0.198m, which fits both objects.
        """
        entity_info = self._entities.get(name)
        if entity_info and _is_fixed_entity(entity_info["entity"]):
            return []
        arm_z_min = min(self.ARM_FOREARM_BBOX_RIGHT["z"][0], self.ARM_FOREARM_BBOX_LEFT["z"][0])
        size = (entity_info or {}).get("size") or (0, 0, 0)
        if self.TABLE_HEIGHT + size[2] <= arm_z_min:
            return []

        def _hull(bbox):
            x0, x1 = bbox["x"][0], bbox["x"][1]
            y0, y1 = bbox["y"][0], bbox["y"][1]
            return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)

        return [_hull(self.ARM_FOREARM_BBOX_RIGHT), _hull(self.ARM_FOREARM_BBOX_LEFT)]

    def _add_custom_entities(self) -> None:
        """Add target patch, stabilizing ring, and tall object."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Target patch (cylinder, fixed, no collision)
        target_patch = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.52, 0.0, 0.7625),
                euler=(0.0, 0.0, 0.0),
                radius=0.10,
                height=0.005,
                fixed=True,
                collision=False,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.0, 0.8, 0.1), roughness=0.6),
        )
        self._entities["target patch"] = {
            "entity": target_patch,
        }

        # Stabilizing ring (mesh)
        ring = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/adbecefa-2e19-4438-b6e1-59e6f4390122/obj.glb", pattern_is_dir=False),
                scale=(0.7, 0.7, 0.8),
                pos=(0.45, -0.15, 0.808),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
                decimate=True,
                decimate_face_num=100,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["stabilizing ring"] = {
            "entity": ring,
        }

        # Tall object with curved bottom (mesh)
        bulb = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/4c5e9552-87c0-46e3-8e40-8c736fee4ff1/obj.glb", pattern_is_dir=False),
                scale=2.0,
                pos=(0.60, 0.2, 0.768),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
                decimate=True,
                decimate_face_num=100,
                group_by_material=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["tall object with curved bottom"] = {
            "entity": bulb,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: tall object on target, upright, stable."""
        objs_info = self.collect_objs_info()

        TABLE_X_MIN, TABLE_X_MAX = 0.21, 1.00
        TABLE_Y_MIN, TABLE_Y_MAX = -0.69, 0.69
        VEL_THRESHOLD = 0.05
        OVERLAP_RATIO_REQ = 0.80
        TABLE_MARGIN = 1e-3

        def get_hull(obj):
            hull = obj.get("convex_hull_2d", None)
            if hull is not None and isinstance(hull, np.ndarray) and hull.shape[0] >= 3:
                return hull.copy()
            b = obj.get("bounds", None)
            if b is None:
                return None
            (xmin, ymin, _), (xmax, ymax, _) = b
            return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float)

        def poly_area(poly):
            if poly is None or len(poly) < 3:
                return 0.0
            x = poly[:, 0]
            y = poly[:, 1]
            return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

        def ensure_ccw(poly):
            if poly is None or len(poly) < 3:
                return poly
            area_signed = 0.5 * float(
                np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], -1))
            )
            if area_signed < 0:
                return poly[::-1].copy()
            return poly

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
            return np.array([px, py], dtype=float)

        def is_left(a, b, p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

        def convex_polygon_intersection(subject, clip):
            if subject is None or clip is None or len(subject) < 3 or len(clip) < 3:
                return np.zeros((0, 2), dtype=float)
            output = subject.copy()
            clip = ensure_ccw(clip)
            for i in range(len(clip)):
                c1 = clip[i]
                c2 = clip[(i + 1) % len(clip)]
                input_list = output
                output = []
                if len(input_list) == 0:
                    return np.zeros((0, 2), dtype=float)
                S = input_list[-1]
                for E in input_list:
                    inside_E = is_left(c1, c2, E) >= 0
                    inside_S = is_left(c1, c2, S) >= 0
                    if inside_E:
                        if not inside_S:
                            inter_pt = line_intersection(S, E, c1, c2)
                            output.append(inter_pt)
                        output.append(E)
                    elif inside_S:
                        inter_pt = line_intersection(S, E, c1, c2)
                        output.append(inter_pt)
                    S = E
                if len(output) == 0:
                    return np.zeros((0, 2), dtype=float)
                output = np.array(output, dtype=float)
            return output

        def hull_within_table(hull):
            if hull is None or len(hull) == 0:
                return False
            xs = hull[:, 0]
            ys = hull[:, 1]
            return (
                np.all(xs >= TABLE_X_MIN - TABLE_MARGIN)
                and np.all(xs <= TABLE_X_MAX + TABLE_MARGIN)
                and np.all(ys >= TABLE_Y_MIN - TABLE_MARGIN)
                and np.all(ys <= TABLE_Y_MAX + TABLE_MARGIN)
            )

        tall_name = "tall object with curved bottom"
        patch_name = "target patch"
        if tall_name not in objs_info or patch_name not in objs_info:
            return torch.tensor([False], dtype=torch.bool, device=self._device)
        tall = objs_info[tall_name]
        patch = objs_info[patch_name]

        tall_hull = get_hull(tall)
        patch_hull = get_hull(patch)
        if tall_hull is None or patch_hull is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        if not hull_within_table(tall_hull):
            return torch.tensor([False], dtype=torch.bool, device=self._device)
        if not hull_within_table(patch_hull):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        b = tall.get("bounds", None)
        is_upright = False
        if b is not None:
            (xmin, ymin, zmin), (xmax, ymax, zmax) = b
            z_ext = zmax - zmin
            xy_max = max(xmax - xmin, ymax - ymin)
            is_upright = z_ext > xy_max

        v = tall.get("vel", None)
        is_still = True
        if v is not None and isinstance(v, np.ndarray) and v.shape == (3,):
            is_still = float(np.linalg.norm(v)) <= VEL_THRESHOLD

        if not (is_upright and is_still):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        tall_hull = ensure_ccw(tall_hull)
        patch_hull = ensure_ccw(patch_hull)
        inter_poly = convex_polygon_intersection(tall_hull, patch_hull)
        inter_area = poly_area(inter_poly)
        tall_area = poly_area(tall_hull)
        if tall_area <= 1e-9:
            return torch.tensor([False], dtype=torch.bool, device=self._device)
        overlap_ratio = inter_area / tall_area

        if overlap_ratio + 1e-6 < OVERLAP_RATIO_REQ:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        if check_being_held(tall, self._robot):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        return torch.tensor([True], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute dense reward based on milestone-based progress score."""
        objs_info = self.collect_objs_info()

        RING = "stabilizing ring"
        TALL = "tall object with curved bottom"
        TARGET = "target patch"

        REACH_THRESHOLD = 0.15
        TOUCH_THRESHOLD = 0.05
        MOVE_THRESHOLD = 0.03
        TARGET_TOUCH_THRESHOLD = 0.02

        INITIAL_RING_POS = np.array([0.45, -0.15, 0.808])
        INITIAL_TALL_POS = np.array([0.60, 0.2, 0.768])

        for name in [RING, TALL, TARGET]:
            if name not in objs_info:
                score_t = torch.tensor(0.0, dtype=torch.float32, device=self._device)
                return score_t, {"score": score_t}

        def get_pos(name):
            p = objs_info[name]["pos"]
            return np.array([p[0], p[1], p[2]], dtype=float)

        def get_bounds_xy(obj_name):
            obj = objs_info.get(obj_name)
            if obj is None:
                return None
            b = obj.get("bounds")
            if b is None:
                return None
            b = np.asarray(b, dtype=float)
            if b.shape != (2, 3):
                return None
            (x_min, y_min, _), (x_max, y_max, _) = b
            return (x_min, y_min, x_max, y_max)

        def get_hull_xy(obj_name):
            obj = objs_info.get(obj_name)
            if obj is None:
                return None
            hull = obj.get("convex_hull_2d")
            if hull is not None:
                try:
                    hull = np.asarray(hull, dtype=float)
                    if hull.ndim == 2 and hull.shape[1] == 2 and len(hull) >= 3:
                        return hull
                except Exception:
                    pass
            b = obj.get("bounds")
            if b is None:
                return None
            b = np.asarray(b, dtype=float)
            if b.shape != (2, 3):
                return None
            (x_min, y_min, _), (x_max, y_max, _) = b
            return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=float)

        def polygon_signed_area(poly):
            if poly is None or len(poly) < 3:
                return 0.0
            x = poly[:, 0]
            y = poly[:, 1]
            return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

        def polygon_area(poly):
            return abs(polygon_signed_area(poly))

        def ensure_ccw(poly):
            if poly is None or len(poly) < 3:
                return poly
            if polygon_signed_area(poly) < 0:
                return np.flipud(poly.copy())
            return poly

        def sutherland_hodgman(subject_polygon, clip_polygon):
            if subject_polygon is None or clip_polygon is None:
                return np.zeros((0, 2), dtype=float)
            output_list = subject_polygon.tolist()
            if len(output_list) < 3 or len(clip_polygon) < 3:
                return np.zeros((0, 2), dtype=float)

            def is_inside(p, a, b):
                return ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= 0.0

            def compute_intersection(p1, p2, p3, p4):
                x1, y1 = p1
                x2, y2 = p2
                x3, y3 = p3
                x4, y4 = p4
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) < 1e-12:
                    return [(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0]
                px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
                py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
                return [px, py]

            clip_pts = clip_polygon.tolist()
            for i in range(len(clip_pts)):
                input_list = output_list
                output_list = []
                if not input_list:
                    break
                A = clip_pts[i]
                B = clip_pts[(i + 1) % len(clip_pts)]
                S = input_list[-1]
                for E in input_list:
                    if is_inside(E, A, B):
                        if not is_inside(S, A, B):
                            output_list.append(compute_intersection(S, E, A, B))
                        output_list.append(E)
                    elif is_inside(S, A, B):
                        output_list.append(compute_intersection(S, E, A, B))
                    S = E
            if not output_list or len(output_list) < 3:
                return np.zeros((0, 2), dtype=float)
            return np.array(output_list, dtype=float)

        def compute_overlap_ratio(obj_name):
            obj_hull = get_hull_xy(obj_name)
            target_hull = get_hull_xy(TARGET)
            if obj_hull is None or target_hull is None:
                return 0.0
            obj_area = polygon_area(ensure_ccw(obj_hull))
            if obj_area <= 1e-8:
                return 0.0
            p1 = ensure_ccw(np.asarray(obj_hull, dtype=float))
            p2 = ensure_ccw(np.asarray(target_hull, dtype=float))
            inter = sutherland_hodgman(p1, p2)
            if inter.size == 0 or len(inter) < 3:
                return 0.0
            inter_area = polygon_area(inter)
            return inter_area / obj_area

        def is_bbox_intersecting_xy(obj_bbox, target_bbox):
            if obj_bbox is None or target_bbox is None:
                return False
            obj_x_min, obj_y_min, obj_x_max, obj_y_max = obj_bbox
            target_x_min, target_y_min, target_x_max, target_y_max = target_bbox
            return not (
                obj_x_max < target_x_min
                or obj_x_min > target_x_max
                or obj_y_max < target_y_min
                or obj_y_min > target_y_max
            )

        def compute_object_score(obj_name, initial_pos, min_dist_to_obj):
            score = 0.0
            obj_pos = get_pos(obj_name)
            target_pos = get_pos(TARGET)

            if min_dist_to_obj <= REACH_THRESHOLD:
                score += 0.05
            if min_dist_to_obj <= TOUCH_THRESHOLD:
                score += 0.1

            displacement = float(np.linalg.norm(obj_pos - initial_pos))
            if displacement >= MOVE_THRESHOLD:
                score += 0.15

            if obj_name == RING:
                obj_to_target_dist = float(np.linalg.norm(obj_pos[:2] - target_pos[:2]))
                overlap_ratio = compute_overlap_ratio(obj_name)
                if overlap_ratio > 0.01 or obj_to_target_dist <= TARGET_TOUCH_THRESHOLD:
                    score += 0.1
            elif obj_name == TALL:
                obj_bbox = get_bounds_xy(obj_name)
                target_bbox = get_bounds_xy(TARGET)
                if is_bbox_intersecting_xy(obj_bbox, target_bbox):
                    score += 0.1

            return score

        right_ee_pos = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee_pos = self._robot.left_ee_pose[0, :3].cpu().numpy()

        ring_pos = get_pos(RING)
        tall_pos = get_pos(TALL)

        dist_right_to_ring = float(np.linalg.norm(right_ee_pos - ring_pos))
        dist_left_to_ring = float(np.linalg.norm(left_ee_pos - ring_pos))
        min_dist_to_ring = min(dist_right_to_ring, dist_left_to_ring)

        dist_right_to_tall = float(np.linalg.norm(right_ee_pos - tall_pos))
        dist_left_to_tall = float(np.linalg.norm(left_ee_pos - tall_pos))
        min_dist_to_tall = min(dist_right_to_tall, dist_left_to_tall)

        ring_score = compute_object_score(RING, INITIAL_RING_POS, min_dist_to_ring)
        tall_score = compute_object_score(TALL, INITIAL_TALL_POS, min_dist_to_tall)

        bonus = 0.0
        tall_obj = objs_info[TALL]
        tall_vel = tall_obj.get("vel")
        tall_bounds = tall_obj.get("bounds")
        if tall_bounds is not None:
            (xmin, ymin, zmin), (xmax, ymax, zmax) = tall_bounds
            z_ext = zmax - zmin
            xy_max = max(xmax - xmin, ymax - ymin)
            is_upright = z_ext > xy_max
            is_stable = True
            if tall_vel is not None and isinstance(tall_vel, np.ndarray) and tall_vel.shape == (3,):
                is_stable = float(np.linalg.norm(tall_vel)) <= 0.05
            if is_upright and is_stable and not check_being_held(tall_obj, self._robot):
                bonus = 0.1

        tall_overlap_ratio = compute_overlap_ratio(TALL)
        tall_overlap_bonus = 0.05 if tall_overlap_ratio > 0.50 else 0.0

        ring_overlap_ratio = compute_overlap_ratio(RING)
        ring_overlap_bonus = 0.05 if ring_overlap_ratio > 0.50 else 0.0

        total_score = ring_score + tall_score + bonus + tall_overlap_bonus + ring_overlap_bonus
        score_t = torch.tensor(total_score, dtype=torch.float32, device=self._device)
        return score_t, {"score": score_t}
