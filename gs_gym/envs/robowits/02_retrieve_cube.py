"""RetrieveCube environment for RoboWits.

Task: Retrieve a cube from a deep narrow container and place it on a target area.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv


@register_task("robowits/02-retrieve-cube-v0")
class RetrieveCubeEnv(RoboWitsEnv):
    """Retrieve a cube from a deep narrow container and place it on a target area.

    The cube is initially inside a deep narrow container that the robot gripper
    cannot directly reach. The robot must tilt or manipulate the container to
    pour the cube out, then move it to the target area.

    Success criteria:
    - Cube is on the table surface (z within tolerance of TABLE_Z)
    - Cube is outside the container (2D footprint overlap with container < 50%)
    - Cube's 2D footprint overlaps target area by at least 50%
    - Objects remain within table bounds

    Note: Works with both "deep narrow container" and "wide open container" variants.
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Retrieve the cube from the container and place it on the green target area."

    # Table height
    TABLE_Z = 0.76

    # Cube configuration
    CUBE_SIZE = (0.015, 0.015, 0.015)

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
        """Placement groups with nested tuple: cube stays relative to container."""
        return (("deep narrow container", "cube"), "target area")

    def _add_custom_entities(self) -> None:
        """Add container, cube, and target area to the scene."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Deep narrow container (mesh)
        container = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path(
                    "blender_kit/ddfb928e-c26d-4805-b41a-ab1545269e99/obj.glb",
                    pattern_is_dir=False,
                ),
                scale=0.6,
                pos=(0.50, 0.00, 0.83992),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Glass(color=(0.9, 0.95, 1.0), double_sided=True),
        )
        self._entities["deep narrow container"] = {
            "entity": container,
        }

        # Cube (inside container initially)
        cube = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.50, 0.00, 0.79),
                euler=(0, 0, 0),
                size=self.CUBE_SIZE,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(color=(0.8, 0.6, 0.4), double_sided=True),
        )
        self._entities["cube"] = {
            "entity": cube,
        }

        # Target area (fixed marker)
        target = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.68, 0.12, 0.761),
                euler=(0, 0, 0),
                size=(0.12, 0.12, 0.002),
                fixed=True,
                collision=False,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(color=(0.1, 0.8, 0.1), double_sided=True),
        )
        self._entities["target area"] = {
            "entity": target,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful."""

        objs_info = self.collect_objs_info()

        TABLE_X_MIN, TABLE_X_MAX = 0.21, 1.00
        TABLE_Y_MIN, TABLE_Y_MAX = -0.69, 0.69
        TABLE_Z = 0.76

        CUBE = "cube"
        TARGET = "target area"
        CONTAINERS = ["deep narrow container", "wide open container"]

        EPS_XY = 1e-2
        Z_TOL_BELOW = 0.01
        Z_TOL_ABOVE = 0.03
        AREA_EPS = 1e-8

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
            return np.array(
                [
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max],
                ],
                dtype=float,
            )

        def get_bounds(obj_name):
            obj = objs_info.get(obj_name)
            if obj is None:
                return None
            b = obj.get("bounds")
            if b is None:
                return None
            b = np.asarray(b, dtype=float)
            if b.shape != (2, 3):
                return None
            return b

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
                return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0.0

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

        def overlap_area(poly1, poly2):
            if poly1 is None or poly2 is None:
                return 0.0
            p1 = ensure_ccw(np.asarray(poly1, dtype=float))
            p2 = ensure_ccw(np.asarray(poly2, dtype=float))
            inter = sutherland_hodgman(p1, p2)
            if inter.size == 0 or len(inter) < 3:
                return 0.0
            return polygon_area(inter)

        def hull_inside_table(hull):
            if hull is None or len(hull) == 0:
                return False
            xs = hull[:, 0]
            ys = hull[:, 1]
            within_x = (xs >= (TABLE_X_MIN - EPS_XY)) & (xs <= (TABLE_X_MAX + EPS_XY))
            within_y = (ys >= (TABLE_Y_MIN - EPS_XY)) & (ys <= (TABLE_Y_MAX + EPS_XY))
            return bool(np.all(within_x & within_y))

        cube_hull = get_hull_xy(CUBE)
        target_hull = get_hull_xy(TARGET)
        cube_bounds = get_bounds(CUBE)

        # Get container hull (try both container types)
        container_hull = None
        for container_name in CONTAINERS:
            container_hull = get_hull_xy(container_name)
            if container_hull is not None:
                break

        if cube_hull is None or target_hull is None or cube_bounds is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        z_min = float(cube_bounds[0, 2])
        if z_min < (TABLE_Z - Z_TOL_BELOW) or z_min > (TABLE_Z + Z_TOL_ABOVE):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        if not hull_inside_table(cube_hull):
            return torch.tensor([False], dtype=torch.bool, device=self._device)
        if not hull_inside_table(target_hull):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        cube_area = polygon_area(ensure_ccw(cube_hull))
        if cube_area <= AREA_EPS:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        # Check cube is outside container (overlap < 50%)
        if container_hull is not None:
            container_overlap = overlap_area(cube_hull, container_hull)
            container_overlap_ratio = container_overlap / cube_area
            if container_overlap_ratio >= 0.5:
                return torch.tensor([False], dtype=torch.bool, device=self._device)

        # Check cube overlaps target area by at least 50%
        inter_area = overlap_area(cube_hull, target_hull)
        overlap_ratio = inter_area / cube_area

        return torch.tensor([overlap_ratio >= 0.5], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute dense reward based on milestone-based progress score."""

        objs_info = self.collect_objs_info()

        TABLE_Z = 0.76
        Z_TOL_BELOW = 0.01
        CUBE_ON_TABLE_Z_MAX = 0.78
        TARGET_OVERLAP_COMPLETE = 0.5
        TARGET_OVERLAP_PARTIAL = 0.25
        MAX_TRANSPORT_DIST = 0.40

        _CONTAINER_NAMES = ["deep narrow container", "wide open container"]
        CONTAINER = next((c for c in _CONTAINER_NAMES if c in objs_info), None)
        CUBE = "cube"
        TARGET = "target area"

        TARGET_CENTER = np.array([0.68, 0.12], dtype=float)

        if CONTAINER is None or CUBE not in objs_info or TARGET not in objs_info:
            score_t = torch.tensor(0.0, dtype=torch.float32, device=self._device)
            return score_t, {"score": score_t}

        def get_pos(name):
            p = objs_info[name]["pos"]
            return np.array([float(p[0]), float(p[1]), float(p[2])], dtype=float)

        def get_hull_2d(name):
            obj = objs_info.get(name)
            if obj is None:
                return None
            hull = obj.get("convex_hull_2d")
            if hull is not None and len(hull) >= 3:
                return np.array([[float(pt[0]), float(pt[1])] for pt in hull])
            b = obj.get("bounds")
            if b is None:
                return None
            b = np.array(b, dtype=float)
            if b.shape != (2, 3):
                return None
            return np.array(
                [[b[0, 0], b[0, 1]], [b[1, 0], b[0, 1]], [b[1, 0], b[1, 1]], [b[0, 0], b[1, 1]]], dtype=float
            )

        def poly_signed_area(poly):
            if poly is None or len(poly) < 3:
                return 0.0
            x, y = poly[:, 0], poly[:, 1]
            return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

        def poly_area(poly):
            return abs(poly_signed_area(poly))

        def ensure_ccw(poly):
            if poly is None or len(poly) < 3:
                return poly
            return poly if poly_signed_area(poly) > 0 else np.flipud(poly.copy())

        def sutherland_hodgman(subject, clip):
            if subject is None or clip is None or len(subject) < 3 or len(clip) < 3:
                return np.zeros((0, 2), dtype=float)
            out = subject.tolist()
            clip_pts = clip.tolist()

            def is_inside(p, a, b):
                return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0.0

            def inter(s1, s2, a, b):
                x1, y1 = s1[0], s1[1]
                x2, y2 = s2[0], s2[1]
                x3, y3 = a[0], a[1]
                x4, y4 = b[0], b[1]
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) < 1e-12:
                    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
                px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
                py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
                return [px, py]

            for i in range(len(clip_pts)):
                A, B = clip_pts[i], clip_pts[(i + 1) % len(clip_pts)]
                new_out, S = [], out[-1]
                for E in out:
                    if is_inside(E, A, B):
                        if not is_inside(S, A, B):
                            new_out.append(inter(S, E, A, B))
                        new_out.append(E)
                    elif is_inside(S, A, B):
                        new_out.append(inter(S, E, A, B))
                    S = E
                out = new_out
                if len(out) < 3:
                    return np.zeros((0, 2), dtype=float)
            return np.array(out, dtype=float)

        def overlap_ratio(hull1, hull2):
            if hull1 is None or hull2 is None:
                return 0.0
            a1 = poly_area(ensure_ccw(np.asarray(hull1, dtype=float)))
            if a1 < 1e-8:
                return 0.0
            inter = sutherland_hodgman(
                ensure_ccw(np.asarray(hull1, dtype=float)), ensure_ccw(np.asarray(hull2, dtype=float))
            )
            return poly_area(inter) / a1 if len(inter) >= 3 else 0.0

        right_ee_pos = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee_pos = self._robot.left_ee_pose[0, :3].cpu().numpy()

        container_pos = get_pos(CONTAINER)
        cube_pos = get_pos(CUBE)

        dist_ee_container = min(
            float(np.linalg.norm(right_ee_pos - container_pos)),
            float(np.linalg.norm(left_ee_pos - container_pos)),
        )

        cube_xy = cube_pos[:2]
        cube_on_table = (TABLE_Z - Z_TOL_BELOW) <= cube_pos[2] <= CUBE_ON_TABLE_Z_MAX
        cube_to_target_dist = float(np.linalg.norm(cube_xy - TARGET_CENTER))

        cube_hull = get_hull_2d(CUBE)
        target_hull = get_hull_2d(TARGET)
        container_hull = get_hull_2d(CONTAINER)

        cube_container_overlap = (
            overlap_ratio(cube_hull, container_hull) if cube_hull is not None and container_hull is not None else 1.0
        )
        cube_out_of_container = cube_container_overlap < 0.5
        target_overlap = (
            overlap_ratio(cube_hull, target_hull) if cube_hull is not None and target_hull is not None else 0.0
        )

        def lerp(lo, hi, t):
            return lo + (hi - lo) * max(0.0, min(1.0, t))

        # Approach score (0→0.33): EE distance to container
        approach_score = lerp(0.0, 0.33, 1.0 - min(1.0, dist_ee_container / 0.3))

        # Extraction score (0→0.60): rises as cube escapes container; zeroed if cube fell off table
        cube_above_floor = cube_pos[2] >= TABLE_Z - 0.20
        extraction_score = lerp(0.0, 0.60, 1.0 - cube_container_overlap) if cube_above_floor else 0.0

        # Transport score (0.60→0.90): distance to target, only when cube is free and on table
        if cube_out_of_container and cube_on_table:
            dist_progress = 1.0 - min(1.0, cube_to_target_dist / MAX_TRANSPORT_DIST)
            transport_score = lerp(0.60, 0.90, dist_progress)
        else:
            transport_score = 0.0

        if cube_out_of_container and cube_on_table and target_overlap >= TARGET_OVERLAP_COMPLETE:
            score = 1.0
        elif cube_out_of_container and cube_on_table and target_overlap >= TARGET_OVERLAP_PARTIAL:
            score = lerp(
                0.90,
                1.0,
                (target_overlap - TARGET_OVERLAP_PARTIAL) / (TARGET_OVERLAP_COMPLETE - TARGET_OVERLAP_PARTIAL),
            )
        else:
            score = max(approach_score, extraction_score, transport_score)

        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"score": score_t}
