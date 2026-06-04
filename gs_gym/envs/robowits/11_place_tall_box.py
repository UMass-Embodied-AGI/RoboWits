"""PlaceTallBox environment for RoboWits.

Task: Move a tall box onto a target area under a beam obstacle.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv


@register_task("robowits/11-place-tall-box-v0")
class PlaceTallBoxEnv(RoboWitsEnv):
    """Move a tall box onto a target area under a beam obstacle.

    The obstacle enforces a geometric clearance constraint. The beam cannot
    be moved or raised. The solution relies on changing orientation to reduce
    the effective height, exploiting the different cross-sectional dimensions
    of the box to fit under the low beam.

    Success criteria:
    - Tall box overlaps target area by at least 80%
    - Both objects remain within table bounds
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Move the rectangular box onto the green target area."

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
        """Tall box independent, obstacles grouped together."""
        return ("tall cardboard box", ("support block left", "support block right", "beam plank", "target area"))

    def _add_custom_entities(self) -> None:
        """Add tall box, support blocks, beam, and target area."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Tall cardboard box (mesh)
        box = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/8fb31a9b-33d5-4246-997f-84307520c1a0/obj.glb", pattern_is_dir=False),
                scale=1.2,
                pos=(0.42, 0.0, 0.8422),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["tall cardboard box"] = {
            "entity": box,
        }

        # Support block left (fixed)
        support_left = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.55, -0.15, 0.8025),
                euler=(0.0, 0.0, 0.0),
                size=(0.10, 0.06, 0.085),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.6, 0.6, 0.6), roughness=0.6, ior=1.5),
        )
        self._entities["support block left"] = {
            "entity": support_left,
        }

        # Support block right (fixed)
        support_right = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.55, 0.15, 0.8025),
                euler=(0.0, 0.0, 0.0),
                size=(0.10, 0.06, 0.085),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.6, 0.6, 0.6), roughness=0.6, ior=1.5),
        )
        self._entities["support block right"] = {
            "entity": support_right,
        }

        # Beam plank (fixed)
        beam = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.55, 0.0, 0.855),
                euler=(0.0, 0.0, 0.0),
                size=(0.12, 0.36, 0.02),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.7, 0.55, 0.3), roughness=0.7, ior=1.5),
        )
        self._entities["beam plank"] = {
            "entity": beam,
        }

        # Target area
        target = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.55, 0.0, 0.761),
                euler=(0.0, 0.0, 0.0),
                size=(0.10, 0.12, 0.002),
                fixed=True,
                collision=False,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.2, 0.8, 0.2), roughness=0.4, ior=1.5),
        )
        self._entities["target area"] = {
            "entity": target,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: box overlaps target by 80%."""
        objs_info = self.collect_objs_info()

        TABLE_BOUNDS = {"x_min": 0.21, "x_max": 1.00, "y_min": -0.69, "y_max": 0.69}
        EPS = 1e-6

        box_name = "tall cardboard box"
        target_name = "target area"
        if box_name not in objs_info or target_name not in objs_info:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        box = objs_info[box_name]
        tgt = objs_info[target_name]

        box_hull = box.get("convex_hull_2d", None)
        tgt_hull = tgt.get("convex_hull_2d", None)
        if box_hull is None or tgt_hull is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        def clean_polygon(poly: np.ndarray) -> np.ndarray:
            poly = np.asarray(poly, dtype=float).reshape(-1, 2)
            if len(poly) >= 2 and np.linalg.norm(poly[0] - poly[-1]) < 1e-9:
                poly = poly[:-1]
            if len(poly) > 1:
                dedup = [poly[0]]
                for p in poly[1:]:
                    if np.linalg.norm(p - dedup[-1]) > 1e-9:
                        dedup.append(p)
                poly = np.array(dedup)
            return poly

        box_hull = clean_polygon(box_hull)
        tgt_hull = clean_polygon(tgt_hull)
        if len(box_hull) < 3 or len(tgt_hull) < 3:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        def poly_area(poly: np.ndarray) -> float:
            x = poly[:, 0]
            y = poly[:, 1]
            return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

        def signed_area(poly: np.ndarray) -> float:
            x = poly[:, 0]
            y = poly[:, 1]
            return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

        def ensure_ccw(poly: np.ndarray) -> np.ndarray:
            return poly if signed_area(poly) > 0 else poly[::-1].copy()

        def line_intersection(p1, p2, q1, q2):
            p1 = np.asarray(p1, float)
            p2 = np.asarray(p2, float)
            q1 = np.asarray(q1, float)
            q2 = np.asarray(q2, float)
            r = p2 - p1
            s = q2 - q1
            rxs = r[0] * s[1] + (-r[1]) * s[0]
            q_p = q1 - p1
            if abs(rxs) < 1e-12:
                return (p2 + q2) / 2.0
            t = (q_p[0] * s[1] - q_p[1] * s[0]) / rxs
            return p1 + t * r

        def suth_hodg_intersection(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
            output = subject.copy()
            clip = ensure_ccw(clip)
            if len(output) == 0:
                return output
            for i in range(len(clip)):
                A = clip[i]
                B = clip[(i + 1) % len(clip)]
                new_output = []
                if len(output) == 0:
                    return output
                S = output[-1]
                for E in output:

                    def is_inside(P, A=A, B=B):
                        return ((B[0] - A[0]) * (P[1] - A[1]) - (B[1] - A[1]) * (P[0] - A[0])) >= -1e-12

                    S_inside = is_inside(S)
                    E_inside = is_inside(E)
                    if E_inside:
                        if not S_inside:
                            inter_pt = line_intersection(S, E, A, B)
                            new_output.append(inter_pt)
                        new_output.append(E)
                    elif S_inside:
                        inter_pt = line_intersection(S, E, A, B)
                        new_output.append(inter_pt)
                    S = E
                output = np.array(new_output, dtype=float) if len(new_output) else np.zeros((0, 2), dtype=float)
            return output

        def hull_within_table(hull: np.ndarray) -> bool:
            if hull.size == 0:
                return False
            x_min, x_max = TABLE_BOUNDS["x_min"], TABLE_BOUNDS["x_max"]
            y_min, y_max = TABLE_BOUNDS["y_min"], TABLE_BOUNDS["y_max"]
            xs = hull[:, 0]
            ys = hull[:, 1]
            return (
                xs.min() >= x_min - 1e-3
                and xs.max() <= x_max + 1e-3
                and ys.min() >= y_min - 1e-3
                and ys.max() <= y_max + 1e-3
            )

        if not (hull_within_table(box_hull) and hull_within_table(tgt_hull)):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        box_area = poly_area(box_hull)
        tgt_area = poly_area(tgt_hull)
        if box_area < EPS or tgt_area < EPS:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        inter_poly = suth_hodg_intersection(ensure_ccw(box_hull), ensure_ccw(tgt_hull))
        inter_area = poly_area(inter_poly) if inter_poly is not None and len(inter_poly) >= 3 else 0.0

        coverage = inter_area / max(min(box_area, tgt_area), EPS)

        return torch.tensor([coverage > 0.6], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute dense reward based on milestone-based progress score."""
        objs_info = self.collect_objs_info()

        REACH_THRESHOLD = 0.12
        TOUCH_THRESHOLD = 0.05
        ROTATION_THRESHOLD = 30.0

        BOX = "tall cardboard box"
        TARGET = "target area"
        SUPPORT_LEFT = "support block left"
        SUPPORT_RIGHT = "support block right"
        PLANK = "beam plank"
        OBSTACLES = [SUPPORT_LEFT, SUPPORT_RIGHT, PLANK]
        TABLE_Z = 0.76

        required_objects = [BOX, TARGET]
        for n in required_objects:
            if n not in objs_info:
                score_t = torch.tensor(0.0, dtype=torch.float32, device=self._device)
                return score_t, {"score": score_t}

        def get_pos(name):
            p = objs_info[name]["pos"]
            return np.array([p[0], p[1], p[2]], dtype=float)

        def get_euler(name):
            e = objs_info[name].get("euler")
            if e is None:
                return None
            return np.array([e[0], e[1], e[2]], dtype=float)

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

        def poly_area(poly):
            if poly is None or len(poly) < 3:
                return 0.0
            x = poly[:, 0]
            y = poly[:, 1]
            return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

        def compute_overlap_ratio(hull1, hull2):
            if hull1 is None or hull2 is None or len(hull1) < 3 or len(hull2) < 3:
                return 0.0

            def signed_area(poly):
                x = poly[:, 0]
                y = poly[:, 1]
                return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

            def ensure_ccw(poly):
                return poly if signed_area(poly) > 0 else poly[::-1].copy()

            def line_intersection(p1, p2, q1, q2):
                p1 = np.asarray(p1, float)
                p2 = np.asarray(p2, float)
                q1 = np.asarray(q1, float)
                q2 = np.asarray(q2, float)
                r = p2 - p1
                s = q2 - q1
                rxs = r[0] * s[1] - r[1] * s[0]
                q_p = q1 - p1
                if abs(rxs) < 1e-12:
                    return (p2 + q2) / 2.0
                t = (q_p[0] * s[1] - q_p[1] * s[0]) / rxs
                return p1 + t * r

            def suth_hodg(subject, clip):
                output = subject.copy()
                clip = ensure_ccw(clip)
                for i in range(len(clip)):
                    A = clip[i]
                    B = clip[(i + 1) % len(clip)]
                    new_output = []
                    if len(output) == 0:
                        return np.zeros((0, 2), dtype=float)
                    S = output[-1]
                    for E in output:

                        def is_inside(P, _A=A, _B=B):
                            return ((_B[0] - _A[0]) * (P[1] - _A[1]) - (_B[1] - _A[1]) * (P[0] - _A[0])) >= -1e-12

                        if is_inside(E):
                            if not is_inside(S):
                                new_output.append(line_intersection(S, E, A, B))
                            new_output.append(E)
                        elif is_inside(S):
                            new_output.append(line_intersection(S, E, A, B))
                        S = E
                    output = np.array(new_output, dtype=float) if len(new_output) else np.zeros((0, 2), dtype=float)
                return output

            area1 = poly_area(hull1)
            area2 = poly_area(hull2)
            denom = min(area1, area2)
            if denom < 1e-8:
                return 0.0
            inter = suth_hodg(ensure_ccw(hull1), ensure_ccw(hull2))
            inter_area = poly_area(inter) if len(inter) >= 3 else 0.0
            return inter_area / denom

        def bounds_touching(name1, name2):
            b1 = get_bounds(name1)
            b2 = get_bounds(name2)
            if b1 is None or b2 is None:
                return False
            tol = 0.01
            overlap_x = b1[1, 0] >= b2[0, 0] - tol and b2[1, 0] >= b1[0, 0] - tol
            overlap_y = b1[1, 1] >= b2[0, 1] - tol and b2[1, 1] >= b1[0, 1] - tol
            overlap_z = b1[1, 2] >= b2[0, 2] - tol and b2[1, 2] >= b1[0, 2] - tol
            return overlap_x and overlap_y and overlap_z

        right_ee_pos = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee_pos = self._robot.left_ee_pose[0, :3].cpu().numpy()

        box_pos = get_pos(BOX)
        box_euler = get_euler(BOX)
        box_bounds = get_bounds(BOX)
        box_hull = get_hull_2d(BOX)
        target_hull = get_hull_2d(TARGET)

        dist_right_box = float(np.linalg.norm(right_ee_pos - box_pos))
        dist_left_box = float(np.linalg.norm(left_ee_pos - box_pos))
        min_dist_box = min(dist_right_box, dist_left_box)

        min_dist_obstacles = float("inf")
        for obs_name in OBSTACLES:
            if obs_name in objs_info:
                obs_pos = get_pos(obs_name)
                dist_right = float(np.linalg.norm(right_ee_pos - obs_pos))
                dist_left = float(np.linalg.norm(left_ee_pos - obs_pos))
                min_dist_obstacles = min(min_dist_obstacles, dist_right, dist_left)

        rotation_amount = 0.0
        if box_euler is not None:
            euler_diff = np.abs(box_euler)
            euler_diff = np.minimum(euler_diff, 360.0 - euler_diff)
            rotation_amount = float(np.max(euler_diff))

        box_rotated_30 = rotation_amount > ROTATION_THRESHOLD

        largest_face_on_table = False
        if box_bounds is not None:
            box_size = box_bounds[1] - box_bounds[0]
            z_extent = float(box_size[2])
            xy_extent = max(float(box_size[0]), float(box_size[1]))
            box_z_min = float(box_bounds[0, 2])
            on_table = abs(box_z_min - TABLE_Z) < 0.05
            largest_face_on_table = on_table and z_extent < xy_extent * 0.8

        box_touches_target = bounds_touching(BOX, TARGET)
        overlap_ratio = compute_overlap_ratio(box_hull, target_hull)
        box_on_target_60 = overlap_ratio >= 0.6

        correct_score = 0.0
        if box_on_target_60:
            correct_score = 1.0
        elif box_touches_target:
            correct_score = 0.9 + overlap_ratio * 0.1
        elif largest_face_on_table:
            correct_score = 0.9 if box_touches_target else 0.8
        elif box_rotated_30:
            rotation_progress = min(1.0, rotation_amount / 90.0)
            correct_score = 0.6 + rotation_progress * 0.19
        else:
            is_reaching_box = min_dist_box <= REACH_THRESHOLD
            is_touching_box = min_dist_box <= TOUCH_THRESHOLD

            if is_touching_box:
                rotation_progress = min(1.0, rotation_amount / ROTATION_THRESHOLD)
                correct_score = 0.4 + rotation_progress * 0.19
            elif is_reaching_box:
                reach_progress = 1.0 - (min_dist_box - TOUCH_THRESHOLD) / (REACH_THRESHOLD - TOUCH_THRESHOLD)
                reach_progress = max(0.0, min(1.0, reach_progress))
                correct_score = 0.2 + reach_progress * 0.2
            else:
                if min_dist_box < 0.3:
                    approach_progress = 1.0 - (min_dist_box - REACH_THRESHOLD) / (0.3 - REACH_THRESHOLD)
                    approach_progress = max(0.0, min(1.0, approach_progress))
                    correct_score = approach_progress * 0.2

        bad_score = 0.0
        is_reaching_obstacles = min_dist_obstacles <= REACH_THRESHOLD
        is_touching_obstacles = min_dist_obstacles <= TOUCH_THRESHOLD

        if is_touching_obstacles:
            bad_score = 0.4
        elif is_reaching_obstacles:
            reach_progress = 1.0 - (min_dist_obstacles - TOUCH_THRESHOLD) / (REACH_THRESHOLD - TOUCH_THRESHOLD)
            reach_progress = max(0.0, min(1.0, reach_progress))
            bad_score = 0.2 + reach_progress * 0.2

        score = max(correct_score, bad_score)
        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"score": score_t}
