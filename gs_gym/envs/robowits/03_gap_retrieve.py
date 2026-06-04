"""GapRetrieve environment for RoboWits.

Task: Retrieve a lemon from a narrow gap and place it on a target area.
"""

from __future__ import annotations

import genesis as gs
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv
from gs_gym.envs.robowits.utils import check_being_held


@register_task("robowits/03-gap-retrieve-v0")
class GapRetrieveEnv(RoboWitsEnv):
    """Retrieve a lemon from a narrow gap and place it on a target area.

    The lemon is located in a narrow channel formed by two boundary blocks.
    The gap is too narrow for the gripper to enter directly. The robot must
    first move one boundary block to widen the gap, then retrieve the lemon.

    Success criteria:
    - Lemon is on the table surface (z within tolerance of TABLE_Z)
    - Lemon's 2D footprint overlaps target area by at least 80%
    - Objects remain within table bounds
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Place the lemon on the green target area."

    TABLE_Z = 0.76
    LEMON_SIZE = (0.05, 0.03, 0.03)  # Approximate

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
        """Lemon and boundary blocks grouped together, target independent."""
        return (("lemon", "left boundary block", "right boundary block"), "target area")

    def _add_custom_entities(self) -> None:
        """Add lemon, boundary blocks, and target area."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Left boundary block
        left_block = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.40, -0.031 - 0.025, 0.785 + 0.075),
                euler=(0.0, 0.0, 0.0),
                size=(0.14, 0.05, 0.2),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500.0, friction=1.0),
            surface=gs.surfaces.Rough(color=(0.6, 0.6, 0.6), double_sided=False),
        )
        self._entities["left boundary block"] = {
            "entity": left_block,
        }

        # Right boundary block
        right_block = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.40, 0.031 + 0.025, 0.785 + 0.075),
                euler=(0.0, 0.0, 0.0),
                size=(0.14, 0.05, 0.2),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500.0, friction=1.0),
            surface=gs.surfaces.Rough(color=(0.6, 0.6, 0.6), double_sided=False),
        )
        self._entities["right boundary block"] = {
            "entity": right_block,
        }

        # Target area
        target = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.60, 0.00, 0.7605),
                euler=(0.0, 0.0, 0.0),
                size=(0.12, 0.10, 0.001),
                fixed=True,
                collision=False,
            ),
            material=gs.materials.Rigid(rho=100.0, friction=0.5),
            surface=gs.surfaces.Default(
                color=(0.1, 0.8, 0.1), roughness=0.9, ior=1.4, double_sided=True, vis_mode="visual"
            ),
        )
        self._entities["target area"] = {
            "entity": target,
        }

        # Lemon (mesh)
        lemon = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c45b108b-2163-469c-8ed4-dcb82260d83f/obj.glb", pattern_is_dir=False),
                scale=0.6,
                pos=(0.40, 0.0, 0.76064 + 0.019),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=1200.0, friction=0.8),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["lemon"] = {
            "entity": lemon,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: lemon on target with 80% overlap."""
        objs_info = self.collect_objs_info()

        TABLE_X_MIN, TABLE_X_MAX = 0.21, 1.00
        TABLE_Y_MIN, TABLE_Y_MAX = -0.69, 0.69
        TABLE_Z = 0.76
        Z_TOL = 0.02

        key_name = "lemon"
        target_name = "target area"

        if key_name not in objs_info or target_name not in objs_info:
            return torch.tensor([False], dtype=torch.bool, device=self._device)
        key = objs_info[key_name]
        target = objs_info[target_name]

        def get_hull2d(obj):
            hull = obj.get("convex_hull_2d", None)
            if hull is not None and len(hull) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in hull]
                return pts
            b = obj.get("bounds", None)
            if b is None:
                return None
            (xmin, ymin, _), (xmax, ymax, _) = b
            return [
                (float(xmin), float(ymin)),
                (float(xmax), float(ymin)),
                (float(xmax), float(ymax)),
                (float(xmin), float(ymax)),
            ]

        def polygon_area(poly):
            if not poly or len(poly) < 3:
                return 0.0
            area = 0.0
            for i in range(len(poly)):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % len(poly)]
                area += x1 * y2 - x2 * y1
            return abs(area) * 0.5

        def ensure_ccw(poly):
            signed = 0.0
            for i in range(len(poly)):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % len(poly)]
                signed += x1 * y2 - x2 * y1
            if signed < 0:
                return list(reversed(poly))
            return poly

        def point_in_table_xy(p):
            x, y = p
            return (TABLE_X_MIN - 1e-4) <= x <= (TABLE_X_MAX + 1e-4) and (TABLE_Y_MIN - 1e-4) <= y <= (
                TABLE_Y_MAX + 1e-4
            )

        def hull_inside_table(poly):
            if not poly:
                return False
            return all(point_in_table_xy(p) for p in poly)

        def cross(ax, ay, bx, by):
            return ax * by - ay * bx

        def line_intersection(p, r, q, svec):
            rxs = cross(r[0], r[1], svec[0], svec[1])
            if abs(rxs) < 1e-12:
                return None
            qp = (q[0] - p[0], q[1] - p[1])
            t = cross(qp[0], qp[1], svec[0], svec[1]) / rxs
            return (p[0] + t * r[0], p[1] + t * r[1])

        def is_inside(p, a, b):
            return cross(b[0] - a[0], b[1] - a[1], p[0] - a[0], p[1] - a[1]) >= -1e-9

        def sutherland_hodgman(subject, clip):
            if not subject or not clip:
                return []
            output = subject[:]
            for i in range(len(clip)):
                cp1 = clip[i]
                cp2 = clip[(i + 1) % len(clip)]
                input_list = output
                output = []
                if not input_list:
                    break
                s = input_list[-1]
                for e in input_list:
                    if is_inside(e, cp1, cp2):
                        if not is_inside(s, cp1, cp2):
                            inter = line_intersection(
                                s, (e[0] - s[0], e[1] - s[1]), cp1, (cp2[0] - cp1[0], cp2[1] - cp1[1])
                            )
                            if inter is not None:
                                output.append(inter)
                        output.append(e)
                    elif is_inside(s, cp1, cp2):
                        inter = line_intersection(
                            s, (e[0] - s[0], e[1] - s[1]), cp1, (cp2[0] - cp1[0], cp2[1] - cp1[1])
                        )
                        if inter is not None:
                            output.append(inter)
                    s = e
            return output

        key_hull = get_hull2d(key)
        target_hull = get_hull2d(target)
        if key_hull is None or target_hull is None or len(key_hull) < 3 or len(target_hull) < 3:
            return torch.tensor([False], dtype=torch.bool, device=self._device)
        key_hull = ensure_ccw(key_hull)
        target_hull = ensure_ccw(target_hull)

        key_bounds = key.get("bounds", None)
        if key_bounds is None:
            kpos = key.get("pos", None)
            if kpos is None:
                return torch.tensor([False], dtype=torch.bool, device=self._device)
            on_table = (kpos[2] - TABLE_Z) <= 0.03
        else:
            z_min = float(key_bounds[0][2])
            on_table = abs(z_min - TABLE_Z) <= Z_TOL

        key_on_table_xy = hull_inside_table(key_hull)
        target_on_table_xy = hull_inside_table(target_hull)
        if not (key_on_table_xy and target_on_table_xy):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        key_area = polygon_area(key_hull)
        if key_area <= 1e-8:
            return torch.tensor([False], dtype=torch.bool, device=self._device)
        inter_poly = sutherland_hodgman(key_hull, target_hull)
        overlap_area = polygon_area(inter_poly) if len(inter_poly) >= 3 else 0.0
        overlap_fraction = overlap_area / key_area

        lemon_not_held = not check_being_held(key, self._robot)

        return torch.tensor(
            [on_table and overlap_fraction >= 0.8 and lemon_not_held],
            dtype=torch.bool,
            device=self._device,
        )

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute dense reward based on milestone-based progress score."""
        import numpy as np

        objs_info = self.collect_objs_info()

        REACH_THRESHOLD = 0.12
        TOUCH_THRESHOLD = 0.04
        BLOCK_MOVE_XY = 0.02
        BLOCK_Z_DELTA = 0.01
        BLOCK_HALF_Y = 0.025
        INITIAL_GAP = 0.062
        GAP_SUFFICIENT = 0.08
        LEMON_ON_TABLE_Z = 0.82

        LEFT_BLOCK = "left boundary block"
        RIGHT_BLOCK = "right boundary block"
        LEMON = "lemon"
        TARGET = "target area"

        INIT_LEFT_POS = np.array([0.40, -0.056, 0.86], dtype=float)
        INIT_RIGHT_POS = np.array([0.40, 0.056, 0.86], dtype=float)
        INIT_LEMON_POS = np.array([0.40, 0.0, 0.77964], dtype=float)
        TARGET_CENTER = np.array([0.60, 0.0], dtype=float)

        for name in [LEFT_BLOCK, RIGHT_BLOCK, LEMON, TARGET]:
            if name not in objs_info:
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
                return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-9

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

        left_pos = get_pos(LEFT_BLOCK)
        right_pos = get_pos(RIGHT_BLOCK)
        lemon_pos = get_pos(LEMON)
        lemon_xy = lemon_pos[:2]

        gap = (right_pos[1] - BLOCK_HALF_Y) - (left_pos[1] + BLOCK_HALF_Y)
        left_moved = float(np.linalg.norm(left_pos[:2] - INIT_LEFT_POS[:2])) > BLOCK_MOVE_XY
        right_moved = float(np.linalg.norm(right_pos[:2] - INIT_RIGHT_POS[:2])) > BLOCK_MOVE_XY
        left_pushed_down = left_pos[2] < (INIT_LEFT_POS[2] - BLOCK_Z_DELTA)
        right_pushed_down = right_pos[2] < (INIT_RIGHT_POS[2] - BLOCK_Z_DELTA)
        block_moved = left_moved or right_moved
        block_pushed = left_pushed_down or right_pushed_down
        gap_widened = gap > INITIAL_GAP
        gap_sufficient = gap >= GAP_SUFFICIENT

        # Lemon is out of channel when it has escaped the block walls geometrically
        BLOCK_HALF_X = 0.07  # half of block x-size (0.14)
        BLOCK_UPRIGHT_Z_MIN = INIT_LEFT_POS[2] - 0.15  # block fallen if z dropped > 15cm
        left_upright = left_pos[2] > BLOCK_UPRIGHT_Z_MIN
        right_upright = right_pos[2] > BLOCK_UPRIGHT_Z_MIN
        # Y-based check is unreliable when a block has fallen (its center Y is meaningless)
        left_inner_y = left_pos[1] + BLOCK_HALF_Y if left_upright else -float("inf")
        right_inner_y = right_pos[1] - BLOCK_HALF_Y if right_upright else float("inf")
        lemon_escaped_x = abs(lemon_pos[0] - INIT_LEMON_POS[0]) > BLOCK_HALF_X
        lemon_escaped_y = lemon_pos[1] < left_inner_y - 0.01 or lemon_pos[1] > right_inner_y + 0.01
        lemon_out_of_channel = lemon_escaped_x or lemon_escaped_y

        lemon_on_table = lemon_pos[2] <= LEMON_ON_TABLE_Z

        lemon_to_target = float(np.linalg.norm(lemon_xy - TARGET_CENTER))
        init_lemon_to_target = float(np.linalg.norm(INIT_LEMON_POS[:2] - TARGET_CENTER))

        dist_ee_block = min(
            float(np.linalg.norm(right_ee_pos - left_pos)),
            float(np.linalg.norm(right_ee_pos - right_pos)),
            float(np.linalg.norm(left_ee_pos - left_pos)),
            float(np.linalg.norm(left_ee_pos - right_pos)),
        )
        dist_ee_lemon = min(
            float(np.linalg.norm(right_ee_pos - lemon_pos)),
            float(np.linalg.norm(left_ee_pos - lemon_pos)),
        )

        def lerp(lo, hi, t):
            return lo + (hi - lo) * max(0.0, min(1.0, t))

        # Block manipulation score (0.0–0.50)
        if gap_sufficient:
            if dist_ee_lemon <= TOUCH_THRESHOLD:
                block_score = 0.50
            elif dist_ee_lemon <= REACH_THRESHOLD:
                t = 1.0 - (dist_ee_lemon - TOUCH_THRESHOLD) / (REACH_THRESHOLD - TOUCH_THRESHOLD)
                block_score = lerp(0.42, 0.50, max(0.0, t))
            else:
                block_score = 0.42
        elif gap_widened or block_pushed:
            gap_progress = min(1.0, (gap - INITIAL_GAP) / (GAP_SUFFICIENT - INITIAL_GAP))
            block_score = lerp(0.25, 0.42, gap_progress)
        elif block_moved or block_pushed:
            move_ratio = 0.0
            if left_moved:
                move_ratio = max(move_ratio, min(1.0, float(np.linalg.norm(left_pos[:2] - INIT_LEFT_POS[:2])) / 0.04))
            if right_moved:
                move_ratio = max(move_ratio, min(1.0, float(np.linalg.norm(right_pos[:2] - INIT_RIGHT_POS[:2])) / 0.04))
            block_score = lerp(0.12, 0.25, move_ratio)
        elif dist_ee_block <= TOUCH_THRESHOLD:
            move_ratio = max(
                min(1.0, float(np.linalg.norm(left_pos[:2] - INIT_LEFT_POS[:2])) / BLOCK_MOVE_XY)
                if BLOCK_MOVE_XY > 0
                else 0.0,
                min(1.0, float(np.linalg.norm(right_pos[:2] - INIT_RIGHT_POS[:2])) / BLOCK_MOVE_XY)
                if BLOCK_MOVE_XY > 0
                else 0.0,
            )
            block_score = lerp(0.06, 0.12, move_ratio)
        elif dist_ee_block <= REACH_THRESHOLD:
            t = 1.0 - (dist_ee_block - TOUCH_THRESHOLD) / (REACH_THRESHOLD - TOUCH_THRESHOLD)
            block_score = lerp(0.0, 0.06, max(0.0, t))
        else:
            block_score = 0.0

        # Extraction score (0.0–0.50): rises as lemon moves out of channel
        lemon_displacement = float(np.linalg.norm(lemon_pos[:2] - INIT_LEMON_POS[:2]))
        extraction_score = lerp(0.0, 0.50, min(1.0, lemon_displacement / 0.10))

        # Transport score (0.50–1.0): proportional to distance to target, only when out and on table
        if lemon_out_of_channel and lemon_on_table:
            dist_progress = max(0.0, 1.0 - lemon_to_target / (init_lemon_to_target + 1e-8))
            transport_score = lerp(0.50, 1.0, dist_progress)
        else:
            transport_score = 0.0

        score = max(block_score, extraction_score, transport_score)

        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"score": score_t}
