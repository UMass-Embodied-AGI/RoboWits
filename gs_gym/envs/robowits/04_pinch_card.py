"""PinchCard environment for RoboWits.

Task: Pick up a bank card from a small table surface using an eraser.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv
from gs_gym.envs.robowits.utils import check_being_held


@register_task("robowits/04-pinch-card-v0")
class PinchCardEnv(RoboWitsEnv):
    """Pick up a bank card from a small table surface using an eraser.

    The bank card is too thin and slippery for a standard gripper to push
    directly on a flat surface. The robot must utilize the friction and
    deformability of the eraser to create enough lateral force to move the
    card to a position where it can be grasped.

    Success criteria:
    - Bank card's 2D footprint overlaps target area by at least 50%
    - Bank card is resting on the target surface (not floating)
    - Bank card is not being held by either gripper
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Pick the bank card up from the 'small table' surface."

    TABLE_Z = 0.76

    # ``small table`` is a support surface (bank_card / eraser rest on it) and
    # ``target cube`` / ``target area`` are landmarks used by the success
    # check; none of them should act as blocking hulls during random
    # placement. Declared here so all ``04_xx`` mutations inherit the rule.
    placement_ignored_hulls = frozenset({"small table", "target cube", "target area"})

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
        table_area = {"x_min": 0.3, "x_max": 0.6, "y_min": -0.15, "y_max": 0.15}
        self._object_reachable_areas["bank card"] = table_area
        self._object_reachable_areas["eraser"] = table_area

    @property
    def placement_groups(self) -> PlacementGroups:
        """Bank card and eraser placed independently."""
        return ("bank card", "eraser")

    def _add_custom_entities(self) -> None:
        """Add bank card, eraser, small table, and target area."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Bank card (mesh)
        bank_card = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/9b4245ca-361f-4940-a23d-090b1e547a52/obj.glb", pattern_is_dir=False),
                scale=0.8,
                pos=(0.46, 0.0, 0.843),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.4),
            surface=gs.surfaces.Smooth(double_sided=True, color=(0.036, 0.177, 0.8)),
        )
        self._entities["bank card"] = {
            "entity": bank_card,
        }

        # Eraser (mesh)
        eraser = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/333832bb-68f0-4f72-a300-658c4fdccfdf/obj.glb", pattern_is_dir=False),
                scale=2,
                pos=(0.48, 0.08, 0.85),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1.2),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["eraser"] = {
            "entity": eraser,
        }

        # Small table (mesh) - fixed
        small_table = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/17674e05-b713-4b2a-a04c-49aee4d4d401/obj.glb", pattern_is_dir=False),
                scale=0.5,
                pos=(0.46, 0, 0.76338),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1.2),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["small table"] = {
            "entity": small_table,
        }

        # Target cube (fixed platform for placing card)
        target_cube = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.7, -0.3, 0.86),
                euler=(0.0, 0.0, 0.0),
                size=(0.15, 0.15, 0.2),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.4, ior=1.5),
        )
        self._entities["target cube"] = {
            "entity": target_cube,
        }

        # Target area (on top of target cube)
        target_area = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.7, -0.3, 0.96),
                euler=(0.0, 0.0, 0.0),
                size=(0.15, 0.15, 0.002),
                fixed=True,
                collision=False,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.2, 0.8, 0.2), roughness=0.4, ior=1.5),
        )
        self._entities["target area"] = {
            "entity": target_area,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: card overlaps target by 50%, resting on it, and not grasped."""

        objs_info = self.collect_objs_info()

        card_name = "bank card"
        target_name = "target area"
        if card_name not in objs_info or target_name not in objs_info:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        card = objs_info[card_name]
        target = objs_info[target_name]

        # Card must be resting on (or very near) the target surface, not floating above it
        TARGET_TOP_Z = 0.961  # top of target cube (pos z=0.96, half-size 0.001)
        RESTING_Z_TOL = 0.05
        card_z = float(card["pos"][2])
        if card_z > TARGET_TOP_Z + RESTING_Z_TOL:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        # Card must not be held by either gripper
        if check_being_held(card, self._robot):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

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

        card_poly = get_poly(card)
        target_poly = get_poly(target)
        if card_poly is None or target_poly is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        def poly_area(poly):
            if poly is None or len(poly) < 3:
                return 0.0
            area = 0.0
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                area += x1 * y2 - x2 * y1
            return abs(area) * 0.5

        def signed_area(poly):
            s = 0.0
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
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

        inter_poly = clip_polygon(card_poly, target_poly)
        card_area = poly_area(card_poly)
        inter_area = poly_area(inter_poly)

        if card_area <= 0.0:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        overlap_ratio = inter_area / card_area
        return torch.tensor([overlap_ratio >= 0.5 - 1e-6], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute dense reward based on milestone-based progress score."""
        objs_info = self.collect_objs_info()

        REACH_THRESHOLD = 0.10
        TOUCH_THRESHOLD = 0.03
        HEIGHT_THRESHOLD = 0.03
        CARD_OUT_AREA_THRESHOLD = 0.10
        INITIAL_CARD_Z = 0.843
        CARD = "bank card"
        SMALL_TABLE = "small table"
        TARGET = "target area"

        for n in [CARD, SMALL_TABLE, TARGET]:
            if n not in objs_info:
                score_t = torch.tensor(0.0, dtype=torch.float32, device=self._device)
                return score_t, {"score": score_t}

        def get_pos(name):
            p = objs_info[name]["pos"]
            return np.array([p[0], p[1], p[2]], dtype=float)

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

        def signed_area(poly):
            if poly is None or len(poly) < 3:
                return 0.0
            x = poly[:, 0]
            y = poly[:, 1]
            return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

        def ensure_ccw(poly):
            if poly is None or len(poly) < 3:
                return poly
            return poly if signed_area(poly) > 0 else poly[::-1].copy()

        def line_intersection(p1, p2, q1, q2):
            p1 = np.asarray(p1, dtype=float)
            p2 = np.asarray(p2, dtype=float)
            q1 = np.asarray(q1, dtype=float)
            q2 = np.asarray(q2, dtype=float)
            r = p2 - p1
            s = q2 - q1
            rxs = r[0] * s[1] - r[1] * s[0]
            q_p = q1 - p1
            if abs(rxs) < 1e-12:
                return (p2 + q2) / 2.0
            t = (q_p[0] * s[1] - q_p[1] * s[0]) / rxs
            return p1 + t * r

        def suth_hodg_intersection(subject, clip):
            if subject is None or clip is None or len(subject) < 3 or len(clip) < 3:
                return np.zeros((0, 2), dtype=float)
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

        def compute_overlap_ratio(hull1, hull2):
            if hull1 is None or hull2 is None:
                return 0.0
            area1 = poly_area(hull1)
            if area1 < 1e-8:
                return 0.0
            inter = suth_hodg_intersection(ensure_ccw(hull1), ensure_ccw(hull2))
            inter_area = poly_area(inter) if len(inter) >= 3 else 0.0
            return inter_area / area1

        right_ee_pos = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee_pos = self._robot.left_ee_pose[0, :3].cpu().numpy()

        card_pos = get_pos(CARD)
        card_hull = get_hull_2d(CARD)
        table_hull = get_hull_2d(SMALL_TABLE)
        target_hull = get_hull_2d(TARGET)

        dist_right = float(np.linalg.norm(right_ee_pos - card_pos))
        dist_left = float(np.linalg.norm(left_ee_pos - card_pos))
        min_dist_to_card = min(dist_right, dist_left)

        is_reaching = min_dist_to_card <= REACH_THRESHOLD
        is_touching = min_dist_to_card <= TOUCH_THRESHOLD
        card_z = float(card_pos[2])
        card_lifted = card_z > (INITIAL_CARD_Z + HEIGHT_THRESHOLD)

        card_inside_table_ratio = compute_overlap_ratio(card_hull, table_hull)
        card_outside_ratio = 1.0 - card_inside_table_ratio
        card_out_of_table = card_outside_ratio > CARD_OUT_AREA_THRESHOLD

        card_on_target_ratio = compute_overlap_ratio(card_hull, target_hull)
        card_on_target = card_on_target_ratio > 0.5

        TARGET_TOP_Z = 0.961  # top of target cube
        RESTING_Z_TOL = 0.05
        card_resting = card_z <= TARGET_TOP_Z + RESTING_Z_TOL
        card_not_held = not check_being_held(objs_info[CARD], self._robot)

        if card_on_target and card_resting and card_not_held:
            score = 1.0
        elif card_on_target:
            # Overlapping target but still held or floating — encourage releasing
            score = 0.95
        elif card_lifted:
            base_score = 0.8
            target_bonus = card_on_target_ratio * 0.15
            score = min(0.94, base_score + target_bonus)
        else:
            interaction_score = 0.0
            if is_touching:
                interaction_score = 0.4
            elif is_reaching:
                reach_progress = 1.0 - (min_dist_to_card - TOUCH_THRESHOLD) / (REACH_THRESHOLD - TOUCH_THRESHOLD)
                reach_progress = max(0.0, min(1.0, reach_progress))
                interaction_score = 0.2 + reach_progress * 0.2
            else:
                if min_dist_to_card < 0.3:
                    approach_progress = 1.0 - (min_dist_to_card - REACH_THRESHOLD) / (0.3 - REACH_THRESHOLD)
                    approach_progress = max(0.0, min(1.0, approach_progress))
                    interaction_score = approach_progress * 0.2

            card_out_score = 0.0
            if card_out_of_table:
                card_out_score = 0.6
            elif card_outside_ratio > 0:
                card_out_score = (card_outside_ratio / CARD_OUT_AREA_THRESHOLD) * 0.6

            score = min(0.8, interaction_score + card_out_score)

        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"score": score_t}
