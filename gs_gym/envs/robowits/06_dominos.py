"""Dominos environment for RoboWits.

Task: Push the red block onto the green target area using a domino chain effect.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv


@register_task("robowits/06-dominos-v0")
class DominosEnv(RoboWitsEnv):
    """Push the red block onto the target area using a domino chain effect.

    The red block and white blocks 1,2,3 are out of direct reach, but form a
    domino structure. The robot must use white block 4 (within reach) to trigger
    a chain reaction that pushes the red block onto the target area.

    Success criteria:
    - Red block's 2D footprint overlaps target area by at least 50%
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Push the red block onto the green target area."

    TABLE_Z = 0.76
    BLOCK_SIZE = (0.02, 0.05, 0.09)

    def __init__(
        self,
        config_name: str = "robowits_default",
        n_envs: int = 1,
        show_viewer: bool = False,
        max_episode_steps: int = 200,
        control_mode: str = "EE_ABS",
        **kwargs,
    ):
        # Initialize per-object reachable areas before super().__init__
        self._object_reachable_areas: dict[str, dict[str, float]] = {}

        super().__init__(
            config_name=config_name,
            n_envs=n_envs,
            show_viewer=show_viewer,
            max_episode_steps=max_episode_steps,
            control_mode=control_mode,
            **kwargs,
        )

        # Set unreachable areas for domino blocks (after init)
        unreachable_area = {
            "x_min": 0.7,
            "x_max": 0.78,
            "y_min": -0.40,
            "y_max": 0.40,
        }
        self._object_reachable_areas["white block 3"] = unreachable_area
        self._object_reachable_areas["white block 2"] = unreachable_area
        self._object_reachable_areas["white block 1"] = unreachable_area
        self._object_reachable_areas["red block"] = unreachable_area
        self._object_reachable_areas["target area"] = unreachable_area

    @property
    def placement_groups(self) -> PlacementGroups:
        """White block 4 independent, others grouped together."""
        return ("white block 4", ("white block 3", "white block 2", "white block 1", "red block", "target area"))

    def _add_custom_entities(self) -> None:
        """Add domino blocks and target area."""
        # Target area
        target = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.96, 0.0, 0.761),
                euler=(0.0, 0.0, 0.0),
                size=(0.06, 0.10, 0.002),
                fixed=True,
                collision=False,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(color=(0.1, 0.7, 0.2), double_sided=False),
        )
        self._entities["target area"] = {
            "entity": target,
        }

        # White block 4 (reachable)
        wb4 = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.50, 0.0, 0.805),
                euler=(0.0, 0.0, 0.0),
                size=self.BLOCK_SIZE,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=300.0),
            surface=gs.surfaces.Smooth(color=(0.95, 0.95, 0.95), double_sided=False),
        )
        self._entities["white block 4"] = {
            "entity": wb4,
        }

        # White block 3
        wb3 = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.735, 0.0, 0.805),
                euler=(0.0, 0.0, 0.0),
                size=self.BLOCK_SIZE,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=300.0),
            surface=gs.surfaces.Smooth(color=(0.95, 0.95, 0.95), double_sided=False),
        )
        self._entities["white block 3"] = {
            "entity": wb3,
        }

        # White block 2
        wb2 = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.795, 0.0, 0.805),
                euler=(0.0, 0.0, 0.0),
                size=self.BLOCK_SIZE,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=300.0),
            surface=gs.surfaces.Smooth(color=(0.95, 0.95, 0.95), double_sided=False),
        )
        self._entities["white block 2"] = {
            "entity": wb2,
        }

        # White block 1
        wb1 = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.855, 0.0, 0.805),
                euler=(0.0, 0.0, 0.0),
                size=self.BLOCK_SIZE,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=300.0),
            surface=gs.surfaces.Smooth(color=(0.95, 0.95, 0.95), double_sided=False),
        )
        self._entities["white block 1"] = {
            "entity": wb1,
        }

        # Red block
        red = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.915, 0.0, 0.805),
                euler=(0.0, 0.0, 0.0),
                size=self.BLOCK_SIZE,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=300.0),
            surface=gs.surfaces.Smooth(color=(0.9, 0.1, 0.1), double_sided=False),
        )
        self._entities["red block"] = {
            "entity": red,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: red block overlaps target by 30%."""
        objs_info = self.collect_objs_info()

        red_name = "red block"
        target_name = "target area"
        if red_name not in objs_info or target_name not in objs_info:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        red = objs_info[red_name]
        target = objs_info[target_name]

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

        red_poly = get_poly(red)
        target_poly = get_poly(target)
        if red_poly is None or target_poly is None:
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

        inter_poly = clip_polygon(red_poly, target_poly)
        red_area = poly_area(red_poly)
        inter_area = poly_area(inter_poly)

        if red_area <= 0.0:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        overlap_ratio = inter_area / red_area
        return torch.tensor([overlap_ratio >= 0.5 - 1e-6], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute dense reward based on milestone-based progress score."""
        objs_info = self.collect_objs_info()

        REACH_THRESHOLD = 0.12
        TOUCH_THRESHOLD = 0.05
        GRASP_GRIPPER_WIDTH = 0.04
        BLOCK_TOUCH_THRESHOLD = 0.03

        WHITE_BLOCK_4 = "white block 4"
        WHITE_BLOCK_3 = "white block 3"
        WHITE_BLOCK_2 = "white block 2"
        RED_BLOCK = "red block"
        TARGET = "target area"

        required_objects = [WHITE_BLOCK_4, WHITE_BLOCK_3, WHITE_BLOCK_2, RED_BLOCK, TARGET]
        for n in required_objects:
            if n not in objs_info:
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

        def blocks_touching(name1, name2):
            b1 = get_bounds(name1)
            b2 = get_bounds(name2)
            if b1 is None or b2 is None:
                return False
            gap_x = max(0, max(b1[0, 0], b2[0, 0]) - min(b1[1, 0], b2[1, 0]))
            gap_y = max(0, max(b1[0, 1], b2[0, 1]) - min(b1[1, 1], b2[1, 1]))
            gap_z = max(0, max(b1[0, 2], b2[0, 2]) - min(b1[1, 2], b2[1, 2]))
            total_gap = np.sqrt(gap_x**2 + gap_y**2 + gap_z**2)
            return total_gap <= BLOCK_TOUCH_THRESHOLD

        def red_on_target():
            def get_poly(name):
                obj = objs_info.get(name)
                if obj is None:
                    return None
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

            red_poly = get_poly(RED_BLOCK)
            target_poly = get_poly(TARGET)
            if red_poly is None or target_poly is None:
                return 0.0
            red_area = poly_area(red_poly)
            if red_area <= 0.0:
                return 0.0
            inter_poly = clip_polygon(red_poly, target_poly)
            inter_area = poly_area(inter_poly)
            return inter_area / red_area

        right_ee_pos = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee_pos = self._robot.left_ee_pose[0, :3].cpu().numpy()
        _dof_pos = self._robot.robot_entity.get_dofs_position()
        right_gripper_width = float(_dof_pos[0, self._robot.right_gripper_dofs[0]]) * 2
        left_gripper_width = float(_dof_pos[0, self._robot.left_gripper_dofs[0]]) * 2

        red_target_overlap = red_on_target()
        if red_target_overlap >= 0.5:
            score_t = torch.tensor(1.0, dtype=torch.float32, device=self._device)
            return score_t, {"score": score_t}

        wb4_pos = get_pos(WHITE_BLOCK_4)
        wb3_pos = get_pos(WHITE_BLOCK_3)

        dist_right_wb4 = float(np.linalg.norm(right_ee_pos - wb4_pos))
        dist_left_wb4 = float(np.linalg.norm(left_ee_pos - wb4_pos))
        min_dist_wb4 = min(dist_right_wb4, dist_left_wb4)
        closest_gripper_wb4 = right_gripper_width if dist_right_wb4 < dist_left_wb4 else left_gripper_width

        dist_right_wb3 = float(np.linalg.norm(right_ee_pos - wb3_pos))
        dist_left_wb3 = float(np.linalg.norm(left_ee_pos - wb3_pos))
        min_dist_wb3 = min(dist_right_wb3, dist_left_wb3)

        wb4_touches_wb3 = blocks_touching(WHITE_BLOCK_4, WHITE_BLOCK_3)
        wb3_touches_wb2 = blocks_touching(WHITE_BLOCK_3, WHITE_BLOCK_2)

        line1_score = 0.0
        if wb4_touches_wb3:
            line1_score = 0.8
            line1_score += red_target_overlap * 0.19
        else:
            is_reaching_wb4 = min_dist_wb4 <= REACH_THRESHOLD
            is_touching_wb4 = min_dist_wb4 <= TOUCH_THRESHOLD
            is_grasping_wb4 = is_touching_wb4 and closest_gripper_wb4 < GRASP_GRIPPER_WIDTH

            if is_grasping_wb4:
                line1_score = 0.6
            elif is_touching_wb4:
                grip_progress = max(0.0, 1.0 - closest_gripper_wb4 / 0.095)
                line1_score = 0.4 + grip_progress * 0.2
            elif is_reaching_wb4:
                reach_progress = 1.0 - (min_dist_wb4 - TOUCH_THRESHOLD) / (REACH_THRESHOLD - TOUCH_THRESHOLD)
                reach_progress = max(0.0, min(1.0, reach_progress))
                line1_score = 0.2 + reach_progress * 0.2
            else:
                if min_dist_wb4 < 0.3:
                    approach_progress = 1.0 - (min_dist_wb4 - REACH_THRESHOLD) / (0.3 - REACH_THRESHOLD)
                    approach_progress = max(0.0, min(1.0, approach_progress))
                    line1_score = approach_progress * 0.2

        line2_score = 0.0
        if wb3_touches_wb2:
            line2_score = 0.8
            line2_score += red_target_overlap * 0.19
        else:
            is_reaching_wb3 = min_dist_wb3 <= REACH_THRESHOLD
            is_touching_wb3 = min_dist_wb3 <= TOUCH_THRESHOLD

            if is_touching_wb3:
                line2_score = 0.6
            elif is_reaching_wb3:
                reach_progress = 1.0 - (min_dist_wb3 - TOUCH_THRESHOLD) / (REACH_THRESHOLD - TOUCH_THRESHOLD)
                reach_progress = max(0.0, min(1.0, reach_progress))
                line2_score = 0.3 + reach_progress * 0.3
            else:
                if min_dist_wb3 < 0.3:
                    approach_progress = 1.0 - (min_dist_wb3 - REACH_THRESHOLD) / (0.3 - REACH_THRESHOLD)
                    approach_progress = max(0.0, min(1.0, approach_progress))
                    line2_score = approach_progress * 0.3

        score = max(line1_score, line2_score)
        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"score": score_t}
