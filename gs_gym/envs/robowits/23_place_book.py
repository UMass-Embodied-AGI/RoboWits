"""PlaceBook environment for RoboWits.

Task: Move the thin book onto the blue target mat without disturbing the heavy block.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits import utils as _utils
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv


@register_task("robowits/23-place-book-v0")
class PlaceBookEnv(RoboWitsEnv):
    """Move the thin book onto the blue target mat without disturbing the heavy block.

    This is a practical tabletop retrieval scenario: a book is pinned under a
    heavy object and must be pulled out without disturbing the weight. The pry
    board acts as a shim to get under the book edge.

    Success criteria:
    - Thin book is on the blue target mat (95% overlap)
    - Heavy block remains in place (displacement <= 10mm)
    - Book is free of heavy block (no overlap)
    - Objects remain within table bounds
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Move the 'thin book' onto the 'blue target mat' without moving the heavy block from its place."

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
        """No randomization for this task."""
        return None

    def _add_custom_entities(self) -> None:
        """Add thin book, heavy block, blue target mat, and pry board."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Thin book (mesh)
        book = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/decafac9-3ae6-4a22-983d-34f365f95642/obj.glb", pattern_is_dir=False),
                scale=(1.720, 1.111, 0.870),
                pos=(0.41, 0.0, 0.766),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=700.0, friction=0.2),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["thin book"] = {
            "entity": book,
        }

        # Heavy block (box)
        heavy_block = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.46, 0.0, 0.802),
                euler=(0.0, 0.0, 0.0),
                size=(0.20, 0.16, 0.06),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=4500.0, friction=1.0),
            surface=gs.surfaces.Smooth(color=(0.25, 0.25, 0.28), double_sided=True),
        )
        self._entities["heavy block"] = {
            "entity": heavy_block,
        }

        # Blue target mat (box)
        target_mat = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.66, 0.31, 0.7615),
                euler=(0.0, 0.0, 0.0),
                size=(0.12, 0.20, 0.003),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=800.0, friction=1.2),
            surface=gs.surfaces.Default(color=(0.1, 0.3, 0.9), roughness=0.8, ior=1.5),
        )
        self._entities["blue target mat"] = {
            "entity": target_mat,
        }

        # Pry board (mesh)
        pry_board = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/08893dc4-bfb1-49ca-9e47-4b3958a21e4b/obj.glb", pattern_is_dir=False),
                scale=(0.1176, 0.0534, 0.0755),
                pos=(0.36, -0.12, 0.7625),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=800.0, friction=0.5),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["pry board"] = {
            "entity": pry_board,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: book on mat, block undisturbed."""
        objs_info = self.collect_objs_info()

        BOOK_ON_MAT_RATIO = 0.95
        BOOK_HB_NO_OVERLAP_RATIO = 0.02
        TABLE_X_MIN, TABLE_X_MAX = 0.21, 1.00
        TABLE_Y_MIN, TABLE_Y_MAX = -0.69, 0.69

        def get_hull_xy(bounds, hull_2d):
            if hull_2d is not None and len(hull_2d) >= 3:
                return [(float(p[0]), float(p[1])) for p in hull_2d]
            if bounds is None:
                return None
            xmin, ymin = float(bounds[0][0]), float(bounds[0][1])
            xmax, ymax = float(bounds[1][0]), float(bounds[1][1])
            return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

        def is_within_table_bounds(poly):
            if poly is None:
                return False
            for x, y in poly:
                if x < TABLE_X_MIN - 1e-3 or x > TABLE_X_MAX + 1e-3:
                    return False
                if y < TABLE_Y_MIN - 1e-3 or y > TABLE_Y_MAX + 1e-3:
                    return False
            return True

        def polygon_area(poly):
            if poly is None or len(poly) < 3:
                return 0.0
            n = len(poly)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += poly[i][0] * poly[j][1]
                area -= poly[j][0] * poly[i][1]
            return abs(area) / 2.0

        def compute_overlap_ratio(poly_a, poly_b):
            inter_area = _utils.intersection_area(poly_a, poly_b)
            a_area = polygon_area(poly_a)
            if a_area <= 1e-9:
                return 0.0
            return inter_area / a_area

        hb = objs_info.get("heavy block")
        book = objs_info.get("thin book")
        mat = objs_info.get("blue target mat")

        if hb is None or book is None or mat is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        hb_hull = get_hull_xy(hb.get("bounds"), hb.get("convex_hull_2d"))
        book_hull = get_hull_xy(book.get("bounds"), book.get("convex_hull_2d"))
        mat_hull = get_hull_xy(mat.get("bounds"), mat.get("convex_hull_2d"))

        if hb_hull is None or book_hull is None or mat_hull is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        if not is_within_table_bounds(hb_hull):
            return torch.tensor([False], dtype=torch.bool, device=self._device)
        if not is_within_table_bounds(book_hull):
            return torch.tensor([False], dtype=torch.bool, device=self._device)
        if not is_within_table_bounds(mat_hull):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        euler = hb.get("euler")
        if euler is not None:
            roll_deg = abs(float(euler[0])) * 180.0 / np.pi
            pitch_deg = abs(float(euler[1])) * 180.0 / np.pi
            roll_ok = roll_deg <= 15.0
            pitch_ok = pitch_deg <= 15.0
            if not (roll_ok and pitch_ok):
                return torch.tensor([False], dtype=torch.bool, device=self._device)

        book_area = polygon_area(book_hull)
        if book_area <= 1e-5:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        inter_b_hb = compute_overlap_ratio(book_hull, hb_hull) * book_area
        free_of_block = inter_b_hb <= max(1e-5, BOOK_HB_NO_OVERLAP_RATIO * book_area)

        inter_b_m = compute_overlap_ratio(book_hull, mat_hull)
        on_mat = inter_b_m >= BOOK_ON_MAT_RATIO

        return torch.tensor([free_of_block and on_mat], dtype=torch.bool, device=self._device)

    def _polygon_area(self, poly: np.ndarray) -> float:
        """Compute polygon area using shoelace formula."""
        if poly is None or len(poly) < 3:
            return 0.0
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute dense reward based on milestone-based progress score."""
        objs_info = self.collect_objs_info()

        TOL = 1e-3
        REACH_THRESHOLD = 0.15
        TOUCH_THRESHOLD = 0.05
        APPROACH_THRESHOLD = 0.3
        SHELF_BOOKS_EULER_THRESHOLD = 40.0
        BOOK_UPRIGHT_THRESHOLD = 50.0

        BOARD = "board"
        BOOK = "book"
        SHELF_LEFT = "shelf left"
        SHELF_RIGHT = "shelf right"
        SHELF_BACK = "shelf back"
        SHELF_TOP = "shelf top"
        BOOKS_ON_SHELF = ["book on shelf 1", "book on shelf 2", "book on shelf 3", "book on shelf 4", "book on shelf 5"]

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

        def get_euler(name):
            obj = objs_info.get(name)
            if obj is None:
                return None
            e = obj.get("euler")
            if e is None:
                return None
            return np.array([e[0], e[1], e[2]], dtype=float)

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

        def get_shelf_bounds():
            sl_b = get_bounds(SHELF_LEFT)
            sr_b = get_bounds(SHELF_RIGHT)
            sb_b = get_bounds(SHELF_BACK)
            st_b = get_bounds(SHELF_TOP)
            if sl_b is None or sr_b is None or sb_b is None or st_b is None:
                return None
            shelf_x_min = float(sl_b[0, 0]) + TOL
            shelf_x_max = float(sb_b[0, 0]) - TOL
            shelf_y_min = float(sr_b[1, 1]) + TOL
            shelf_y_max = float(sl_b[0, 1]) - TOL
            shelf_z_min = float(sl_b[0, 2])
            shelf_z_max = float(st_b[0, 2]) - TOL
            return np.array(
                [[shelf_x_min, shelf_y_min, shelf_z_min], [shelf_x_max, shelf_y_max, shelf_z_max]], dtype=float
            )

        def bounds_intersect(b1, b2):
            if b1 is None or b2 is None:
                return False
            tol = 0.01
            overlap_x = b1[1, 0] >= b2[0, 0] - tol and b2[1, 0] >= b1[0, 0] - tol
            overlap_y = b1[1, 1] >= b2[0, 1] - tol and b2[1, 1] >= b1[0, 1] - tol
            overlap_z = b1[1, 2] >= b2[0, 2] - tol and b2[1, 2] >= b1[0, 2] - tol
            return overlap_x and overlap_y and overlap_z

        def poly_area(poly):
            if poly is None or len(poly) < 3:
                return 0.0
            x = poly[:, 0]
            y = poly[:, 1]
            return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

        def compute_overlap_ratio_with_rect(hull, rect_min, rect_max):
            if hull is None or len(hull) < 3:
                return 0.0
            rect_poly = np.array(
                [
                    [rect_min[0], rect_min[1]],
                    [rect_max[0], rect_min[1]],
                    [rect_max[0], rect_max[1]],
                    [rect_min[0], rect_max[1]],
                ],
                dtype=float,
            )

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

            area = poly_area(hull)
            if area < 1e-8:
                return 0.0
            inter = suth_hodg(ensure_ccw(hull), ensure_ccw(rect_poly))
            inter_area = poly_area(inter) if len(inter) >= 3 else 0.0
            return inter_area / area

        shelf_bounds = get_shelf_bounds()
        if shelf_bounds is None:
            score_t = torch.tensor(0.0, dtype=torch.float32, device=self._device)
            return score_t, {"score": score_t}

        right_ee_pos = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee_pos = self._robot.left_ee_pose[0, :3].cpu().numpy()

        board_score = 0.0
        if BOARD in objs_info:
            board_pos = get_pos(BOARD)
            if board_pos is not None:
                dist_right_board = float(np.linalg.norm(right_ee_pos - board_pos))
                dist_left_board = float(np.linalg.norm(left_ee_pos - board_pos))
                min_dist_board = min(dist_right_board, dist_left_board)
            else:
                min_dist_board = float("inf")

            board_bounds = get_bounds(BOARD)
            board_hull = get_hull_2d(BOARD)
            board_inside_ratio = compute_overlap_ratio_with_rect(board_hull, shelf_bounds[0, :2], shelf_bounds[1, :2])
            board_intersects_shelf = bounds_intersect(board_bounds, shelf_bounds) if board_bounds is not None else False

            books_upright_count = 0
            for book_name in BOOKS_ON_SHELF:
                if book_name in objs_info:
                    euler = get_euler(book_name)
                    if euler is not None and abs(float(euler[0])) < SHELF_BOOKS_EULER_THRESHOLD:
                        books_upright_count += 1
            books_upright_progress = books_upright_count / len(BOOKS_ON_SHELF)
            progress_to_intersect = (
                min(1.0, board_inside_ratio * 10.0)
                if board_hull is not None
                else (1.0 if board_intersects_shelf else 0.0)
            )

            if min_dist_board <= REACH_THRESHOLD:
                reach_progress = 1.0 - (min_dist_board - TOUCH_THRESHOLD) / (REACH_THRESHOLD - TOUCH_THRESHOLD)
                reach_progress = max(0.0, min(1.0, reach_progress))
                ee_based_board = (
                    0.2 + reach_progress * 0.1 if min_dist_board <= TOUCH_THRESHOLD else 0.1 + reach_progress * 0.1
                )
            elif min_dist_board < APPROACH_THRESHOLD:
                approach_progress = 1.0 - (min_dist_board - REACH_THRESHOLD) / (APPROACH_THRESHOLD - REACH_THRESHOLD)
                approach_progress = max(0.0, min(1.0, approach_progress))
                ee_based_board = approach_progress * 0.1
            else:
                ee_based_board = 0.0

            if books_upright_progress >= 1.0:
                board_state_score = 0.4
            elif board_intersects_shelf:
                board_state_score = 0.3 + books_upright_progress * 0.1
            elif min_dist_board <= TOUCH_THRESHOLD:
                board_state_score = 0.2 + progress_to_intersect * 0.1
            else:
                board_state_score = ee_based_board

            board_score = max(board_state_score, ee_based_board)

        book_score = 0.0
        if BOOK in objs_info:
            book_pos = get_pos(BOOK)
            if book_pos is not None:
                dist_right_book = float(np.linalg.norm(right_ee_pos - book_pos))
                dist_left_book = float(np.linalg.norm(left_ee_pos - book_pos))
                min_dist_book = min(dist_right_book, dist_left_book)
            else:
                min_dist_book = float("inf")

            book_bounds = get_bounds(BOOK)
            book_hull = get_hull_2d(BOOK)
            book_inside_ratio = compute_overlap_ratio_with_rect(book_hull, shelf_bounds[0, :2], shelf_bounds[1, :2])
            book_intersects_shelf = bounds_intersect(book_bounds, shelf_bounds) if book_bounds is not None else False
            book_euler = get_euler(BOOK)
            book_euler_x = abs(float(book_euler[0])) if book_euler is not None else float("inf")
            book_upright_progress = (
                max(0.0, 1.0 - book_euler_x / BOOK_UPRIGHT_THRESHOLD) if book_euler_x <= BOOK_UPRIGHT_THRESHOLD else 0.0
            )

            progress_to_intersect = min(1.0, book_inside_ratio * 10.0)
            if min_dist_book <= REACH_THRESHOLD:
                reach_progress = 1.0 - (min_dist_book - TOUCH_THRESHOLD) / (REACH_THRESHOLD - TOUCH_THRESHOLD)
                reach_progress = max(0.0, min(1.0, reach_progress))
                ee_based = (
                    0.2 + reach_progress * 0.1 if min_dist_book <= TOUCH_THRESHOLD else 0.1 + reach_progress * 0.1
                )
            elif min_dist_book < APPROACH_THRESHOLD:
                approach_progress = 1.0 - (min_dist_book - REACH_THRESHOLD) / (APPROACH_THRESHOLD - REACH_THRESHOLD)
                approach_progress = max(0.0, min(1.0, approach_progress))
                ee_based = approach_progress * 0.2
            else:
                ee_based = 0.0

            if book_inside_ratio > 0.5 and book_upright_progress >= 1.0:
                book_state_score = 0.6
            elif book_inside_ratio > 0.5:
                book_state_score = 0.5 + book_upright_progress * 0.1
            elif book_inside_ratio > 0.0:
                progress_to_50 = min(1.0, book_inside_ratio / 0.5)
                book_state_score = 0.4 + progress_to_50 * 0.1
            elif book_intersects_shelf:
                book_state_score = 0.4
            elif min_dist_book <= TOUCH_THRESHOLD:
                book_state_score = 0.3 + progress_to_intersect * 0.1
            else:
                book_state_score = ee_based

            book_score = max(book_state_score, ee_based)

        total_score = board_score + book_score
        score_t = torch.tensor(total_score, dtype=torch.float32, device=self._device)
        return score_t, {"score": score_t}
