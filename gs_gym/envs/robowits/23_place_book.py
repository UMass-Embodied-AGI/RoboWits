"""PlaceBook environment for RoboWits.

Task: Place the book onto the shelf without disturbing the books already on it.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv

BOOK_MESH = "blender_kit/ee74f10e-acc7-4f5a-80e2-1b2999c6743e/obj.glb"
BOOK_SCALE = (0.4, 0.4, 0.4)
SHELF_COLOR = (27 / 255, 103 / 255, 235 / 255)


@register_task("robowits/23-place-book-v0")
class PlaceBookEnv(RoboWitsEnv):
    """Place the book onto the shelf.

    A book must be slid onto a shelf while keeping the books already on the
    shelf upright. A board can be used as a shim under the book edge.
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Shelve the book upright without knocking over the other books."

    def __init__(
        self,
        config_name: str = "robowits_default",
        n_envs: int = 1,
        show_viewer: bool = False,
        max_episode_steps: int = 200,
        control_mode: str = "EE_ABS",
        **kwargs,
    ):
        self._object_reachable_areas: dict[str, dict[str, float]] = {}

        super().__init__(
            config_name=config_name,
            n_envs=n_envs,
            show_viewer=show_viewer,
            max_episode_steps=max_episode_steps,
            control_mode=control_mode,
            **kwargs,
        )

        shelf_area = {"x_min": 0.45, "x_max": 0.75, "y_min": -0.2, "y_max": 0.2}
        close_area = {"x_min": 0.1, "x_max": 0.55, "y_min": -0.35, "y_max": 0.35}
        for name in (
            "shelf_left",
            "shelf_right",
            "shelf_back",
            "shelf_top",
            "book_on_shelf_1",
            "book_on_shelf_2",
            "book_on_shelf_3",
            "book_on_shelf_4",
            "book_on_shelf_5",
        ):
            self._object_reachable_areas[name] = shelf_area
        self._object_reachable_areas["book"] = close_area
        self._object_reachable_areas["board"] = close_area

    @property
    def placement_groups(self) -> PlacementGroups:
        """Define placement groups for random initialization."""
        return (
            (
                "shelf_left",
                "shelf_right",
                "shelf_back",
                "shelf_top",
                "book_on_shelf_1",
                "book_on_shelf_2",
                "book_on_shelf_3",
                "book_on_shelf_4",
                "book_on_shelf_5",
            ),
            "book",
            "board",
        )

    def _add_book_mesh(
        self,
        name: str,
        pos: tuple[float, float, float],
        euler: tuple[float, float, float],
        coacd_options: gs.options.CoacdOptions,
        scale: tuple[float, float, float] = BOOK_SCALE,
    ) -> None:
        entity = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path(BOOK_MESH, pattern_is_dir=False),
                scale=scale,
                pos=pos,
                euler=euler,
                fixed=False,
                collision=True,
                group_by_material=True,
            ),
            material=gs.materials.Rigid(rho=50.0, friction=0.5),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities[name] = {"entity": entity}

    def _build_shelf_scene(
        self,
        coacd_options: gs.options.CoacdOptions,
        *,
        shelf_dx: float = 0.0,
        shelf_dy: float = 0.0,
        book_pos: tuple[float, float, float] = (0.4, 0.0, 0.83),
        book_euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
        book_scale: tuple[float, float, float] = BOOK_SCALE,
        board_pos: tuple[float, float, float] = (0.2, 0.2, 0.80),
        include_shelf_books: bool = True,
        shelf_top_fixed: bool = True,
        include_board: bool = True,
    ) -> None:
        """Build the shelf, the books already on it, the target book, and the board.

        The shelf keeps its cubby topology (left wall at +y, right wall at -y, back
        wall at +x, opening toward the robot) so the inherited reward/success can
        derive the interior from the wall bounds. ``shelf_dx``/``shelf_dy`` translate
        the whole shelf-and-books group together; the target book and board are placed
        independently so variants can perturb the starting layout. With
        ``shelf_top_fixed=False`` the top panel is spawned unanchored, so the robot can
        lift it off; the interior cap is then taken from the side walls instead of the
        panel's live pose.
        """
        dx, dy = shelf_dx, shelf_dy
        shelf_surface = gs.surfaces.Smooth(color=SHELF_COLOR, double_sided=True)
        shelf_material = gs.materials.Rigid(rho=450.0, friction=1.0)

        walls = (
            ("shelf_left", (0.58 + dx, 0.15 + dy, 0.86), (0.16, 0.02, 0.2)),
            ("shelf_right", (0.58 + dx, -0.15 + dy, 0.86), (0.16, 0.02, 0.2)),
            ("shelf_back", (0.67 + dx, 0.0 + dy, 0.86), (0.02, 0.32, 0.2)),
            ("shelf_top", (0.62 + dx, 0.0 + dy, 0.96), (0.1, 0.32, 0.02)),
        )
        self._shelf_top_fixed = shelf_top_fixed
        for name, pos, size in walls:
            wall_fixed = shelf_top_fixed if name == "shelf_top" else True
            entity = self._scene.scene.add_entity(
                gs.morphs.Box(pos=pos, euler=(0.0, 0.0, 0.0), size=size, fixed=wall_fixed, collision=True),
                material=shelf_material,
                surface=shelf_surface,
            )
            self._entities[name] = {"entity": entity}

        if include_shelf_books:
            shelf_books = (
                ("book_on_shelf_1", (0.6 + dx, -0.077 + dy, 0.83), (-43.5, 0.0, 0.0)),
                ("book_on_shelf_2", (0.6 + dx, -0.03 + dy, 0.83), (-43.3, 0.0, 0.0)),
                ("book_on_shelf_3", (0.6 + dx, 0.038 + dy, 0.83), (-15.0, 0.0, 0.0)),
                ("book_on_shelf_4", (0.6 + dx, 0.07 + dy, 0.83), (-15.0, 0.0, 0.0)),
                ("book_on_shelf_5", (0.6 + dx, 0.123 + dy, 0.83), (-1.0, 0.0, 0.0)),
            )
            for name, pos, euler in shelf_books:
                self._add_book_mesh(name, pos, euler, coacd_options)

        self._add_book_mesh("book", book_pos, book_euler, coacd_options, scale=book_scale)

        if include_board:
            board = self._scene.scene.add_entity(
                gs.morphs.Box(
                    pos=board_pos, euler=(0.0, 0.0, 0.0), size=(0.2, 0.03, 0.03), fixed=False, collision=True
                ),
                material=gs.materials.Rigid(rho=50.0, friction=1.0),
                surface=gs.surfaces.Smooth(color=(0.3, 0.3, 0.3), double_sided=True),
            )
            self._entities["board"] = {"entity": board}

    def _add_custom_entities(self) -> None:
        """Add shelf, books on shelf, target book, and board."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )
        self._build_shelf_scene(coacd_options)

    def _check_success(self) -> torch.Tensor:
        """Check if the book is inside the shelf and upright."""
        objs_info = self.collect_objs_info()

        BOOK = "book"
        SHELF_LEFT = "shelf_left"
        SHELF_RIGHT = "shelf_right"
        SHELF_BACK = "shelf_back"
        SHELF_TOP = "shelf_top"
        TOL = 1e-3
        UPRIGHT_TILT_DEG = 50.0

        book = objs_info.get(BOOK)
        shelf_left = objs_info.get(SHELF_LEFT)
        shelf_right = objs_info.get(SHELF_RIGHT)
        shelf_back = objs_info.get(SHELF_BACK)
        shelf_top = objs_info.get(SHELF_TOP)

        if book is None or shelf_left is None or shelf_right is None or shelf_back is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        def get_bounds(obj):
            b = obj.get("bounds")
            if b is None:
                return None
            return np.asarray(b)

        sl_b = get_bounds(shelf_left)
        sr_b = get_bounds(shelf_right)
        sb_b = get_bounds(shelf_back)
        st_b = get_bounds(shelf_top) if shelf_top is not None and self._shelf_top_fixed else None

        if sl_b is None or sr_b is None or sb_b is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        shelf_x_min = float(sl_b[0][0]) + TOL
        shelf_x_max = float(sb_b[0][0]) - TOL
        shelf_y_min = float(sr_b[1][1]) + TOL
        shelf_y_max = float(sl_b[0][1]) - TOL
        # A movable "shelf_top" can be carried off, so the side walls cap the interior instead.
        shelf_z_max = float(st_b[0][2] if st_b is not None else sl_b[1][2]) - TOL

        pos = book.get("pos")
        if pos is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)
        pos = np.asarray(pos)
        bx, by, bz = float(pos[0]), float(pos[1]), float(pos[2])

        inside_shelf = (
            bx > shelf_x_min and bx < shelf_x_max and by > shelf_y_min and by < shelf_y_max and bz < shelf_z_max
        )

        upright = True
        euler = book.get("euler")
        if euler is not None:
            euler = np.asarray(euler).astype(float)
            # collect_objs_info returns euler in radians; the threshold is in degrees.
            upright = abs(np.degrees(euler[0])) <= UPRIGHT_TILT_DEG

        return torch.tensor([bool(inside_shelf and upright)], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Milestone-based progress score in [0, 1] for placing the book on the shelf.

        Phases, in order:
          0.00 -> 0.30  a gripper approaches, reaches, and touches the target book
          0.30 -> 0.70  the book's footprint slides onto the shelf (overlap ratio)
          0.70 -> 0.90  the book is mostly inside and standing upright
          0.90 -> 1.00  the book center is fully inside the shelf and upright (success)

        The ``board`` is only an optional shim, so it earns no reward of its own.
        Everything above the reach floor (0.30) is scaled by how undisturbed the
        books already on the shelf remain, so shoving them out of place is penalised.
        """
        objs_info = self.collect_objs_info()

        TOL = 1e-3
        REACH_THRESHOLD = 0.15
        TOUCH_THRESHOLD = 0.05
        APPROACH_THRESHOLD = 0.3
        BOOK_UPRIGHT_THRESHOLD = 50.0
        # A shelf book that drifts less than the deadzone counts as undisturbed (the
        # leaning books settle by ~1cm on their own); beyond it the "still in place"
        # score ramps down, reaching 0 at DISTURB_MAX_DISP.
        DISTURB_DEADZONE = 0.02
        DISTURB_MAX_DISP = 0.08

        BOOK = "book"
        SHELF_LEFT = "shelf_left"
        SHELF_RIGHT = "shelf_right"
        SHELF_BACK = "shelf_back"
        SHELF_TOP = "shelf_top"
        BOOKS_ON_SHELF = [
            "book_on_shelf_1",
            "book_on_shelf_2",
            "book_on_shelf_3",
            "book_on_shelf_4",
            "book_on_shelf_5",
        ]

        def get_pos(name: str) -> np.ndarray | None:
            obj = objs_info.get(name)
            if obj is None:
                return None
            p = obj.get("pos")
            if p is None:
                return None
            return np.array([p[0], p[1], p[2]], dtype=float)

        def get_bounds(name: str) -> np.ndarray | None:
            obj = objs_info.get(name)
            if obj is None:
                return None
            b = obj.get("bounds")
            if b is None:
                return None
            return np.array(b, dtype=float)

        def get_euler(name: str) -> np.ndarray | None:
            obj = objs_info.get(name)
            if obj is None:
                return None
            e = obj.get("euler")
            if e is None:
                return None
            return np.array([e[0], e[1], e[2]], dtype=float)

        def get_spawn_pos(name: str) -> np.ndarray | None:
            info = self._entities.get(name)
            if info is None:
                return None
            sp = info.get("spawn_pos")
            if sp is None:
                sp = info.get("initial_pos")
            return np.array(sp, dtype=float) if sp is not None else None

        def get_hull_2d(name: str) -> np.ndarray | None:
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
                [
                    [b[0, 0], b[0, 1]],
                    [b[1, 0], b[0, 1]],
                    [b[1, 0], b[1, 1]],
                    [b[0, 0], b[1, 1]],
                ],
                dtype=float,
            )

        def get_shelf_bounds() -> np.ndarray | None:
            sl_b = get_bounds(SHELF_LEFT)
            sr_b = get_bounds(SHELF_RIGHT)
            sb_b = get_bounds(SHELF_BACK)
            st_b = get_bounds(SHELF_TOP) if self._shelf_top_fixed else None
            if sl_b is None or sr_b is None or sb_b is None:
                return None
            shelf_x_min = float(sl_b[0, 0]) + TOL
            shelf_x_max = float(sb_b[0, 0]) - TOL
            shelf_y_min = float(sr_b[1, 1]) + TOL
            shelf_y_max = float(sl_b[0, 1]) - TOL
            shelf_z_min = float(sl_b[0, 2])
            # A movable "shelf_top" can be carried off, so the side walls cap the interior instead.
            shelf_z_max = float(st_b[0, 2] if st_b is not None else sl_b[1, 2]) - TOL
            return np.array(
                [[shelf_x_min, shelf_y_min, shelf_z_min], [shelf_x_max, shelf_y_max, shelf_z_max]],
                dtype=float,
            )

        def bounds_intersect(b1: np.ndarray | None, b2: np.ndarray | None) -> bool:
            if b1 is None or b2 is None:
                return False
            tol = 0.01
            overlap_x = b1[1, 0] >= b2[0, 0] - tol and b2[1, 0] >= b1[0, 0] - tol
            overlap_y = b1[1, 1] >= b2[0, 1] - tol and b2[1, 1] >= b1[0, 1] - tol
            overlap_z = b1[1, 2] >= b2[0, 2] - tol and b2[1, 2] >= b1[0, 2] - tol
            return overlap_x and overlap_y and overlap_z

        def poly_area(poly: np.ndarray | None) -> float:
            if poly is None or len(poly) < 3:
                return 0.0
            x = poly[:, 0]
            y = poly[:, 1]
            return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

        def compute_overlap_ratio_with_rect(
            hull: np.ndarray | None, rect_min: np.ndarray, rect_max: np.ndarray
        ) -> float:
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
                rxs = r[0] * s[1] - r[1] * s[0]
                q_p = q1 - p1
                if abs(rxs) < 1e-12:
                    return (p2 + q2) / 2.0
                t = (q_p[0] * s[1] - q_p[1] * s[0]) / rxs
                return p1 + t * r

            def suth_hodg(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
                output = subject.copy()
                clip = ensure_ccw(clip)
                for i in range(len(clip)):
                    a_pt = clip[i]
                    b_pt = clip[(i + 1) % len(clip)]
                    new_output = []
                    if len(output) == 0:
                        return np.zeros((0, 2), dtype=float)
                    s_pt = output[-1]
                    for e_pt in output:

                        def is_inside(p, _a=a_pt, _b=b_pt):
                            return ((_b[0] - _a[0]) * (p[1] - _a[1]) - (_b[1] - _a[1]) * (p[0] - _a[0])) >= -1e-12

                        if is_inside(e_pt):
                            if not is_inside(s_pt):
                                new_output.append(line_intersection(s_pt, e_pt, a_pt, b_pt))
                            new_output.append(e_pt)
                        elif is_inside(s_pt):
                            new_output.append(line_intersection(s_pt, e_pt, a_pt, b_pt))
                        s_pt = e_pt
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

        shelf_min = shelf_bounds[0]
        shelf_max = shelf_bounds[1]
        shelf_z_min = float(shelf_min[2])
        shelf_z_max = float(shelf_max[2])

        right_ee_pos = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee_pos = self._robot.left_ee_pose[0, :3].cpu().numpy()

        # Progress of the target book from the table into the shelf. The board is only
        # an optional shim and earns no reward of its own -- any usefulness it provides
        # shows up as book progress.
        book_score = 0.0
        book_success = False
        if BOOK in objs_info:
            book_pos = get_pos(BOOK)
            if book_pos is not None:
                bx, by, bz = float(book_pos[0]), float(book_pos[1]), float(book_pos[2])
                min_dist_book = min(
                    float(np.linalg.norm(right_ee_pos - book_pos)),
                    float(np.linalg.norm(left_ee_pos - book_pos)),
                )
            else:
                bx = by = bz = float("inf")
                min_dist_book = float("inf")

            book_bounds = get_bounds(BOOK)
            book_hull = get_hull_2d(BOOK)
            book_inside_ratio = compute_overlap_ratio_with_rect(book_hull, shelf_min[:2], shelf_max[:2])
            book_intersects_shelf = bounds_intersect(book_bounds, shelf_bounds) if book_bounds is not None else False
            book_euler = get_euler(BOOK)
            # collect_objs_info returns euler in radians; BOOK_UPRIGHT_THRESHOLD is in degrees.
            book_euler_x = abs(np.degrees(float(book_euler[0]))) if book_euler is not None else float("inf")
            book_upright_progress = (
                max(0.0, 1.0 - book_euler_x / BOOK_UPRIGHT_THRESHOLD) if book_euler_x <= BOOK_UPRIGHT_THRESHOLD else 0.0
            )

            # The book has to end up at shelf level, not resting on top of the shelf.
            z_in_shelf = bz < shelf_z_max + TOL and bz > shelf_z_min - 0.10

            # Same criterion as _check_success(): center inside the shelf box and upright.
            book_success = (
                shelf_min[0] < bx < shelf_max[0]
                and shelf_min[1] < by < shelf_max[1]
                and bz < shelf_z_max
                and book_euler_x <= BOOK_UPRIGHT_THRESHOLD
            )

            # EE guidance: approach -> reach -> touch, capped at 0.30.
            if min_dist_book <= REACH_THRESHOLD:
                reach_progress = 1.0 - (min_dist_book - TOUCH_THRESHOLD) / (REACH_THRESHOLD - TOUCH_THRESHOLD)
                reach_progress = max(0.0, min(1.0, reach_progress))
                ee_based = 0.15 + reach_progress * 0.15
            elif min_dist_book < APPROACH_THRESHOLD:
                approach_progress = 1.0 - (min_dist_book - REACH_THRESHOLD) / (APPROACH_THRESHOLD - REACH_THRESHOLD)
                approach_progress = max(0.0, min(1.0, approach_progress))
                ee_based = approach_progress * 0.15
            else:
                ee_based = 0.0

            # Placement: how much of the book footprint sits over the shelf, at shelf level.
            if book_inside_ratio > 0.5 and z_in_shelf:
                progress_to_full = min(1.0, (book_inside_ratio - 0.5) / 0.5)
                placement_score = 0.70 + progress_to_full * 0.15 + book_upright_progress * 0.05
            elif book_inside_ratio > 0.0 and z_in_shelf:
                placement_score = 0.30 + min(1.0, book_inside_ratio / 0.5) * 0.40
            elif book_intersects_shelf and z_in_shelf:
                placement_score = 0.30
            else:
                placement_score = 0.0

            book_score = max(ee_based, placement_score)

        # The books already on the shelf must stay put. Score each one by its
        # displacement from its per-episode spawn pose and take the mean fraction that
        # is still in place.
        undisturbed_scores = []
        for book_name in BOOKS_ON_SHELF:
            spawn = get_spawn_pos(book_name)
            cur = get_pos(book_name)
            if spawn is None or cur is None:
                continue
            disp = float(np.linalg.norm(cur - spawn))
            excess = max(0.0, disp - DISTURB_DEADZONE)
            undisturbed_scores.append(max(0.0, min(1.0, 1.0 - excess / (DISTURB_MAX_DISP - DISTURB_DEADZONE))))
        undisturbed_frac = float(np.mean(undisturbed_scores)) if undisturbed_scores else 1.0

        # Keep the reach/approach guidance unconditional (floor of 0.30) so the policy is
        # always led to the book; gate everything above it on leaving the existing books
        # undisturbed.
        floor = min(book_score, 0.30)
        gated_score = floor + (book_score - floor) * undisturbed_frac

        # Full success tops the score out, still discounted if the existing books moved.
        if book_success:
            gated_score = max(gated_score, 0.90 + 0.10 * undisturbed_frac)

        total_score = float(max(0.0, min(1.0, gated_score)))
        score_t = torch.tensor(total_score, dtype=torch.float32, device=self._device)
        return score_t, {"score": score_t}
