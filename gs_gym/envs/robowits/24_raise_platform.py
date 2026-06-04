"""RaisePlatform environment for RoboWits.

Task: Place the heavy book on top of the cup supports.
"""

from __future__ import annotations

import math

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits import utils as _utils
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv


@register_task("robowits/24-raise-platform-v0")
class RaisePlatformEnv(RoboWitsEnv):
    """Place the heavy book on top of the cup supports.

    The goal requires supporting a load above the table. Assembling two
    equal-height supports and a rigid deck creates a stable platform that
    can hold the book without touching the table.

    Success criteria:
    - Book is resting on both cup supports
    - Book is elevated above the table surface
    - Involved objects haven't fallen below the table
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Place the book on the cup supports."

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
        large_area = {"x_min": 0.25, "x_max": 0.65, "y_min": -0.5, "y_max": 0.5}
        self._object_reachable_areas["cup support 2"] = large_area
        self._object_reachable_areas["heavy book"] = large_area

    @property
    def placement_groups(self) -> PlacementGroups:
        """Book and support cube grouped, other support cube independent."""
        return (("heavy book", "support cube 2"), "support cube 1")

    def _add_custom_entities(self) -> None:
        """Add heavy book, cup supports, and support cubes."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Heavy book (mesh)
        book = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/196d92f1-10b5-4563-ae7f-0b26b615ce51/obj.glb", pattern_is_dir=False),
                scale=0.7,
                pos=(0.70, -0.32, 0.7742 + 0.03),
                euler=(90.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=100),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["heavy book"] = {
            "entity": book,
        }

        # Cup support 1 (mesh, fixed)
        cup1 = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/94134b02-c73e-4f2c-ad1d-a00a78160d98/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.64, -0.10, 0.8082),
                euler=(0.0, 180.0, 0.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["cup support 1"] = {
            "entity": cup1,
        }

        # Cup support 2 (mesh, fixed)
        cup2 = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/94134b02-c73e-4f2c-ad1d-a00a78160d98/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.64, 0.10, 0.8082),
                euler=(180.0, 0.0, 0.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["cup support 2"] = {
            "entity": cup2,
        }

        # Support cube 1 (box, fixed)
        cube1 = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.57, 0.0, 0.3),
                euler=(0.0, 0.0, 0.0),
                size=(0.02, 0.02, 0.03),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5),
        )
        self._entities["support cube 1"] = {
            "entity": cube1,
        }

        # Support cube 2 (box, fixed)
        cube2 = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.68, -0.32, 0.76 + 0.015),
                euler=(0.0, 0.0, 0.0),
                size=(0.02, 0.02, 0.03),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5),
        )
        self._entities["support cube 2"] = {
            "entity": cube2,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: book resting on cup supports."""
        objs_info = self.collect_objs_info()

        ELEV_CLEARANCE = 0.005
        CONTACT_TOL = 0.035
        SUPPORT_HORIZ_TOL = 0.05
        FALL_BELOW_TOL = 0.05

        cup1 = objs_info.get("cup support 1")
        cup2 = objs_info.get("cup support 2")
        book = objs_info.get("heavy book")

        if cup1 is None or cup2 is None or book is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        if cup1.get("bounds") is None or cup2.get("bounds") is None or book.get("bounds") is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        def get_poly(obj):
            hull = obj.get("convex_hull_2d")
            if hull is not None and len(hull) >= 3:
                return [(float(p[0]), float(p[1])) for p in hull]
            b = obj.get("bounds")
            if b is None:
                return None
            xmin, ymin = b[0][0], b[0][1]
            xmax, ymax = b[1][0], b[1][1]
            return [
                (float(xmin), float(ymin)),
                (float(xmax), float(ymin)),
                (float(xmax), float(ymax)),
                (float(xmin), float(ymax)),
            ]

        def obj_center_xy(obj):
            p = obj.get("pos")
            if p is None:
                return None
            if len(p) < 2:
                return None
            return (float(p[0]), float(p[1]))

        def point_in_poly(pt, poly):
            if pt is None or poly is None or len(poly) < 3:
                return False
            x, y = pt
            inside = False
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                if (y1 > y) != (y2 > y):
                    xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
                    if x < xinters:
                        inside = not inside
            return inside

        def point_to_poly_dist(pt, poly):
            if pt is None or poly is None or len(poly) < 2:
                return float("inf")
            x, y = pt
            min_d = float("inf")
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                dx, dy = x2 - x1, y2 - y1
                seg_len2 = dx * dx + dy * dy
                if seg_len2 == 0:
                    d = math.hypot(x - x1, y - y1)
                else:
                    t = ((x - x1) * dx + (y - y1) * dy) / seg_len2
                    t = max(0.0, min(1.0, t))
                    projx = x1 + t * dx
                    projy = y1 + t * dy
                    d = math.hypot(x - projx, y - projy)
                if d < min_d:
                    min_d = d
            return min_d

        book_poly = get_poly(book)
        cup1_center = obj_center_xy(cup1)
        cup2_center = obj_center_xy(cup2)

        cup1_zmax = float(cup1["bounds"][1][2])
        cup2_zmax = float(cup2["bounds"][1][2])
        book_zmin = float(book["bounds"][0][2])

        for obj in (cup1, cup2, book):
            if float(obj["bounds"][0][2]) < (self.TABLE_Z - FALL_BELOW_TOL):
                return torch.tensor([False], dtype=torch.bool, device=self._device)

        if (book_zmin - self.TABLE_Z) <= ELEV_CLEARANCE:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        def is_cup_under_book(cup_center, _book_poly=book_poly):
            if _book_poly is None or cup_center is None:
                return False
            if point_in_poly(cup_center, _book_poly):
                return True
            return point_to_poly_dist(cup_center, _book_poly) <= SUPPORT_HORIZ_TOL

        if not (is_cup_under_book(cup1_center) and is_cup_under_book(cup2_center)):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        def is_in_contact_vert(book_bottom, support_top):
            return abs(book_bottom - support_top) <= CONTACT_TOL and (book_bottom >= support_top - CONTACT_TOL)

        if not (is_in_contact_vert(book_zmin, cup1_zmax) and is_in_contact_vert(book_zmin, cup2_zmax)):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        if _utils.check_being_held(book, self._robot):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        return torch.tensor([True], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute reward: 1.0 on success, else milestone progress in [0, 1)."""
        success = self._check_success()
        if success.item():
            reward = torch.ones(1, dtype=torch.float32, device=self._device)
            return reward, {"success": reward}

        objs_info = self.collect_objs_info()
        cup1 = objs_info.get("cup support 1")
        cup2 = objs_info.get("cup support 2")
        book = objs_info.get("heavy book")

        zero = torch.zeros(1, dtype=torch.float32, device=self._device)
        if cup1 is None or cup2 is None or book is None:
            return zero, {"success": zero}
        if book.get("bounds") is None or cup1.get("bounds") is None or cup2.get("bounds") is None:
            return zero, {"success": zero}

        book_bounds = np.array(book["bounds"], dtype=float)
        cup1_bounds = np.array(cup1["bounds"], dtype=float)
        cup2_bounds = np.array(cup2["bounds"], dtype=float)

        book_zmin = book_bounds[0, 2]
        book_pos = np.array(book.get("pos", [0.0, 0.0, 0.0]), dtype=float)
        cup1_zmax = cup1_bounds[1, 2]
        cup2_zmax = cup2_bounds[1, 2]
        target_z = (cup1_zmax + cup2_zmax) / 2.0

        # EE proximity to book
        APPROACH, REACH, TOUCH = 0.3, 0.15, 0.05
        right_ee = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee = self._robot.left_ee_pose[0, :3].cpu().numpy()
        min_dist = min(float(np.linalg.norm(right_ee - book_pos)), float(np.linalg.norm(left_ee - book_pos)))
        if min_dist <= TOUCH:
            ee_score = 0.2
        elif min_dist <= REACH:
            ee_score = 0.1 + 0.1 * (REACH - min_dist) / (REACH - TOUCH)
        elif min_dist < APPROACH:
            ee_score = 0.1 * (APPROACH - min_dist) / (APPROACH - REACH)
        else:
            ee_score = 0.0

        # Book lift progress toward cup tops (0.2 → 0.5)
        initial_z = self.TABLE_Z
        lift_frac = max(0.0, min(1.0, (book_zmin - initial_z) / max(target_z - initial_z, 1e-6)))
        lift_score = 0.2 + 0.3 * lift_frac

        # XY alignment: how close book center is to spanning both cup centers (0.5 → 0.8)
        cup1_center = np.array([cup1_bounds[:, 0].mean(), cup1_bounds[:, 1].mean()])
        cup2_center = np.array([cup2_bounds[:, 0].mean(), cup2_bounds[:, 1].mean()])
        cups_midpoint = (cup1_center + cup2_center) / 2.0
        xy_dist = float(np.linalg.norm(book_pos[:2] - cups_midpoint))
        book_half = max(book_bounds[1, 0] - book_bounds[0, 0], book_bounds[1, 1] - book_bounds[0, 1]) / 2.0
        align_frac = max(0.0, min(1.0, 1.0 - xy_dist / max(book_half, 1e-6)))
        align_score = 0.5 + 0.3 * align_frac

        # Placement: book near cup top height and aligned (0.8 → 1.0, but not held)
        near_height = abs(book_zmin - target_z) <= 0.05
        held = _utils.check_being_held(book, self._robot)
        place_score = 0.8 + 0.2 * align_frac if near_height and align_frac > 0.5 and not held else 0.0

        score = max(ee_score, lift_score, align_score if lift_frac > 0.5 else 0.0, place_score)
        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"success": zero}
