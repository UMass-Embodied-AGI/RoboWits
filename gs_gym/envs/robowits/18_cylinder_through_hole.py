"""CylinderThroughHole environment for RoboWits.

Task: Insert a cylindrical peg through a matching hole plate.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv


@register_task("robowits/18-cylinder-through-hole-v0")
class CylinderThroughHoleEnv(RoboWitsEnv):
    """Insert a cylindrical peg through a matching hole plate.

    A circular peg requires coaxial alignment to pass a circular hole.
    This uses cross-section geometry and orientation matching.

    Success criteria:
    - At least half the peg length has passed through the hole plate
    - Peg straddles the plate thickness (part above, part below)
    - Peg lies over the collection zone
    - Peg remains within table bounds
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Insert the long rod through the opening so it rests over the green colored patch."

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
        closer_area = {"x_min": 0.25, "x_max": 0.55, "y_min": -0.3, "y_max": 0.3}
        self._object_reachable_areas["cylindrical_peg"] = closer_area

    @property
    def placement_groups(self) -> PlacementGroups:
        """Only peg is randomized."""
        return ("cylindrical_peg",)

    def _add_custom_entities(self) -> None:
        """Add collection zone, hole plate, and cylindrical peg."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Collection zone (box, fixed, no collision)
        collection_zone = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.55, 0.0, 0.761),
                euler=(0.0, 0.0, 0.0),
                size=(0.22, 0.18, 0.002),
                fixed=True,
                collision=False,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.1, 0.8, 0.2), roughness=0.6),
        )
        self._entities["collection_zone"] = {
            "entity": collection_zone,
        }

        # Hole plate (mesh, fixed)
        coacd_fine = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=200, max_convex_hull=50, decimate=True
        )
        hole_plate = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_fine,
                file=get_asset_path("hf_assets/holeplate.glb", pattern_is_dir=False),
                scale=(1.2, 1.0, 1.2),
                pos=(0.55, 0.0, 0.7823),
                euler=(0.0, 90.0, -90.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(color=(0.95, 0.95, 0.95), double_sided=True),
        )
        self._entities["hole_plate"] = {
            "entity": hole_plate,
        }

        # Cylindrical peg (mesh)
        peg = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/ecfc80a0-4318-4ac8-8ec0-fa1e355d1521/obj.glb", pattern_is_dir=False),
                scale=0.8,
                pos=(0.67, -0.18, 0.78),
                euler=(0.0, 90.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.8),
            surface=gs.surfaces.Smooth(color=(0.8, 0.8, 0.0), double_sided=True),
        )
        self._entities["cylindrical_peg"] = {
            "entity": peg,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: peg through hole, over collection zone."""
        objs_info = self.collect_objs_info()

        TABLE_X_MIN, TABLE_X_MAX = 0.21, 1.00
        TABLE_Y_MIN, TABLE_Y_MAX = -0.69, 0.69

        def get_obj(name):
            if name in objs_info:
                return objs_info[name]
            lname = name.lower()
            for k in objs_info:
                if k.lower() == lname:
                    return objs_info[k]
            return None

        peg = get_obj("cylindrical peg") or get_obj("cylindrical_peg")
        plate = get_obj("hole plate") or get_obj("hole_plate")
        zone = get_obj("collection zone") or get_obj("collection_zone")

        if peg is None or plate is None or zone is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        def get_poly_xy(obj):
            poly = obj.get("convex_hull_2d", None)
            if poly is not None and isinstance(poly, np.ndarray) and poly.shape[0] >= 3:
                return poly
            b = obj.get("bounds", None)
            if b is None:
                return None
            (xmin, ymin, _), (xmax, ymax, _) = b
            return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float)

        def point_in_poly(pt, poly):
            x, y = pt
            inside = False
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                if (y1 > y) != (y2 > y):
                    xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
                    if x <= xinters:
                        inside = not inside
            return inside

        def polys_overlap(p1, p2):
            if p1 is None or p2 is None:
                return False
            if any(point_in_poly((v[0], v[1]), p2) for v in p1):
                return True
            return any(point_in_poly((v[0], v[1]), p1) for v in p2)

        peg_bounds = peg.get("bounds", None)
        plate_bounds = plate.get("bounds", None)
        if peg_bounds is None or plate_bounds is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        peg_min = np.array(peg_bounds[0])
        peg_max = np.array(peg_bounds[1])
        peg_sizes = peg_max - peg_min
        peg_length_est = float(np.max(peg_sizes))

        plate_z_min = float(plate_bounds[0][2])
        plate_z_max = float(plate_bounds[1][2])

        peg_z_min = float(peg_bounds[0][2])
        peg_z_max = float(peg_bounds[1][2])

        straddles_plate = (peg_z_min < plate_z_min) and (peg_z_max > plate_z_max)
        inserted_below_bottom = max(0.0, plate_z_min - peg_z_min)
        enough_insertion = inserted_below_bottom >= 0.5 * peg_length_est

        peg_center_xy = np.array(peg.get("pos", [0, 0, 0]))[:2].astype(float)
        zone_poly = get_poly_xy(zone)
        peg_poly = get_poly_xy(peg)
        over_zone = False
        if zone_poly is not None:
            over_zone = point_in_poly(peg_center_xy, zone_poly)
            if not over_zone and peg_poly is not None:
                over_zone = polys_overlap(peg_poly, zone_poly)

        def hull_within_table(poly):
            if poly is None:
                return False
            xs = poly[:, 0]
            ys = poly[:, 1]
            return (
                xs.min() >= TABLE_X_MIN - 1e-3
                and xs.max() <= TABLE_X_MAX + 1e-3
                and ys.min() >= TABLE_Y_MIN - 1e-3
                and ys.max() <= TABLE_Y_MAX + 1e-3
            )

        peg_on_table = hull_within_table(peg_poly)

        success = straddles_plate and enough_insertion and over_zone and peg_on_table
        return torch.tensor([success], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute reward: 1.0 on success, else milestone progress in [0, 1)."""
        success = self._check_success()
        if success.item():
            reward = torch.ones(1, dtype=torch.float32, device=self._device)
            return reward, {"success": reward}

        objs_info = self.collect_objs_info()

        def get_obj(name):
            if name in objs_info:
                return objs_info[name]
            lname = name.lower()
            for k in objs_info:
                if k.lower() == lname:
                    return objs_info[k]
            return None

        peg = get_obj("cylindrical peg") or get_obj("cylindrical_peg")
        plate = get_obj("hole plate") or get_obj("hole_plate")
        zone = get_obj("collection zone") or get_obj("collection_zone")

        zero = torch.zeros(1, dtype=torch.float32, device=self._device)
        if peg is None or plate is None or zone is None:
            return zero, {"success": zero}

        peg_bounds = peg.get("bounds")
        plate_bounds = plate.get("bounds")
        if peg_bounds is None or plate_bounds is None:
            return zero, {"success": zero}

        peg_min = np.array(peg_bounds[0], dtype=float)
        peg_max = np.array(peg_bounds[1], dtype=float)
        peg_length_est = float(np.max(peg_max - peg_min))
        peg_z_min = peg_min[2]
        peg_z_max = peg_max[2]

        plate_z_min = float(plate_bounds[0][2])
        plate_z_max = float(plate_bounds[1][2])

        # How much of the peg is below the plate bottom (insertion depth)
        insertion_below = max(0.0, plate_z_min - peg_z_min)
        target_insertion = 0.5 * peg_length_est
        insertion_frac = min(1.0, insertion_below / max(target_insertion, 1e-6))

        straddles_plate = (peg_z_min < plate_z_min) and (peg_z_max > plate_z_max)
        enough_insertion = insertion_below >= target_insertion

        # XY distance from peg center to zone center
        peg_pos = np.array(peg.get("pos", [0.0, 0.0, 0.0]), dtype=float)
        zone_pos = np.array(zone.get("pos", [0.0, 0.0, 0.0]), dtype=float)
        zone_bounds = zone.get("bounds")
        if zone_bounds is not None:
            zone_half = (np.array(zone_bounds[1]) - np.array(zone_bounds[0]))[:2] / 2.0
            zone_reach = float(np.max(zone_half))
        else:
            zone_reach = 0.1
        xy_dist_to_zone = float(np.linalg.norm(peg_pos[:2] - zone_pos[:2]))
        zone_approach = max(0.0, min(1.0, 1.0 - xy_dist_to_zone / max(zone_reach * 3.0, 1e-6)))

        # EE proximity to peg
        APPROACH = 0.3
        REACH = 0.15
        TOUCH = 0.05
        right_ee = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee = self._robot.left_ee_pose[0, :3].cpu().numpy()
        min_dist = min(float(np.linalg.norm(right_ee - peg_pos)), float(np.linalg.norm(left_ee - peg_pos)))
        if min_dist <= TOUCH:
            ee_score = 0.2
        elif min_dist <= REACH:
            ee_score = 0.1 + 0.1 * (REACH - min_dist) / (REACH - TOUCH)
        elif min_dist < APPROACH:
            ee_score = 0.1 * (APPROACH - min_dist) / (APPROACH - REACH)
        else:
            ee_score = 0.0

        # Peg lifted to plate level (0.2 → 0.3)
        if peg_z_max >= plate_z_max:
            lift_score = 0.3
        elif peg_z_max >= plate_z_min:
            lift_score = 0.2 + 0.1 * (peg_z_max - plate_z_min) / max(plate_z_max - plate_z_min, 1e-6)
        else:
            lift_score = 0.0

        # Insertion (0.3 → 0.7)
        insertion_score = 0.3 + 0.4 * insertion_frac if straddles_plate else 0.0

        # Over zone while enough inserted (0.7 → 1.0)
        zone_score = 0.7 + 0.3 * zone_approach if enough_insertion else 0.0

        score = max(ee_score, lift_score, insertion_score, zone_score)
        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"success": zero}
