"""InsertCylinder environment for RoboWits.

Task: Stand a cylindrical peg upright in one of the vertical holes of a hole plate.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv


@register_task("robowits/18-insert-cylinder-v0")
class InsertCylinderEnv(RoboWitsEnv):
    """Stand a cylindrical peg upright in one of the vertical holes of a hole plate.

    The peg starts lying flat on the table, so it must be re-oriented from horizontal to
    vertical and then aligned coaxially with a hole whose radial clearance is under 3 mm.
    This exercises cross-section geometry and orientation matching.

    The plate is a fixed block resting on the table (world z 0.753 - 0.813) with two
    vertical through-holes; because the plate bottom sits at/below the table surface, a
    peg driven all the way down stops on the table with its top still above the plate.
    "Through the hole" therefore means seated on the table while protruding out the top,
    not passed out the underside.

    Success criteria:
    - Peg is upright (vertical extent is at least ``UPRIGHT_FRAC`` of its length)
    - Peg has descended through a hole until it rests on the table surface
    - Peg still protrudes above the top face of the plate
    - Peg axis is within ``HOLE_XY_TOL`` of one of the two hole centers
    - Peg remains within table bounds
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Insert the yellow rod into one of the round holes in the white block so it stands upright in the hole."

    TABLE_Z = 0.76

    # Peg dimensions of blender_kit/ecfc80a0 at scale 0.8, in meters, as measured from the
    # built scene (91 mm long, 19.5 mm across).
    PEG_LENGTH = 0.091
    PEG_RADIUS = 0.0098

    # Hole centers as (dx, dy) offsets from the plate origin. Measured by ray-casting the
    # plate's visual mesh in world frame after the morph's scale and euler are applied: the
    # two bores run vertically and sit side by side along world y. They are slightly
    # elliptical, ~27 mm across in x and ~33 mm in y, so radial clearance around the peg is
    # under 4 mm at the tight axis.
    HOLE_OFFSETS_XY = ((0.004, -0.027), (0.004, 0.026))

    # A peg tilted more than ~26 deg from vertical cannot be seated in a bore this tight.
    UPRIGHT_FRAC = 0.9
    # Slack on "resting on the table" and on coaxial alignment with a hole center. The
    # alignment tolerance is kept just under the radial clearance: the plate's convex
    # collision hull is permeable in places, so this check -- not contact -- is what
    # separates a real insertion from the peg sinking through the plate body.
    SEAT_TOL = 0.008
    HOLE_XY_TOL = 0.008

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

    TABLE_X_MIN, TABLE_X_MAX = 0.21, 1.00
    TABLE_Y_MIN, TABLE_Y_MAX = -0.69, 0.69

    @staticmethod
    def _get_poly_xy(obj):
        poly = obj.get("convex_hull_2d", None)
        if poly is not None and isinstance(poly, np.ndarray) and poly.shape[0] >= 3:
            return poly
        b = obj.get("bounds", None)
        if b is None:
            return None
        (xmin, ymin, _), (xmax, ymax, _) = b
        return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float)

    def _hull_within_table(self, poly) -> bool:
        if poly is None:
            return False
        xs, ys = poly[:, 0], poly[:, 1]
        return (
            xs.min() >= self.TABLE_X_MIN - 1e-3
            and xs.max() <= self.TABLE_X_MAX + 1e-3
            and ys.min() >= self.TABLE_Y_MIN - 1e-3
            and ys.max() <= self.TABLE_Y_MAX + 1e-3
        )

    def _hole_xy_dist(self, plate_pos: np.ndarray, peg_xy: np.ndarray) -> float:
        """Planar distance from the peg axis to the nearest of the two hole centers."""
        return min(
            float(np.linalg.norm(peg_xy - (plate_pos[:2] + np.asarray(off, dtype=float))))
            for off in self.HOLE_OFFSETS_XY
        )

    def _insertion_state(self) -> dict | None:
        """Geometry shared by success and reward, or None when objects are unavailable."""
        objs_info = self.collect_objs_info()

        peg = objs_info.get("cylindrical_peg")
        plate = objs_info.get("hole_plate")
        if peg is None or plate is None:
            return None

        peg_bounds = peg.get("bounds")
        plate_bounds = plate.get("bounds")
        if peg_bounds is None or plate_bounds is None:
            return None

        peg_min = np.array(peg_bounds[0], dtype=float)
        peg_max = np.array(peg_bounds[1], dtype=float)
        peg_poly = self._get_poly_xy(peg)

        # AABB center is used instead of the body origin so that the reference point stays
        # on the peg axis regardless of where the mesh pivot sits.
        peg_xy = (peg_min[:2] + peg_max[:2]) / 2.0
        plate_pos = np.array(plate.get("pos", [0.0, 0.0, 0.0]), dtype=float)

        # A vertical extent close to the full length means the peg is standing up; it shrinks
        # as cos(tilt), so this doubles as an orientation measure without needing the quat.
        upright_frac = float(peg_max[2] - peg_min[2]) / self.PEG_LENGTH

        return {
            "peg_z_min": float(peg_min[2]),
            "peg_z_max": float(peg_max[2]),
            "peg_pos": np.array(peg.get("pos", [0.0, 0.0, 0.0]), dtype=float),
            "peg_poly": peg_poly,
            "plate_z_max": float(plate_bounds[1][2]),
            "upright_frac": upright_frac,
            "hole_dist": self._hole_xy_dist(plate_pos, peg_xy),
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: peg standing upright in a hole, seated on the table."""
        st = self._insertion_state()
        if st is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        upright = st["upright_frac"] >= self.UPRIGHT_FRAC
        seated = st["peg_z_min"] <= self.TABLE_Z + self.SEAT_TOL
        protrudes = st["peg_z_max"] >= st["plate_z_max"]
        aligned = st["hole_dist"] <= self.HOLE_XY_TOL
        on_table = self._hull_within_table(st["peg_poly"])

        success = upright and seated and protrudes and aligned and on_table
        return torch.tensor([success], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute reward: 1.0 on success, else milestone progress in [0, 1)."""
        success = self._check_success()
        if success.item():
            reward = torch.ones(1, dtype=torch.float32, device=self._device)
            return reward, {"success": reward}

        zero = torch.zeros(1, dtype=torch.float32, device=self._device)
        st = self._insertion_state()
        if st is None:
            return zero, {"success": zero}

        peg_z_min = st["peg_z_min"]
        plate_z_max = st["plate_z_max"]
        upright_frac = st["upright_frac"]
        hole_dist = st["hole_dist"]

        # EE proximity to the peg (0.0 -> 0.2)
        APPROACH = 0.3
        REACH = 0.15
        TOUCH = 0.05
        right_ee = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee = self._robot.left_ee_pose[0, :3].cpu().numpy()
        peg_pos = st["peg_pos"]
        min_dist = min(float(np.linalg.norm(right_ee - peg_pos)), float(np.linalg.norm(left_ee - peg_pos)))
        if min_dist <= TOUCH:
            ee_score = 0.2
        elif min_dist <= REACH:
            ee_score = 0.1 + 0.1 * (REACH - min_dist) / (REACH - TOUCH)
        elif min_dist < APPROACH:
            ee_score = 0.1 * (APPROACH - min_dist) / (APPROACH - REACH)
        else:
            ee_score = 0.0

        # Peg lifted clear of the plate top, so it can be lowered into a bore (0.2 -> 0.4)
        clearance = plate_z_max - self.TABLE_Z
        lift_frac = min(1.0, max(0.0, (peg_z_min - self.TABLE_Z) / max(clearance, 1e-6)))
        # Gated on an actual lift so a peg still lying on the table scores through ee_score
        # only, keeping the floor of an idle episode near zero.
        lift_score = 0.2 + 0.2 * lift_frac if peg_z_min >= self.TABLE_Z + 0.01 else 0.0
        cleared = peg_z_min >= plate_z_max

        # Uprighted while held clear (0.4 -> 0.6)
        upright_progress = min(1.0, max(0.0, upright_frac / self.UPRIGHT_FRAC))
        upright_score = 0.4 + 0.2 * upright_progress if cleared else 0.0

        # Coaxial alignment with the nearest hole, upright and clear (0.6 -> 0.8)
        ALIGN_RANGE = 0.10
        align_frac = min(1.0, max(0.0, 1.0 - hole_dist / ALIGN_RANGE))
        ready = cleared and upright_frac >= self.UPRIGHT_FRAC
        align_score = 0.6 + 0.2 * align_frac if ready else 0.0

        # Descent through the bore down to the table (0.8 -> 1.0)
        descent_frac = min(1.0, max(0.0, (plate_z_max - peg_z_min) / max(clearance, 1e-6)))
        inserting = upright_frac >= self.UPRIGHT_FRAC and hole_dist <= self.HOLE_XY_TOL
        descent_score = 0.8 + 0.2 * descent_frac if inserting else 0.0

        score = max(ee_score, lift_score, upright_score, align_score, descent_score)
        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"success": zero}
