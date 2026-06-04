"""DifferentiateCubes environment for RoboWits.

Task: Identify and place the wooden cube on the target area, keep the metal cube in the mug.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits import utils as _utils
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv


@register_task("robowits/30-differentiate-cubes-v0")
class DifferentiateCubesEnv(RoboWitsEnv):
    """Differentiate wooden and metal cubes using buoyancy.

    The robot must differentiate two similar-looking objects with different
    material properties. By exploiting buoyancy (wood floats, metal sinks),
    the robot can identify which cube is wooden.

    Success criteria:
    - Wooden cube overlaps target area by at least 50%
    - Metal cube remains inside the mug
    - Cubes remain within table bounds
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Place the wooden cube to the green target area. Keep the metal cube inside the mug."

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

    def _config_to_env_args(self, config):
        args = super()._config_to_env_args(config)
        args = args.model_copy(update={"scene_args": args.scene_args.model_copy(update={"substeps": 100})})
        return args

    @property
    def placement_groups(self) -> PlacementGroups:
        """No randomization for this task."""
        return (("mug", "wooden cube", "metal cube"), "target area")  # , ("water pitcher", "water")

    def _add_custom_entities(self) -> None:
        """Add mug, wooden cube, metal cube, water pitcher, water, and target area."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Mug (mesh)
        mug = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/94134b02-c73e-4f2c-ad1d-a00a78160d98/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.48, 0.0, 0.76 + 0.0477),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=1200.0),
            surface=gs.surfaces.Smooth(color=(0.95, 0.8, 0.6), double_sided=True),
        )
        self._entities["mug"] = {
            "entity": mug,
        }

        # Wooden cube (box)
        wooden_cube = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.48, 0.0, 0.8125),
                euler=(0.0, 0.0, 0.0),
                size=(0.02, 0.02, 0.02),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(1.0, 0.9, 0.1), roughness=0.6),
        )
        self._entities["wooden cube"] = {
            "entity": wooden_cube,
        }

        # Metal cube (box)
        metal_cube = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.48, 0.0, 0.8025),
                euler=(0.0, 0.0, 0.0),
                size=(0.02, 0.02, 0.02),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=800.0),
            surface=gs.surfaces.Metal(color=(0.9, 0.1, 0.1), double_sided=False, metal_type="iron"),
        )
        self._entities["metal cube"] = {
            "entity": metal_cube,
        }

        # Water pitcher (mesh)
        pitcher = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path(
                    "hf_assets/simplify_red_cup.obj",
                    pattern_is_dir=False,
                ),
                scale=1.5,
                pos=(0.68, -0.10, 0.854),
                euler=(90.0, 0.0, 0.0),
                fixed=False,
                collision=True,
                convexify=False,
            ),
            material=gs.materials.Rigid(rho=500.0, sdf_max_res=256),
            surface=gs.surfaces.Glass(color=(0.4, 0.4, 0.4), double_sided=True),
        )
        self._entities["water pitcher"] = {
            "entity": pitcher,
        }

        # Water (SPH liquid)
        water = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.68, -0.1, 1.105),
                euler=(0.0, 0.0, 0.0),
                radius=0.035,
                height=0.4,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.SPH.Liquid(
                rho=500.0, stiffness=8e5, exponent=7.0, mu=0.005, gamma=0.01, sampler="pbs"
            ),
            surface=gs.surfaces.Default(color=(0.55, 0.75, 1.0), double_sided=True, vis_mode="recon"),
        )
        self._entities["water"] = {
            "entity": water,
        }

        # Target area (box, fixed, no collision)
        target_area = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.70, 0.35, 0.76 + 0.001),
                euler=(0.0, 0.0, 0.0),
                size=(0.12, 0.12, 0.002),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.1, 0.8, 0.2), roughness=0.9),
        )
        self._entities["target area"] = {
            "entity": target_area,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: wooden cube on target, metal cube in mug."""
        objs_info = self.collect_objs_info()

        TABLE_X_MIN, TABLE_X_MAX = 0.21, 1.00
        TABLE_Y_MIN, TABLE_Y_MAX = -0.69, 0.69

        def get_hull_xy(bounds, hull_2d):
            if hull_2d is not None and len(hull_2d) >= 3:
                return np.array([(float(p[0]), float(p[1])) for p in hull_2d])
            if bounds is None:
                return None
            xmin, ymin = float(bounds[0][0]), float(bounds[0][1])
            xmax, ymax = float(bounds[1][0]), float(bounds[1][1])
            return np.array([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])

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

        wood = objs_info.get("wooden cube")
        metal = objs_info.get("metal cube")
        mug = objs_info.get("mug")
        target = objs_info.get("target area")

        if wood is None or metal is None or mug is None or target is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        wood_hull = get_hull_xy(wood.get("bounds"), wood.get("convex_hull_2d"))
        metal_hull = get_hull_xy(metal.get("bounds"), metal.get("convex_hull_2d"))
        mug_hull = get_hull_xy(mug.get("bounds"), mug.get("convex_hull_2d"))
        target_hull = get_hull_xy(target.get("bounds"), target.get("convex_hull_2d"))

        if wood_hull is None or not is_within_table_bounds(wood_hull):
            return torch.tensor([False], dtype=torch.bool, device=self._device)
        if metal_hull is None or not is_within_table_bounds(metal_hull):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        if target_hull is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        overlap = compute_overlap_ratio(wood_hull, target_hull)
        wood_on_target_ok = overlap >= 0.50

        if mug_hull is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        metal_in_xy = compute_overlap_ratio(metal_hull, mug_hull) >= 0.50
        wood_in_xy = compute_overlap_ratio(wood_hull, mug_hull) >= 0.50

        metal_bounds = metal.get("bounds")
        mug_bounds = mug.get("bounds")
        wood_bounds = wood.get("bounds")
        if metal_bounds is None or mug_bounds is None or wood_bounds is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        m_zmin, m_zmax = metal_bounds[0][2], metal_bounds[1][2]
        w_zmin, w_zmax = wood_bounds[0][2], wood_bounds[1][2]
        g_zmin, g_zmax = mug_bounds[0][2], mug_bounds[1][2]
        lower_margin = 0.005
        upper_margin = 0.005
        metal_in_z = (m_zmin >= g_zmin + lower_margin) and (m_zmax <= g_zmax - upper_margin)
        wood_in_z = (w_zmin >= g_zmin + lower_margin) and (w_zmax <= g_zmax - upper_margin)

        metal_in_mug_ok = metal_in_xy and metal_in_z
        wood_out_of_mug_ok = not (wood_in_xy and wood_in_z)

        return torch.tensor(
            [wood_on_target_ok and metal_in_mug_ok and wood_out_of_mug_ok], dtype=torch.bool, device=self._device
        )

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute reward: 1.0 on success, else milestone progress in [0, 1)."""
        success = self._check_success()
        if success.item():
            reward = torch.ones(1, dtype=torch.float32, device=self._device)
            return reward, {"success": reward}

        objs_info = self.collect_objs_info()
        wood = objs_info.get("wooden cube")
        metal = objs_info.get("metal cube")
        mug = objs_info.get("mug")
        target = objs_info.get("target area")

        zero = torch.zeros(1, dtype=torch.float32, device=self._device)
        if wood is None or metal is None or mug is None or target is None:
            return zero, {"success": zero}

        def get_hull_xy(obj):
            hull = obj.get("convex_hull_2d")
            if hull is not None and len(hull) >= 3:
                return np.array([(float(p[0]), float(p[1])) for p in hull])
            b = obj.get("bounds")
            if b is None:
                return None
            xmin, ymin = float(b[0][0]), float(b[0][1])
            xmax, ymax = float(b[1][0]), float(b[1][1])
            return np.array([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])

        # EE proximity to either cube (stage 1: 0 → 0.2)
        APPROACH, REACH, TOUCH = 0.3, 0.15, 0.05
        right_ee = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee = self._robot.left_ee_pose[0, :3].cpu().numpy()
        ee_score = 0.0
        for obj in [wood, metal]:
            p = obj.get("pos")
            if p is None:
                continue
            ref = np.array(p[:3], dtype=float)
            d = min(float(np.linalg.norm(right_ee - ref)), float(np.linalg.norm(left_ee - ref)))
            if d <= TOUCH:
                s = 0.2
            elif d <= REACH:
                s = 0.1 + 0.1 * (REACH - d) / (REACH - TOUCH)
            elif d < APPROACH:
                s = 0.1 * (APPROACH - d) / (APPROACH - REACH)
            else:
                s = 0.0
            ee_score = max(ee_score, s)

        # Wooden cube XY distance to target center (stage 2: 0.2 → 0.6)
        wood_pos = wood.get("pos")
        target_pos = target.get("pos")
        dist_score = 0.0
        if wood_pos is not None and target_pos is not None:
            dist = float(np.linalg.norm(np.array(wood_pos[:2]) - np.array(target_pos[:2])))
            dist_frac = max(0.0, min(1.0, 1.0 - dist / 0.4))
            dist_score = 0.2 + 0.4 * dist_frac

        # Wooden cube overlap with target (stage 3: 0.6 → 0.8)
        wood_hull = get_hull_xy(wood)
        target_hull = get_hull_xy(target)
        overlap_score = 0.0
        wood_overlap = 0.0
        if wood_hull is not None and target_hull is not None:

            def polygon_area(poly):
                n = len(poly)
                area = 0.0
                for i in range(n):
                    j = (i + 1) % n
                    area += poly[i][0] * poly[j][1]
                    area -= poly[j][0] * poly[i][1]
                return abs(area) / 2.0

            inter = _utils.intersection_area(wood_hull, target_hull)
            w_area = polygon_area(wood_hull)
            wood_overlap = inter / w_area if w_area > 1e-9 else 0.0
            if wood_overlap > 0:
                overlap_score = 0.6 + 0.2 * min(1.0, wood_overlap / 0.50)

        # Metal cube in mug (stage 4: 0.8 → 1.0), gated on wooden cube on target
        metal_score = 0.0
        if wood_overlap >= 0.50:
            mug_hull = get_hull_xy(mug)
            metal_hull = get_hull_xy(metal)
            if mug_hull is not None and metal_hull is not None:
                mug_c = np.mean(mug_hull, axis=0)
                metal_c = np.mean(metal_hull, axis=0)
                metal_dist = float(np.linalg.norm(metal_c - mug_c))
                metal_frac = max(0.0, min(1.0, 1.0 - metal_dist / 0.1))
                metal_score = 0.8 + 0.2 * metal_frac

        score = max(ee_score, dist_score, overlap_score, metal_score)
        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"success": zero}
