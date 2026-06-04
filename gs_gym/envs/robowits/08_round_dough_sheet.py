"""RoundDoughSheet environment for RoboWits.

Task: Flatten a dough ball and make a perfect round sheet.

Note: This task uses MPM (Material Point Method) for soft body simulation
of the dough. It may not be fully supported in all Genesis versions.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv


@register_task("robowits/08-round-dough-sheet-v0")
class RoundDoughSheetEnv(RoboWitsEnv):
    """Flatten a dough ball and make a perfect round sheet.

    The task requires creating a precise round shape. The robot uses the
    planar surface of the board to distribute pressure and the circular
    symmetry of the round cutter to achieve the shape.

    Note: This environment uses MPM (Material Point Method) for the dough
    simulation, which is a particle-based soft body simulation.

    Success criteria:
    - At least 95% of particles are within table bounds
    - Dough height (z_95 - z_5) ≤ 0.035m (must be genuinely flattened, not just gravity-settled ~4.4cm)
    - At least 70% of particles within the target cylinder (radius ~0.053m, height 0.03m)
    - XY convex hull circularity ≥ 0.85 (4π·Area/Perimeter² == 1 for perfect circle)
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Flattern the dough ball and make a perfect round sheet"

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
    def placement_groups(self) -> PlacementGroups | None:
        """No random placement for this task."""
        return ("large flat board", "round cutter", "dough ball")

    def _add_custom_entities(self) -> None:
        """Add dough ball, flat board, and round cutter."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Dough ball (MPM soft body)
        # Note: This uses MPM.ElastoPlastic material for deformable simulation
        dough = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.37, -0.06, 0.812),
                euler=(0, 0, 0),
                size=(0.08, 0.08, 0.08),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.MPM.ElastoPlastic(
                E=1e4,
                nu=0.25,
                rho=200.0,
                sampler="pbs",
                von_mises_yield_stress=100.0,
            ),
            surface=gs.surfaces.Default(color=(0.95, 0.85, 0.65), vis_mode="recon"),
        )
        self._entities["dough ball"] = {
            "entity": dough,
        }

        # Large flat board (mesh)
        board = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/e8afda3b-6dea-4bfc-859f-88a35bb623a0/obj.glb", pattern_is_dir=False),
                scale=0.8,
                pos=(0.6, -0.05, 0.768),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(
                rho=500.0,
                friction=0.8,
                coup_friction=0.3,
                coup_softness=0.01,
                coup_restitution=0.0,
            ),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["large flat board"] = {
            "entity": board,
        }

        # Round cutter (mesh)
        cutter = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/662e6635-9e9f-4aed-991a-760c63592eb3/obj.glb", pattern_is_dir=False),
                scale=2.0,
                pos=(0.57, 0.165, 0.7932),
                euler=(180, 0, 90),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(
                rho=500.0,
                friction=0.4,
                coup_friction=0.3,
                coup_softness=0.01,
                coup_restitution=0.0,
            ),
            surface=gs.surfaces.Smooth(color=(0.72, 0.45, 0.20), double_sided=True),
        )
        self._entities["round cutter"] = {
            "entity": cutter,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: dough forms a flat round shape."""
        objs_info = self.collect_objs_info()

        TABLE_X_MIN, TABLE_X_MAX = 0.21, 1.00
        TABLE_Y_MIN, TABLE_Y_MAX = -0.69, 0.69

        dough_info = objs_info.get("dough ball")

        if dough_info is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        if dough_info.get("material") != "particle":
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        pos = dough_info.get("pos")
        if pos is None or len(pos) == 0:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        pos = np.asarray(pos)
        if pos.size == 0 or pos.ndim != 2 or pos.shape[1] != 3:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

        eps_xy = 0.01
        inside_xy = (
            (x >= TABLE_X_MIN - eps_xy)
            & (x <= TABLE_X_MAX + eps_xy)
            & (y >= TABLE_Y_MIN - eps_xy)
            & (y <= TABLE_Y_MAX + eps_xy)
        )
        above_floor = z >= (self.TABLE_Z - 0.06)
        safe_ratio = np.mean(inside_xy & above_floor)

        if safe_ratio < 0.95:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        pts_xy = pos[:, :2]
        z_base = float(np.percentile(z, 5))
        z_top = z_base + 0.03 + 0.002

        center_xy = pts_xy.mean(axis=0)
        radial = np.linalg.norm(pts_xy - center_xy, axis=1)
        sphere_radius = 0.04
        target_height = 0.03
        R = np.sqrt((4.0 / 3.0) * sphere_radius**3 / target_height)  # volume conservation

        if R <= 0.0:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        # Dough must actually be flat: 95th-percentile height ≤ target + 5mm tolerance.
        # Without this, a sphere that merely settles under gravity (~5cm tall) passes the
        # frac_in check because 78% of its particles happen to sit in the bottom 3cm window.
        z_height = float(np.percentile(z, 95)) - z_base
        if z_height > target_height + 0.005:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        in_cyl = (radial <= R) & (z >= (z_base - 0.002)) & (z <= z_top)
        frac_in = np.mean(in_cyl)
        if frac_in < 0.70:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        # Check the XY outline is roughly circular via convex hull circularity.
        # circularity = 4π·Area / Perimeter² == 1 for a perfect circle, < 1 for elongated shapes.
        from scipy.spatial import ConvexHull

        try:
            hull = ConvexHull(pts_xy)
            circularity = 4.0 * np.pi * hull.volume / (hull.area**2)
        except Exception:
            circularity = 0.0

        return torch.tensor([circularity >= 0.85], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute reward: 1.0 on success, else flatness × roundness progress in [0, 1)."""
        success = self._check_success()
        if success.item():
            reward = torch.ones(1, dtype=torch.float32, device=self._device)
            return reward, {"success": reward}

        objs_info = self.collect_objs_info()
        dough_info = objs_info.get("dough ball")

        zero = torch.zeros(1, dtype=torch.float32, device=self._device)
        if dough_info is None or dough_info.get("material") != "particle":
            return zero, {"success": zero}

        pos = np.asarray(dough_info.get("pos", []))
        if pos.size == 0 or pos.ndim != 2 or pos.shape[1] != 3:
            return zero, {"success": zero}

        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

        TABLE_X_MIN, TABLE_X_MAX = 0.21, 1.00
        TABLE_Y_MIN, TABLE_Y_MAX = -0.69, 0.69
        eps_xy = 0.01
        inside_xy = (
            (x >= TABLE_X_MIN - eps_xy)
            & (x <= TABLE_X_MAX + eps_xy)
            & (y >= TABLE_Y_MIN - eps_xy)
            & (y <= TABLE_Y_MAX + eps_xy)
        )
        safe_ratio = float(np.mean(inside_xy & (z >= self.TABLE_Z - 0.06)))

        sphere_radius = 0.04
        target_height = 0.03
        initial_height = 2.0 * sphere_radius  # sphere diameter

        z_base = float(np.percentile(z, 5))
        current_height = float(np.percentile(z, 95)) - z_base
        height_progress = max(0.0, min(1.0, (initial_height - current_height) / (initial_height - target_height)))

        R = np.sqrt((4.0 / 3.0) * sphere_radius**3 / target_height)
        z_top = z_base + target_height + 0.002
        pts_xy = pos[:, :2]
        center_xy = pts_xy.mean(axis=0)
        radial = np.linalg.norm(pts_xy - center_xy, axis=1)
        in_cyl = (radial <= R) & (z >= z_base - 0.002) & (z <= z_top)
        frac_in = float(np.mean(in_cyl))
        # Roundness bonus: scales from 0.7 (no roundness credit) to 1.0 (fully round)
        roundness = 0.7 + 0.3 * min(1.0, frac_in / 0.70)

        score = safe_ratio * height_progress * roundness
        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"success": zero}
