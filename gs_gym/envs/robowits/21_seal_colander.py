"""SealColander environment for RoboWits.

Task: Make the perforated container hold water.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv


@register_task("robowits/21-seal-colander-v0")
class SealColanderEnv(RoboWitsEnv):
    """Make a perforated container hold water.

    The perforations of the perforated container prevent it from holding liquid.
    By placing the curved holder onto the perforated container, the robot can
    pour the water from the pitcher into the curved holder onto the perforated
    container and verify it holds.

    Success criteria:
    - Over 10% of water particles are inside the perforated container
    - Container hasn't fallen off the table
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Make the container hold water for a short time."

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
        args = args.model_copy(update={"scene_args": args.scene_args.model_copy(update={"substeps": 30})})
        return args

    @property
    def placement_groups(self) -> PlacementGroups:
        """Pitcher and water are fixed; only container and holder are randomized."""
        return ("perforated container", "curved holder")

    def _add_custom_entities(self) -> None:
        """Add perforated container, pitcher, curved holder, and water."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Perforated container (mesh, fixed)
        container = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/slotted_spoon.glb", pattern_is_dir=False),
                scale=0.8847,
                pos=(0.44, -0.06, 0.779),
                euler=(90.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.6),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["perforated container"] = {
            "entity": container,
        }

        # Pitcher (mesh, fixed)
        pitcher = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c242567e-052d-4561-b2c0-2fed8a5e576b/obj.glb", pattern_is_dir=False),
                scale=1.2476,
                pos=(0.61, 0.0, 0.85),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["pitcher"] = {
            "entity": pitcher,
        }
        # Invisible blockers to contain SPH water inside the pitcher.
        # Water cylinder: radius=0.045, center z=0.86. Blocker ring at radius=0.055.
        _px, _py = 0.61, 0.0
        self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(_px, _py, 0.80),
                radius=0.06,
                height=0.04,
                fixed=True,
                visualization=False,
                collision=True,
            ),
        )
        for rot in range(0, 360, 30):
            sr, cr = np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot))
            self._scene.scene.add_entity(
                gs.morphs.Box(
                    pos=(_px + 0.055 * cr, _py + 0.055 * sr, 0.875),
                    euler=(0.0, 0.0, float(rot)),
                    size=(0.01, 0.04, 0.15),
                    fixed=True,
                    visualization=False,
                    collision=True,
                )
            )

        # Curved holder (sphere, fixed)
        curved_holder = self._scene.scene.add_entity(
            gs.morphs.Sphere(
                radius=0.012,
                pos=(0.50, -0.15, 0.772),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.6),
            surface=gs.surfaces.Smooth(color=(0.85, 0.85, 0.85), double_sided=True),
        )
        self._entities["curved holder"] = {
            "entity": curved_holder,
        }

        # Water (SPH liquid)
        water = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.045,
                height=0.13,
                pos=(0.61, 0.0, 0.86),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.SPH.Liquid(
                rho=500.0, stiffness=80000.0, exponent=7.0, mu=0.005, gamma=0.01, sampler="pbs"
            ),
            surface=gs.surfaces.Default(color=(0.6, 0.7, 1.0), double_sided=True, vis_mode="recon"),
            # surface=gs.surfaces.Default(color=(0.6, 0.7, 1.0), double_sided=True, vis_mode="particle"),
        )
        self._entities["water"] = {
            "entity": water,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: water in perforated container."""
        objs_info = self.collect_objs_info()

        TABLE_X_MIN, TABLE_X_MAX = 0.21, 1.00
        TABLE_Y_MIN, TABLE_Y_MAX = -0.69, 0.69

        container = objs_info.get("perforated container")
        water = objs_info.get("water")

        if container is None or water is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        container_hull = container.get("convex_hull_2d")
        container_bounds = container.get("bounds")

        if container_hull is not None:
            container_poly = [(float(p[0]), float(p[1])) for p in container_hull]
        elif container_bounds is not None:
            xmin, ymin = container_bounds[0][0], container_bounds[0][1]
            xmax, ymax = container_bounds[1][0], container_bounds[1][1]
            container_poly = [
                (float(xmin), float(ymin)),
                (float(xmax), float(ymin)),
                (float(xmax), float(ymax)),
                (float(xmin), float(ymax)),
            ]
        else:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        def polygon_within_table(poly):
            for x, y in poly:
                if x < TABLE_X_MIN - 1e-3 or x > TABLE_X_MAX + 1e-3:
                    return False
                if y < TABLE_Y_MIN - 1e-3 or y > TABLE_Y_MAX + 1e-3:
                    return False
            return True

        if not polygon_within_table(container_poly):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        container_bounds_for_z = container.get("bounds")
        if container_bounds_for_z is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)
        z_min = float(container_bounds_for_z[0][2])
        z_max = float(container_bounds_for_z[1][2])

        if water.get("material") != "particle":
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        water_pos = water.get("pos")
        if water_pos is None or np.size(water_pos) == 0:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        water_pos = np.asarray(water_pos)

        def point_in_polygon(px, py, polygon):
            inside = False
            n = len(polygon)
            if n < 3:
                return False
            for i in range(n):
                x1, y1 = polygon[i]
                x2, y2 = polygon[(i + 1) % n]
                if (y1 > py) != (y2 > py):
                    denom = y2 - y1
                    if abs(denom) < 1e-12:
                        continue
                    x_intersect = x1 + (py - y1) * (x2 - x1) / denom
                    if x_intersect >= px:
                        inside = not inside
            return inside

        total = len(water_pos)
        if total <= 0:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        inside_count = 0
        z_lo = z_min + 0.005
        z_hi = z_max + 0.02

        for i in range(total):
            px = float(water_pos[i][0])
            py = float(water_pos[i][1])
            pz = float(water_pos[i][2])
            if pz < z_lo or pz > z_hi:
                continue
            if point_in_polygon(px, py, container_poly):
                inside_count += 1

        frac_inside = inside_count / float(total)
        return torch.tensor([frac_inside > 0.10], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute reward: 1.0 on success, else milestone progress in [0, 1)."""
        success = self._check_success()
        if success.item():
            reward = torch.ones(1, dtype=torch.float32, device=self._device)
            return reward, {"success": reward}

        objs_info = self.collect_objs_info()
        container = objs_info.get("perforated container")
        water = objs_info.get("water")
        pitcher = objs_info.get("pitcher")

        zero = torch.zeros(1, dtype=torch.float32, device=self._device)
        if container is None or water is None:
            return zero, {"success": zero}

        # EE proximity to pitcher or container (stage 1: 0 → 0.2)
        APPROACH, REACH, TOUCH = 0.3, 0.15, 0.05
        right_ee = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee = self._robot.left_ee_pose[0, :3].cpu().numpy()
        ee_score = 0.0
        for obj in [pitcher, container]:
            if obj is None:
                continue
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

        # Water fraction inside container (stage 2: 0.2 → 1.0)
        inside_score = 0.0
        container_hull = container.get("convex_hull_2d")
        container_bounds = container.get("bounds")
        if container_hull is not None:
            container_poly = [(float(p[0]), float(p[1])) for p in container_hull]
        elif container_bounds is not None:
            xmin, ymin = container_bounds[0][0], container_bounds[0][1]
            xmax, ymax = container_bounds[1][0], container_bounds[1][1]
            container_poly = [
                (float(xmin), float(ymin)),
                (float(xmax), float(ymin)),
                (float(xmax), float(ymax)),
                (float(xmin), float(ymax)),
            ]
        else:
            container_poly = None

        if container_poly is not None and container_bounds is not None and water.get("material") == "particle":
            water_pos = water.get("pos")
            if water_pos is not None and np.size(water_pos) > 0:
                water_pos = np.asarray(water_pos)
                z_lo = float(container_bounds[0][2]) + 0.005
                z_hi = float(container_bounds[1][2]) + 0.02
                total = len(water_pos)

                def point_in_polygon(px, py, polygon):
                    inside = False
                    n = len(polygon)
                    for i in range(n):
                        x1, y1 = polygon[i]
                        x2, y2 = polygon[(i + 1) % n]
                        if (y1 > py) != (y2 > py):
                            denom = y2 - y1
                            if abs(denom) < 1e-12:
                                continue
                            if x1 + (py - y1) * (x2 - x1) / denom >= px:
                                inside = not inside
                    return inside

                inside_count = sum(
                    1
                    for i in range(total)
                    if z_lo <= float(water_pos[i][2]) <= z_hi
                    and point_in_polygon(float(water_pos[i][0]), float(water_pos[i][1]), container_poly)
                )
                frac_inside = inside_count / float(total) if total > 0 else 0.0
                if frac_inside > 1e-3:
                    inside_score = 0.2 + 0.8 * min(1.0, frac_inside / 0.10)

        score = max(ee_score, inside_score)
        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"success": zero}
