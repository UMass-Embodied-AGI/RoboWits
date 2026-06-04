"""SeparateMarblesAndSand environment for RoboWits.

Task: Remove sand from the jar and keep the marbles in the jar.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv
from gs_gym.scenes.schema import MPMOptionsArgs


@register_task("robowits/15-separate-marbles-and-sand-v0")
class SeparateMarblesAndSandEnv(RoboWitsEnv):
    """Separate marbles from sand using a colander over a bowl.

    Individual grains are too small to pick efficiently. By assembling a
    colander over a bowl, the robot creates a gravity-powered separator
    that splits marbles from sand and enables clean transfer into the jar.

    Success criteria:
    - All five marbles are inside the jar's 3D AABB
    - At least 80% of sand particles are outside the jar's 3D AABB
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Remove sand from the jar and keep the marbles in the jar."

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
        args = args.model_copy(
            update={
                "scene_args": args.scene_args.model_copy(
                    update={
                        "substeps": 200,
                        "mpm_options": MPMOptionsArgs(
                            grid_density=64,
                            particle_size=0.002,
                            enable_CPIC=True,
                        ),
                    }
                )
            }
        )
        return args

    @property
    def placement_groups(self) -> PlacementGroups:
        """Colander and bowl placed independently; jar, marbles, and sand move as one cluster (MPM sand follows)."""
        return ("colander", "bowl", ("jar", "marble 1", "marble 2", "marble 3", "marble 4", "marble 5", "sand"))

    def _add_custom_entities(self) -> None:
        """Add colander, bowl, jar, marbles, and sand."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Colander (mesh)
        colander = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/633e9459-f7ed-4507-ad97-ce2783b06a02/obj.glb", pattern_is_dir=False),
                scale=0.864,
                pos=(0.68, -0.12, 0.8038),
                euler=(180.0, 0.0, 0.0),
                fixed=False,
                collision=True,
                # convexify=False
            ),
            material=gs.materials.Rigid(rho=300.0, friction=1.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["colander"] = {
            "entity": colander,
        }

        # Bowl (mesh)
        bowl = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c652cf0f-d2eb-44bd-9e68-a2ceca698591/obj.glb", pattern_is_dir=False),
                scale=1.38,
                pos=(0.50, -0.12, 0.808576),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=300.0, friction=1.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["bowl"] = {
            "entity": bowl,
        }

        jar = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path(
                    "hf_assets/simplify_red_cup.obj",
                    pattern_is_dir=False,
                ),
                scale=1,
                pos=(0.68, 0.07, 0.855),
                euler=(90.0, 0.0, 0.0),
                fixed=True,
                collision=True,
                convexify=False,
            ),
            material=gs.materials.Rigid(rho=300.0, friction=0.9, sdf_max_res=256),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["jar"] = {
            "entity": jar,
        }

        # Marbles (5 mesh objects)
        marble_positions = [
            (0.68, 0.07, 0.79),
            (0.68, 0.07, 0.815),
            (0.68, 0.07, 0.84),
            (0.68, 0.07, 0.865),
            (0.68, 0.07, 0.89),
        ]
        for i, pos in enumerate(marble_positions, start=1):
            marble = self._scene.scene.add_entity(
                gs.morphs.Mesh(
                    coacd_options=coacd_options,
                    file=get_asset_path(
                        "blender_kit/5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb", pattern_is_dir=False
                    ),
                    scale=0.12,
                    pos=pos,
                    euler=(0.0, 0.0, 0.0),
                    fixed=False,
                    collision=True,
                ),
                material=gs.materials.Rigid(rho=1200.0, friction=0.6),
                surface=gs.surfaces.Smooth(double_sided=True),
            )
            self._entities[f"marble {i}"] = {
                "entity": marble,
            }

        # Sand (MPM particle)
        sand = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.69, 0.08, 1.19),
                euler=(0.0, 0.0, 0.0),
                radius=0.02,
                height=0.2,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.MPM.Sand(rho=800.0, sampler="pbs", friction_angle=30),
            surface=gs.surfaces.Default(color=(0.9, 0.8, 0.3), vis_mode="particle", double_sided=True),
        )
        self._entities["sand"] = {
            "entity": sand,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: marbles in jar, sand outside jar."""
        objs_info = self.collect_objs_info()

        TABLE_X_MIN, TABLE_X_MAX = 0.21, 1.00
        TABLE_Y_MIN, TABLE_Y_MAX = -0.69, 0.69

        jar = objs_info.get("jar")
        sand = objs_info.get("sand")
        marble_names = [f"marble {i}" for i in range(1, 6)]
        marbles = [objs_info.get(n) for n in marble_names]

        if jar is None or any(m is None for m in marbles) or sand is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        jar_bounds = jar.get("bounds")
        if jar_bounds is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        all_marbles_in_jar = True
        lo, hi = jar_bounds[0], jar_bounds[1]
        for m in marbles:
            mpos = m.get("pos")
            if mpos is None:
                all_marbles_in_jar = False
                break
            if not (
                (mpos[0] >= lo[0] - 1e-4)
                and (mpos[0] <= hi[0] + 1e-4)
                and (mpos[1] >= lo[1] - 1e-4)
                and (mpos[1] <= hi[1] + 1e-4)
                and (mpos[2] >= lo[2] - 1e-4)
                and (mpos[2] <= hi[2] + 1e-4)
            ):
                all_marbles_in_jar = False
                break

        if not all_marbles_in_jar:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        sand_pos = sand.get("pos")
        if sand_pos is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        sand_pos = np.asarray(sand_pos)
        if sand_pos.ndim != 2 or sand_pos.shape[1] != 3:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        if sand_pos.shape[0] == 0:
            frac_sand_outside_jar = 1.0
        else:
            inside_x = (sand_pos[:, 0] >= lo[0] - 1e-4) & (sand_pos[:, 0] <= hi[0] + 1e-4)
            inside_y = (sand_pos[:, 1] >= lo[1] - 1e-4) & (sand_pos[:, 1] <= hi[1] + 1e-4)
            inside_z = (sand_pos[:, 2] >= lo[2] - 1e-4) & (sand_pos[:, 2] <= hi[2] + 1e-4)
            sand_inside_mask = inside_x & inside_y & inside_z
            frac_sand_outside_jar = 1.0 - np.mean(sand_inside_mask.astype(float))

        sand_condition = frac_sand_outside_jar >= 0.8 - 1e-6

        def rigid_on_table(obj):
            hull = obj.get("convex_hull_2d")
            if hull is not None and isinstance(hull, np.ndarray) and hull.size > 0:
                xin = (hull[:, 0] >= TABLE_X_MIN - 1e-3) & (hull[:, 0] <= TABLE_X_MAX + 1e-3)
                yin = (hull[:, 1] >= TABLE_Y_MIN - 1e-3) & (hull[:, 1] <= TABLE_Y_MAX + 1e-3)
                return np.all(xin & yin)
            pos = obj.get("pos")
            if pos is None:
                return False
            return (TABLE_X_MIN - 1e-3 <= pos[0] <= TABLE_X_MAX + 1e-3) and (
                TABLE_Y_MIN - 1e-3 <= pos[1] <= TABLE_Y_MAX + 1e-3
            )

        jar_on_table = rigid_on_table(jar)
        marbles_on_table = all(rigid_on_table(m) for m in marbles)

        if sand_pos.shape[0] == 0:
            frac_sand_on_table = 1.0
        else:
            xin = (sand_pos[:, 0] >= TABLE_X_MIN - 1e-3) & (sand_pos[:, 0] <= TABLE_X_MAX + 1e-3)
            yin = (sand_pos[:, 1] >= TABLE_Y_MIN - 1e-3) & (sand_pos[:, 1] <= TABLE_Y_MAX + 1e-3)
            frac_sand_on_table = np.mean(xin & yin)
        sand_on_table = frac_sand_on_table >= 0.5

        result = (
            bool(all_marbles_in_jar)
            and bool(sand_condition)
            and bool(jar_on_table)
            and bool(marbles_on_table)
            and bool(sand_on_table)
        )
        return torch.tensor([result], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute reward: 1.0 on success, else sand-removal × marble-retention progress."""
        success = self._check_success()
        if success.item():
            reward = torch.ones(1, dtype=torch.float32, device=self._device)
            return reward, {"success": reward}

        objs_info = self.collect_objs_info()
        jar = objs_info.get("jar")
        sand = objs_info.get("sand")
        marble_names = [f"marble {i}" for i in range(1, 6)]

        zero = torch.zeros(1, dtype=torch.float32, device=self._device)
        if jar is None or sand is None:
            return zero, {"success": zero}

        jar_bounds = jar.get("bounds")
        if jar_bounds is None:
            return zero, {"success": zero}
        lo, hi = np.array(jar_bounds[0], dtype=float), np.array(jar_bounds[1], dtype=float)

        # EE proximity to jar (0 → 0.2)
        APPROACH, REACH, TOUCH = 0.3, 0.15, 0.05
        jar_pos = np.array(jar.get("pos", [0.0, 0.0, 0.0]), dtype=float)
        right_ee = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee = self._robot.left_ee_pose[0, :3].cpu().numpy()
        min_dist = min(float(np.linalg.norm(right_ee - jar_pos)), float(np.linalg.norm(left_ee - jar_pos)))
        if min_dist <= TOUCH:
            ee_score = 0.2
        elif min_dist <= REACH:
            ee_score = 0.1 + 0.1 * (REACH - min_dist) / (REACH - TOUCH)
        elif min_dist < APPROACH:
            ee_score = 0.1 * (APPROACH - min_dist) / (APPROACH - REACH)
        else:
            ee_score = 0.0

        # Fraction of sand outside jar
        sand_pos = sand.get("pos")
        if sand_pos is not None:
            sand_pos = np.asarray(sand_pos)
            if sand_pos.ndim == 2 and sand_pos.shape[1] == 3 and sand_pos.shape[0] > 0:
                inside = (
                    (sand_pos[:, 0] >= lo[0] - 1e-4)
                    & (sand_pos[:, 0] <= hi[0] + 1e-4)
                    & (sand_pos[:, 1] >= lo[1] - 1e-4)
                    & (sand_pos[:, 1] <= hi[1] + 1e-4)
                    & (sand_pos[:, 2] >= lo[2] - 1e-4)
                    & (sand_pos[:, 2] <= hi[2] + 1e-4)
                )
                frac_sand_outside = float(1.0 - np.mean(inside))
            else:
                frac_sand_outside = 1.0
        else:
            frac_sand_outside = 1.0

        # Fraction of marbles still inside jar
        marbles_in = 0
        n_marbles = len(marble_names)
        for name in marble_names:
            m = objs_info.get(name)
            if m is None:
                continue
            mpos = m.get("pos")
            if mpos is None:
                continue
            if (
                lo[0] - 1e-4 <= mpos[0] <= hi[0] + 1e-4
                and lo[1] - 1e-4 <= mpos[1] <= hi[1] + 1e-4
                and lo[2] - 1e-4 <= mpos[2] <= hi[2] + 1e-4
            ):
                marbles_in += 1
        frac_marbles_in = marbles_in / max(n_marbles, 1)

        # Sand removal: 0 → 0.8; marble retention bonus: up to 0.2
        removal_frac = min(1.0, frac_sand_outside / 0.8)
        task_score = 0.8 * removal_frac + 0.2 * frac_marbles_in * removal_frac

        score = max(ee_score, task_score)
        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"success": zero}
