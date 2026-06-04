from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

RoundDoughSheetEnv = _import("gs_gym.envs.robowits.08_round_dough_sheet").RoundDoughSheetEnv


@register_task("robowits/08_03-v0")
class RoundDoughSheetMut3Env(RoundDoughSheetEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("dough ball", "large flat board", "round cutter", "pot lid")

    def _config_to_env_args(self, config):
        args = super()._config_to_env_args(config)
        return args.model_copy(update={"scene_args": args.scene_args.model_copy(update={"substeps": 100})})

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: dough ball
        _e = self._scene.scene.add_entity(
            gs.morphs.Sphere(pos=(0.32, -0.15, 0.812), radius=0.04, fixed=False, collision=True),
            material=gs.materials.MPM.ElastoPlastic(
                E=2e5, nu=0.3, rho=800.0, sampler="pbs", von_mises_yield_stress=500.0
            ),
            surface=gs.surfaces.Default(color=(0.95, 0.85, 0.65), vis_mode="recon"),
        )
        self._entities["dough ball"] = {"entity": _e}
        # Code Block: large flat board
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/e8afda3b-6dea-4bfc-859f-88a35bb623a0/obj.glb", pattern_is_dir=False),
                scale=0.75,
                pos=(0.51, 0.00, 0.770),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(
                rho=400.0, friction=1.0, coup_friction=0.3, coup_softness=0.01, coup_restitution=0.0
            ),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["large flat board"] = {"entity": _e}
        # Code Block: round cutter
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/662e6635-9e9f-4aed-991a-760c63592eb3/obj.glb", pattern_is_dir=False),
                scale=1.1,
                pos=(0.37, 0.15, 0.76 + 0.0329 * 1.1),
                euler=(180, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(
                rho=500.0, friction=0.8, coup_friction=0.3, coup_softness=0.01, coup_restitution=0.0
            ),
            surface=gs.surfaces.Smooth(color=(0.72, 0.45, 0.20), double_sided=True),
        )
        self._entities["round cutter"] = {"entity": _e}
        # Code Block: pot lid
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/426c2c95-4628-4418-816f-3e31ce5b8243/obj.glb", pattern_is_dir=False),
                scale=0.6,
                pos=(0.68, 0.18, 0.8238),
                euler=(180, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500.0, friction=0.8),
            surface=gs.surfaces.Reflective(double_sided=True),
        )
        self._entities["pot lid"] = {"entity": _e}
