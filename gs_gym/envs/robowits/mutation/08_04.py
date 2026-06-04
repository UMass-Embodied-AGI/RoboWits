from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

RoundDoughSheetEnv = _import("gs_gym.envs.robowits.08_round_dough_sheet").RoundDoughSheetEnv


@register_task("robowits/08_04-v0")
class RoundDoughSheetMut4Env(RoundDoughSheetEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("dough ball", "large hollow cylinder", "round pusher")

    def _config_to_env_args(self, config):
        args = super()._config_to_env_args(config)
        return args.model_copy(update={"scene_args": args.scene_args.model_copy(update={"substeps": 100})})

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: large hollow cylinder
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/662e6635-9e9f-4aed-991a-760c63592eb3/obj.glb", pattern_is_dir=False),
                scale=1.1,
                pos=(0.50, 0.00, 0.799),
                euler=(180.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(
                rho=200.0, friction=1.0, coup_friction=0.3, coup_softness=0.01, coup_restitution=0.0
            ),
            surface=gs.surfaces.Smooth(color=(0.72, 0.45, 0.20), double_sided=True),
        )
        self._entities["large hollow cylinder"] = {"entity": _e}
        # Code Block: dough ball
        _e = self._scene.scene.add_entity(
            gs.morphs.Sphere(pos=(0.50, 0.00, 0.812), euler=(0.0, 0.0, 0.0), radius=0.04, fixed=False, collision=True),
            material=gs.materials.MPM.ElastoPlastic(
                E=2e5, nu=0.3, rho=800.0, sampler="pbs", von_mises_yield_stress=500.0
            ),
            surface=gs.surfaces.Default(color=(0.95, 0.85, 0.65), vis_mode="recon"),
        )
        self._entities["dough ball"] = {"entity": _e}
        # Code Block: round pusher
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.66, 0.00, 0.830), euler=(0.0, 0.0, 0.0), radius=0.028, height=0.12, fixed=False, collision=True
            ),
            material=gs.materials.Rigid(
                rho=500.0, friction=1.0, coup_friction=0.3, coup_softness=0.01, coup_restitution=0.0
            ),
            surface=gs.surfaces.Default(color=(0.65, 0.45, 0.30), roughness=0.7, ior=1.5, double_sided=True),
        )
        self._entities["round pusher"] = {"entity": _e}
