from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

RoundDoughSheetEnv = _import("gs_gym.envs.robowits.08_round_dough_sheet").RoundDoughSheetEnv


@register_task("robowits/08_06-v0")
class RoundDoughSheetMut6Env(RoundDoughSheetEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("dough ball", "large flat board", "round cutter", "paper cup")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: dough ball
        _e = self._scene.scene.add_entity(
            gs.morphs.Sphere(pos=(0.36, 0.00, 0.812), euler=(0, 0, 0), radius=0.04, fixed=False, collision=True),
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
                scale=0.7,
                pos=(0.58, -0.08, 0.771),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(
                rho=400.0, friction=0.6, coup_friction=0.3, coup_softness=0.01, coup_restitution=0.0
            ),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["large flat board"] = {"entity": _e}
        # Code Block: round cutter
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/662e6635-9e9f-4aed-991a-760c63592eb3/obj.glb", pattern_is_dir=False),
                scale=1.11,
                pos=(0.66, 0.12, 0.799),
                euler=(180, 0, 30),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(
                rho=600.0, friction=0.5, coup_friction=0.3, coup_softness=0.01, coup_restitution=0.0
            ),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["round cutter"] = {"entity": _e}
        # Code Block: paper cup
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/0734b65a-09ef-4793-becc-9f462c558f79/obj.glb", pattern_is_dir=False),
                scale=1.389,
                pos=(0.70, 0.22, 0.8112),
                euler=(180, 0, 0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.6, coup_friction=0.2),
            surface=gs.surfaces.Default(color=(0.85, 0.40, 0.10), roughness=0.6),
        )
        self._entities["paper cup"] = {"entity": _e}
