from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

RoundDoughSheetEnv = _import("gs_gym.envs.robowits.08_round_dough_sheet").RoundDoughSheetEnv


@register_task("robowits/08_02-v0")
class RoundDoughSheetMut2Env(RoundDoughSheetEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("dough ball", "large flat board", "round cutter", "small chocolate cake", "medic hammer")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: large flat board
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/e8afda3b-6dea-4bfc-859f-88a35bb623a0/obj.glb", pattern_is_dir=False),
                scale=0.8,
                pos=(0.56, 0.06, 0.768),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(
                rho=500.0, friction=1.0, coup_friction=0.3, coup_softness=0.01, coup_restitution=0.0
            ),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["large flat board"] = {"entity": _e}
        # Code Block: round cutter
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/662e6635-9e9f-4aed-991a-760c63592eb3/obj.glb", pattern_is_dir=False),
                scale=0.9,
                pos=(0.675, 0.28, 0.7915),
                euler=(180.0, 0.0, 0.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(
                rho=650.0, friction=0.4, coup_friction=0.3, coup_softness=0.01, coup_restitution=0.0
            ),
            surface=gs.surfaces.Smooth(color=(0.72, 0.45, 0.20), double_sided=True),
        )
        self._entities["round cutter"] = {"entity": _e}
        # Code Block: small chocolate cake
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/f8fe61ad-1f17-4ac1-87aa-bb268c05557c/obj.glb", pattern_is_dir=False),
                scale=0.9,
                pos=(0.665, -0.18, 0.7948),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=False,
            ),
            material=gs.materials.Rigid(
                rho=450.0, friction=0.6, coup_friction=0.15, coup_softness=0.003, coup_restitution=0.0
            ),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["small chocolate cake"] = {"entity": _e}
        # Code Block: medic hammer
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/5ea7cb84-9853-4887-b15e-24522658466f/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.668, -0.39, 0.778),
                euler=(0.0, 0.0, 90.0),
                fixed=True,
                collision=False,
            ),
            material=gs.materials.Rigid(
                rho=650.0, friction=0.6, coup_friction=0.15, coup_softness=0.003, coup_restitution=0.0
            ),
            surface=gs.surfaces.Smooth(color=(0.55, 0.27, 0.07), double_sided=True),
        )
        self._entities["medic hammer"] = {"entity": _e}
        # Code Block: dough ball
        _e = self._scene.scene.add_entity(
            gs.morphs.Sphere(pos=(0.44, -0.12, 0.823), radius=0.04, fixed=False, collision=True),
            material=gs.materials.MPM.ElastoPlastic(
                E=2e5, nu=0.3, rho=800.0, sampler="pbs", von_mises_yield_stress=500.0
            ),
            surface=gs.surfaces.Default(color=(0.95, 0.85, 0.65), vis_mode="recon"),
        )
        self._entities["dough ball"] = {"entity": _e}
