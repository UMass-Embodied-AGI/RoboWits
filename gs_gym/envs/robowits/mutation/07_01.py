from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

StandPagesEnv = _import("gs_gym.envs.robowits.07_stand_pages").StandPagesEnv


@register_task("robowits/07_01-v0")
class StandPagesMut1Env(StandPagesEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("page A with eyelets", "page B with eyelets", "stabilizing bar", "chocolate bar", "glass mug")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: page A with eyelets
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/pageA.glb", pattern_is_dir=False),
                scale=0.6,
                pos=(0.40, -0.06, 0.796),
                euler=(0.0, 90.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=300.0, friction=1.0, coup_friction=0.2),
            surface=gs.surfaces.Smooth(color=(0.7, 0.7, 0.7), double_sided=True),
        )
        self._entities["page A with eyelets"] = {"entity": _e}
        # Code Block: page B with eyelets
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/pageB.glb", pattern_is_dir=False),
                scale=0.6,
                pos=(0.62, 0.06, 0.796),
                euler=(0.0, 90.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=300.0, friction=1.0, coup_friction=0.2),
            surface=gs.surfaces.Smooth(color=(0.7, 0.7, 0.7), double_sided=True),
        )
        self._entities["page B with eyelets"] = {"entity": _e}
        # Code Block: stabilizing bar
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.49, -0.446, 0.764),
                euler=(0.0, 90.0, 0.0),
                radius=0.0105,
                height=0.26,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=800.0, friction=0.8, coup_friction=0.2),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.2, ior=1.5),
        )
        self._entities["stabilizing bar"] = {"entity": _e}
        # Code Block: chocolate bar
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/4b504df0-2e47-4db7-83f7-e8e045dfcbd7/obj.glb", pattern_is_dir=False),
                scale=0.7,
                pos=(0.70, -0.30, 0.7658),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500.0, friction=0.9, coup_friction=0.2),
            surface=gs.surfaces.Rough(color=(0.4, 0.2, 0.1), double_sided=True),
        )
        self._entities["chocolate bar"] = {"entity": _e}
        # Code Block: glass mug
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/ddfb928e-c26d-4805-b41a-ab1545269e99/obj.glb", pattern_is_dir=False),
                scale=0.3755,
                pos=(0.70, -0.43, 0.81),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=2500.0, friction=0.8, coup_friction=0.2),
            surface=gs.surfaces.Smooth(color=(0.9, 0.9, 0.95), double_sided=True),
        )
        self._entities["glass mug"] = {"entity": _e}
