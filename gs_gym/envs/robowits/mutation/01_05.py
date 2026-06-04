from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

AlignBlocksEnv = _import("gs_gym.envs.robowits.01_align_blocks").AlignBlocksEnv


@register_task("robowits/01_05-v0")
class AlignBlocksMut5Env(AlignBlocksEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("first target cube", "second target cube", "third target cube", "long rigid ruler", "kitchen sponge")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: first target cube
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.70, -0.14, 0.785), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.6, 0.7, 0.9), roughness=0.5),
        )
        self._entities["first target cube"] = {"entity": _e}
        # Code Block: second target cube
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.69, 0.12, 0.785), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.6, 0.7, 0.9), roughness=0.5),
        )
        self._entities["second target cube"] = {"entity": _e}
        # Code Block: third target cube
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.66, -0.01, 0.785), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.6, 0.7, 0.9), roughness=0.5),
        )
        self._entities["third target cube"] = {"entity": _e}
        # Code Block: long rigid ruler
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/33679be6-fc3f-40e0-ae2b-c6329d2d0ac8/obj.glb", pattern_is_dir=False),
                scale=1.2,
                pos=(0.61, 0.0, 0.76 + 0.0097 * 1.2),
                euler=(0, 0, 90),
                fixed=False,
                collision=True,
                group_by_material=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["long rigid ruler"] = {"entity": _e}
        # Code Block: kitchen sponge
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/a939eb10-a16c-4e6e-b766-0805109143d4/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.69, -0.38, 0.7800),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=300.0, friction=0.9),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["kitchen sponge"] = {"entity": _e}
