from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

StackCubesEnv = _import("gs_gym.envs.robowits.14_stack_cubes").StackCubesEnv


@register_task("robowits/14_01-v0")
class StackCubesMut1Env(StackCubesEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("apex target", "base cube 1", "base cube 2", "apex cube", "bluetooth keyboard", "tv remote control")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: apex target
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.605, 0.0, 0.7605), euler=(0, 0, 0), size=(0.03, 0.03, 0.001), fixed=True, collision=False
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.1, 0.8, 0.2), roughness=0.4),
        )
        self._entities["apex target"] = {"entity": _e}
        # Code Block: base cube 1
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(pos=(0.45, -0.05, 0.7825), euler=(0, 0, 0), size=(0.045, 0.045, 0.045)),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5),
        )
        self._entities["base cube 1"] = {"entity": _e}
        # Code Block: base cube 2
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(pos=(0.68, 0.12, 0.7825), euler=(0, 0, 0), size=(0.045, 0.045, 0.045)),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5),
        )
        self._entities["base cube 2"] = {"entity": _e}
        # Code Block: apex cube
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(pos=(0.66, 0.02, 0.7825), euler=(0, 0, 0), size=(0.045, 0.045, 0.045)),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.9, 0.1, 0.1), roughness=0.4),
        )
        self._entities["apex cube"] = {"entity": _e}
        # Code Block: bluetooth keyboard
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/2d4bb6af-14cd-42db-bebb-076a72adddd7/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.46, 0.13, 0.7681),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["bluetooth keyboard"] = {"entity": _e}
        # Code Block: tv remote control
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/e3a03694-8872-4428-b864-836b1dfa6614/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.44, -0.12, 0.7735),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["tv remote control"] = {"entity": _e}
