from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

StackCubesEnv = _import("gs_gym.envs.robowits.14_stack_cubes").StackCubesEnv


@register_task("robowits/14_05-v0")
class StackCubesMut5Env(StackCubesEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("apex_target", "base_cube_1", "base_cube_2", "apex_cube", "sponge")

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
        self._entities["apex_target"] = {"entity": _e}
        # Code Block: base cube 1
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(pos=(0.50, -0.10, 0.785), euler=(0, 0, 0), size=(0.05, 0.05, 0.05)),
            material=gs.materials.Rigid(rho=600.0, friction=1.0),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5),
        )
        self._entities["base_cube_1"] = {"entity": _e}
        # Code Block: base cube 2
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(pos=(0.66, 0.10, 0.785), euler=(0, 0, 0), size=(0.05, 0.05, 0.05)),
            material=gs.materials.Rigid(rho=600.0, friction=1.0),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5),
        )
        self._entities["base_cube_2"] = {"entity": _e}
        # Code Block: apex cube
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(pos=(0.60, 0.15, 0.785), euler=(0, 0, 0), size=(0.05, 0.05, 0.05)),
            material=gs.materials.Rigid(rho=600.0, friction=1.0),
            surface=gs.surfaces.Default(color=(0.9, 0.1, 0.1), roughness=0.4),
        )
        self._entities["apex_cube"] = {"entity": _e}
        # Code Block: sponge
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/a939eb10-a16c-4e6e-b766-0805109143d4/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.70, -0.15, 0.7786),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.8),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["sponge"] = {"entity": _e}
