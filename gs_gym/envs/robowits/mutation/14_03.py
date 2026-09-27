from importlib import import_module as _import

import genesis as gs

from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

StackCubesEnv = _import("gs_gym.envs.robowits.14_stack_cubes").StackCubesEnv


@register_task("robowits/14_03-v0")
class StackCubesMut3Env(StackCubesEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("apex_target", "base_cube_1", "base_cube_2", "apex_cube", "coaster")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        # Code Block: apex target
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.607, 0.0, 0.7581 + 0.0005),
                euler=(0, 0, 0),
                size=(0.02, 0.02, 0.001),
                fixed=True,
                collision=False,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.5),
            surface=gs.surfaces.Default(color=(0.1, 0.8, 0.2), roughness=0.4),
        )
        self._entities["apex_target"] = {"entity": _e}
        # Code Block: base cube 1
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.50, -0.10, 0.7581 + 0.025), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.05, coup_friction=0.05),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5),
        )
        self._entities["base_cube_1"] = {"entity": _e}
        # Code Block: base cube 2
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.66, -0.10, 0.7581 + 0.025), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.05, coup_friction=0.05),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5),
        )
        self._entities["base_cube_2"] = {"entity": _e}
        # Code Block: apex cube
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.52, 0.12, 0.7581 + 0.025), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.05, coup_friction=0.05),
            surface=gs.surfaces.Default(color=(0.9, 0.1, 0.1), roughness=0.4),
        )
        self._entities["apex_cube"] = {"entity": _e}
        # Code Block: coaster
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.69, 0.10, 0.7581 + 0.004),
                euler=(0, 0, 0),
                radius=0.06,
                height=0.008,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=300.0, friction=1.5, coup_friction=1.0),
            surface=gs.surfaces.Default(color=(0.6, 0.4, 0.2), roughness=0.7),
        )
        self._entities["coaster"] = {"entity": _e}
