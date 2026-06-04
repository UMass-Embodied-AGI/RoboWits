from importlib import import_module as _import

import genesis as gs

from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

StackCubesEnv = _import("gs_gym.envs.robowits.14_stack_cubes").StackCubesEnv


@register_task("robowits/14_02-v0")
class StackCubesMut2Env(StackCubesEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("apex target", "base cube 1", "base cube 2", "apex cube", "rubber ball", "sunglasses")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        # Code Block: apex target
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.60725, 0.00335, 0.7586), euler=(0, 0, 0), size=(0.02, 0.02, 0.001), fixed=True, collision=False
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.1, 0.8, 0.2), roughness=0.4),
        )
        self._entities["apex target"] = {"entity": _e}
        # Code Block: base cube 1
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.55, -0.12, 0.7831), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=400.0, friction=1.0),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5),
        )
        self._entities["base cube 1"] = {"entity": _e}
        # Code Block: base cube 2
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.66, 0.12, 0.7831), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=400.0, friction=1.0),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5),
        )
        self._entities["base cube 2"] = {"entity": _e}
        # Code Block: apex cube
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.48, 0.02, 0.7831), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=400.0, friction=1.0),
            surface=gs.surfaces.Default(color=(0.9, 0.1, 0.1), roughness=0.4),
        )
        self._entities["apex cube"] = {"entity": _e}
        # Code Block: rubber ball
        _e = self._scene.scene.add_entity(
            gs.morphs.Sphere(pos=(0.70, -0.15, 0.7756), euler=(0, 0, 0), radius=0.0175, fixed=False, collision=True),
            material=gs.materials.Rigid(rho=300.0, friction=0.2, coup_friction=0.05),
            surface=gs.surfaces.Smooth(color=(0.9, 0.7, 0.1), double_sided=False),
        )
        self._entities["rubber ball"] = {"entity": _e}
        # Code Block: sunglasses
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.39, -0.05, 0.7656), euler=(0, 0, 0), size=(0.15, 0.06, 0.015), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=300.0, friction=0.7),
            surface=gs.surfaces.Default(color=(0.05, 0.05, 0.06), roughness=0.4, ior=1.5),
        )
        self._entities["sunglasses"] = {"entity": _e}
