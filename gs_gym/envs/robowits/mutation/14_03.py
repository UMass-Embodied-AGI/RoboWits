from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

StackCubesEnv = _import("gs_gym.envs.robowits.14_stack_cubes").StackCubesEnv


@register_task("robowits/14_03-v0")
class StackCubesMut3Env(StackCubesEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("apex target", "base cube 1", "base cube 2", "apex cube", "rubber mat")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

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
        self._entities["apex target"] = {"entity": _e}
        # Code Block: base cube 1
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.50, -0.10, 0.7581 + 0.025), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.05, coup_friction=0.05),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5),
        )
        self._entities["base cube 1"] = {"entity": _e}
        # Code Block: base cube 2
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.66, -0.10, 0.7581 + 0.025), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.05, coup_friction=0.05),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5),
        )
        self._entities["base cube 2"] = {"entity": _e}
        # Code Block: apex cube
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.52, 0.12, 0.7581 + 0.025), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.05, coup_friction=0.05),
            surface=gs.surfaces.Default(color=(0.9, 0.1, 0.1), roughness=0.4),
        )
        self._entities["apex cube"] = {"entity": _e}
        # Code Block: rubber mat
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c88f2500-1105-4df1-8bc0-320879db514c/obj.glb", pattern_is_dir=False),
                scale=(0.08443, 0.13333, 0.06877),
                pos=(0.69, 0.10, 0.7581 + 0.002),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=300.0, friction=1.5, coup_friction=1.0),
            surface=gs.surfaces.Rough(double_sided=True, vis_mode="visual"),
        )
        self._entities["rubber mat"] = {"entity": _e}
