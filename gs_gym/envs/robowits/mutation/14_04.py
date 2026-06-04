from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

StackCubesEnv = _import("gs_gym.envs.robowits.14_stack_cubes").StackCubesEnv


@register_task("robowits/14_04-v0")
class StackCubesMut4Env(StackCubesEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("apex target", "base cube 1", "base cube 2", "apex cube", "rubber mat", "magazine page")

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
            material=gs.materials.Rigid(rho=200.0, friction=0.5),
            surface=gs.surfaces.Default(color=(0.1, 0.8, 0.2), roughness=0.4),
        )
        self._entities["apex target"] = {"entity": _e}
        # Code Block: base cube 1
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.45, -0.10, 0.785), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.05, coup_friction=0.05),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5),
        )
        self._entities["base cube 1"] = {"entity": _e}
        # Code Block: base cube 2
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.55, 0.12, 0.785), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.05, coup_friction=0.05),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5),
        )
        self._entities["base cube 2"] = {"entity": _e}
        # Code Block: apex cube
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.64, -0.05, 0.785), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.1, coup_friction=0.1),
            surface=gs.surfaces.Default(color=(0.9, 0.1, 0.1), roughness=0.4),
        )
        self._entities["apex cube"] = {"entity": _e}
        # Code Block: rubber mat
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c88f2500-1105-4df1-8bc0-320879db514c/obj.glb", pattern_is_dir=False),
                scale=(0.1266, 0.0667, 0.0688),
                pos=(0.70, 0.08, 0.761),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1.0, coup_friction=0.8),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["rubber mat"] = {"entity": _e}
        # Code Block: magazine page
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.39, 0.425, 0.7605), euler=(0, 0, 0), size=(0.18, 0.05, 0.001), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.05, coup_friction=0.05),
            surface=gs.surfaces.Smooth(color=(0.9, 0.9, 0.9), double_sided=True),
        )
        self._entities["magazine page"] = {"entity": _e}
