from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

MoveCubeEnv = _import("gs_gym.envs.robowits.28_move_cube").MoveCubeEnv


@register_task("robowits/28_06-v0")
class MoveCubeMut6Env(MoveCubeEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("cube 1", "cube 2", "cube 3", "goal line", "smooth slope", "rough slope", "doorstop")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: cube 1
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.40, -0.05, 0.785), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.01),
            surface=gs.surfaces.Default(color=(0.9, 0.2, 0.2), roughness=0.5),
        )
        self._entities["cube 1"] = {"entity": _e}
        # Code Block: cube 2
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.50, 0.00, 0.785), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.01),
            surface=gs.surfaces.Default(color=(0.2, 0.9, 0.2), roughness=0.5),
        )
        self._entities["cube 2"] = {"entity": _e}
        # Code Block: cube 3
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.60, 0.05, 0.785), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.01),
            surface=gs.surfaces.Default(color=(0.2, 0.2, 0.9), roughness=0.5),
        )
        self._entities["cube 3"] = {"entity": _e}
        # Code Block: goal line
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.90, 0.0, 0.761), euler=(0, 0, 0), size=(0.002, 1.10, 0.002), fixed=True, collision=False
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.5),
            surface=gs.surfaces.Default(color=(1.0, 0.9, 0.1), roughness=0.3),
        )
        self._entities["goal line"] = {"entity": _e}
        # Code Block: smooth slope
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/ab77c172-c69f-41b5-956a-600ea7b64c73/obj.glb", pattern_is_dir=False),
                scale=0.43,
                pos=(0.66, -0.15, 0.7834),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.01),
            surface=gs.surfaces.Smooth(color=(0.8, 0.8, 0.85), double_sided=True),
        )
        self._entities["smooth slope"] = {"entity": _e}
        # Code Block: rough slope
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/ab77c172-c69f-41b5-956a-600ea7b64c73/obj.glb", pattern_is_dir=False),
                scale=0.43,
                pos=(0.70, 0.15, 0.7834),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1.5),
            surface=gs.surfaces.Rough(color=(0.5, 0.4, 0.3), double_sided=True),
        )
        self._entities["rough slope"] = {"entity": _e}
        # Code Block: doorstop
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/ab77c172-c69f-41b5-956a-600ea7b64c73/obj.glb", pattern_is_dir=False),
                scale=0.215,
                pos=(0.68, 0.00, 0.773),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(
                rho=900.0, friction=1.8, coup_friction=0.25, coup_softness=0.002, coup_restitution=0.0
            ),
            surface=gs.surfaces.Rough(color=(0.35, 0.35, 0.35), double_sided=False, vis_mode="visual"),
        )
        self._entities["doorstop"] = {"entity": _e}
