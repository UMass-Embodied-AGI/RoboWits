from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

MoveCubeEnv = _import("gs_gym.envs.robowits.28_move_cube").MoveCubeEnv


@register_task("robowits/28_01-v0")
class MoveCubeMut1Env(MoveCubeEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("cube 1", "cube 2", "cube 3", "goal line", "smooth ramp", "spice jar", "wine bottle")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: cube 1
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.45, -0.05, 0.76 + 0.02), euler=(0, 0, 0), size=(0.04, 0.04, 0.04), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.01),
            surface=gs.surfaces.Default(color=(0.9, 0.2, 0.2), roughness=0.5),
        )
        self._entities["cube 1"] = {"entity": _e}
        # Code Block: cube 2
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.52, 0.05, 0.76 + 0.02), euler=(0, 0, 0), size=(0.04, 0.04, 0.04), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.01),
            surface=gs.surfaces.Default(color=(0.2, 0.9, 0.2), roughness=0.5),
        )
        self._entities["cube 2"] = {"entity": _e}
        # Code Block: cube 3
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.60, 0.15, 0.76 + 0.02), euler=(0, 0, 0), size=(0.04, 0.04, 0.04), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.2),
            surface=gs.surfaces.Default(color=(0.2, 0.2, 0.9), roughness=0.5),
        )
        self._entities["cube 3"] = {"entity": _e}
        # Code Block: goal line
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.90, 0.0, 0.76 + 0.001), euler=(0, 0, 0), size=(0.005, 0.60, 0.002), fixed=True, collision=False
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.5),
            surface=gs.surfaces.Default(color=(1.0, 0.9, 0.1), roughness=0.3),
        )
        self._entities["goal line"] = {"entity": _e}
        # Code Block: smooth ramp
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/ab77c172-c69f-41b5-956a-600ea7b64c73/obj.glb", pattern_is_dir=False),
                scale=(0.54, 0.85, 1.52),
                pos=(0.67, 0.0, 0.76 + 0.0586 * 0.85),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.01),
            surface=gs.surfaces.Smooth(color=(0.8, 0.8, 0.85), double_sided=True),
        )
        self._entities["smooth ramp"] = {"entity": _e}
        # Code Block: spice jar
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/0679bdbd-36cd-4026-ac4a-195c97c8198f/obj.glb", pattern_is_dir=False),
                scale=0.25,
                pos=(0.68, -0.42, 0.76 + (0.1367 * 0.25)),
                euler=(0, 0, 15),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.6),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["spice jar"] = {"entity": _e}
        # Code Block: wine bottle
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c1ca6be7-d3dc-4817-8df7-6a34c8abda4a/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.66, 0.35, 0.76 + 0.155),
                euler=(0, 0, -10),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.5),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["wine bottle"] = {"entity": _e}
