from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

MoveCubeEnv = _import("gs_gym.envs.robowits.28_move_cube").MoveCubeEnv


@register_task("robowits/28_04-v0")
class MoveCubeMut4Env(MoveCubeEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("cube 1", "cube 2", "cube 3", "goal line", "ruler", "rough slope")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: goal line
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.9, 0.0, 0.761), euler=(0, 0, 0), size=(0.004, 0.9, 0.002), fixed=True, collision=False
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.5),
            surface=gs.surfaces.Default(color=(1.0, 0.9, 0.1), roughness=0.3),
        )
        self._entities["goal line"] = {"entity": _e}
        # Code Block: cube 1
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.40, 0.00, 0.7825), euler=(0, 0, 0), size=(0.045, 0.045, 0.045), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.01),
            surface=gs.surfaces.Default(color=(0.9, 0.2, 0.2), roughness=0.5),
        )
        self._entities["cube 1"] = {"entity": _e}
        # Code Block: cube 2
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.60, 0.12, 0.7825), euler=(0, 0, 0), size=(0.045, 0.045, 0.045), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.2),
            surface=gs.surfaces.Default(color=(0.2, 0.9, 0.2), roughness=0.5),
        )
        self._entities["cube 2"] = {"entity": _e}
        # Code Block: cube 3
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.60, -0.12, 0.7825), euler=(0, 0, 0), size=(0.045, 0.045, 0.045), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.01),
            surface=gs.surfaces.Default(color=(0.2, 0.2, 0.9), roughness=0.5),
        )
        self._entities["cube 3"] = {"entity": _e}
        # Code Block: ruler
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.46, -0.05, 0.7625), euler=(0, 0, 0), size=(0.28, 0.03, 0.005), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.6),
            surface=gs.surfaces.Smooth(color=(0.85, 0.7, 0.5), double_sided=True),
        )
        self._entities["ruler"] = {"entity": _e}
        # Code Block: rough slope
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/ab77c172-c69f-41b5-956a-600ea7b64c73/obj.glb", pattern_is_dir=False),
                scale=0.6,
                pos=(0.68, 0.0, 0.79516),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1.5),
            surface=gs.surfaces.Rough(color=(0.5, 0.4, 0.3), double_sided=True),
        )
        self._entities["rough slope"] = {"entity": _e}
