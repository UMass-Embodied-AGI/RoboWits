from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

RetrieveCubeEnv = _import("gs_gym.envs.robowits.02_retrieve_cube").RetrieveCubeEnv


@register_task("robowits/02_07-v0")
class RetrieveCubeMut7Env(RetrieveCubeEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (("wide open container", "cube"), "target area")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: wide open container
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/d8dd7f3f-103d-4daf-b579-188178dc4d9e/obj.glb", pattern_is_dir=False),
                scale=1.27,
                pos=(0.50, 0.00, 0.7905),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Glass(color=(1.0, 1.0, 1.0), double_sided=True),
        )
        self._entities["wide open container"] = {"entity": _e}
        # Code Block: cube
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.50, 0.00, 0.7933), euler=(0.0, 0.0, 0.0), size=(0.015, 0.015, 0.015), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(color=(0.8, 0.6, 0.4), double_sided=True),
        )
        self._entities["cube"] = {"entity": _e}
        # Code Block: target area
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.64, 0.435, 0.761), euler=(0.0, 0.0, 0.0), size=(0.12, 0.12, 0.002), fixed=True, collision=False
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(color=(0.1, 0.8, 0.1), double_sided=True),
        )
        self._entities["target area"] = {"entity": _e}
