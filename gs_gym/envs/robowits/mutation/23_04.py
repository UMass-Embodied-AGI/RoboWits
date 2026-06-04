from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

PlaceBookEnv = _import("gs_gym.envs.robowits.23_place_book").PlaceBookEnv


@register_task("robowits/23_04-v0")
class PlaceBookMut4Env(PlaceBookEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return None

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: blue target mat
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.68, 0.26, 0.762), euler=(0, 0, 0), size=(0.08, 0.14, 0.004), fixed=True, collision=True
            ),
            material=gs.materials.Rigid(rho=800.0, friction=1.2, coup_friction=0.4),
            surface=gs.surfaces.Default(color=(0.1, 0.3, 0.9), roughness=0.8, ior=1.5),
        )
        self._entities["blue target mat"] = {"entity": _e}
        # Code Block: thin book
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/decafac9-3ae6-4a22-983d-34f365f95642/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.37, 0.0, 0.7654),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500.0, friction=0.2, coup_friction=0.15),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["thin book"] = {"entity": _e}
        # Code Block: heavy block
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.350, 0.0, 0.7961), euler=(0, 0, 0), size=(0.085, 0.16, 0.05), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=8000.0, friction=0.9, coup_friction=0.6),
            surface=gs.surfaces.Smooth(color=(0.25, 0.25, 0.28), double_sided=True),
        )
        self._entities["heavy block"] = {"entity": _e}
        # Code Block: eraser
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/361aa36f-d104-465c-afb1-ec68c2f6611f/obj.glb", pattern_is_dir=False),
                scale=1.8,
                pos=(0.39, 0.16, 0.7729),
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=1100.0, friction=1.5, coup_friction=0.8),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["eraser"] = {"entity": _e}
