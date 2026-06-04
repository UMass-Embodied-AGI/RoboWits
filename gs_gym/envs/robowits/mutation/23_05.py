from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

PlaceBookEnv = _import("gs_gym.envs.robowits.23_place_book").PlaceBookEnv


@register_task("robowits/23_05-v0")
class PlaceBookMut5Env(PlaceBookEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return None

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: thin book
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/decafac9-3ae6-4a22-983d-34f365f95642/obj.glb", pattern_is_dir=False),
                scale=(1.5657, 1.3889, 0.8119),
                pos=(0.405, 0.0, 0.7675),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=400.0, friction=None),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["thin book"] = {"entity": _e}
        # Code Block: heavy block
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.455, 0.0, 0.8075), euler=(0, 0, 0), size=(0.22, 0.19, 0.07), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=5000.0, friction=1.0),
            surface=gs.surfaces.Smooth(color=(0.25, 0.25, 0.28), double_sided=True),
        )
        self._entities["heavy block"] = {"entity": _e}
        # Code Block: pry board
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/08893dc4-bfb1-49ca-9e47-4b3958a21e4b/obj.glb", pattern_is_dir=False),
                scale=(0.1097, 0.0641, 0.0881),
                pos=(0.44, -0.18, 0.763),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=700.0, friction=None),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["pry board"] = {"entity": _e}
        # Code Block: hammer
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/ab77c172-c69f-41b5-956a-600ea7b64c73/obj.glb", pattern_is_dir=False),
                scale=(0.538, 0.3414, 0.3386),
                pos=(0.44, 0.15, 0.78),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=1000.0, friction=None),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["hammer"] = {"entity": _e}
        # Code Block: blue target mat
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.70, 0.34, 0.7625), euler=(0, 0, 0), size=(0.16, 0.09, 0.005), fixed=True, collision=True
            ),
            material=gs.materials.Rigid(rho=500.0, friction=1.2),
            surface=gs.surfaces.Default(color=(0.1, 0.3, 0.9), roughness=0.8, ior=1.5),
        )
        self._entities["blue target mat"] = {"entity": _e}
