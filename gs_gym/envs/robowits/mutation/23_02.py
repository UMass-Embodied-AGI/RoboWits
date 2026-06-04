from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

PlaceBookEnv = _import("gs_gym.envs.robowits.23_place_book").PlaceBookEnv


@register_task("robowits/23_02-v0")
class PlaceBookMut2Env(PlaceBookEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return None

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: blue_target_mat
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.66, 0.32, 0.762), euler=(0, 0, 0), size=(0.10, 0.26, 0.004), fixed=True, collision=True
            ),
            material=gs.materials.Rigid(rho=500.0, friction=1.2, coup_friction=0.2),
            surface=gs.surfaces.Default(color=(0.1, 0.3, 0.9), roughness=0.8, ior=1.5),
        )
        self._entities["blue_target_mat"] = {"entity": _e}
        # Code Block: thin_book
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/decafac9-3ae6-4a22-983d-34f365f95642/obj.glb", pattern_is_dir=False),
                scale=1.1604,
                pos=(0.38, -0.05, 0.7663),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.25, coup_friction=0.1),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["thin_book"] = {"entity": _e}
        # Code Block: heavy_block
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.38, -0.022, 0.8031), euler=(0, 0, 0), size=(0.14, 0.14, 0.06), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=8000.0, friction=0.9, coup_friction=0.2),
            surface=gs.surfaces.Smooth(color=(0.25, 0.25, 0.28), double_sided=True),
        )
        self._entities["heavy_block"] = {"entity": _e}
        # Code Block: pry_board
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/08893dc4-bfb1-49ca-9e47-4b3958a21e4b/obj.glb", pattern_is_dir=False),
                scale=0.0979,
                pos=(0.595, -0.15, 0.7646),
                euler=(0, 0, 12),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=600.0, friction=0.5, coup_friction=0.15),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["pry_board"] = {"entity": _e}
        # Code Block: notepad
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/53356b02-b7e3-49a9-b573-13fa5032c1e5/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.66, -0.30, 0.7653),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.5, coup_friction=0.1),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["notepad"] = {"entity": _e}
        # Code Block: books
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c07cdc08-3140-4603-9680-6e01ddfa6c5c/obj.glb", pattern_is_dir=False),
                scale=0.746,
                pos=(0.62, 0.06, 0.8056),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.5, coup_friction=0.1),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["books"] = {"entity": _e}
