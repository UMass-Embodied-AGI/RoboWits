from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

PlaceBookEnv = _import("gs_gym.envs.robowits.23_place_book").PlaceBookEnv


@register_task("robowits/23_03-v0")
class PlaceBookMut3Env(PlaceBookEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Reuse the seed task's groups so the shelf cluster and book still randomize.

        Returning None here left every episode with the identical canonical layout.
        Names the parent lists but this variant lacks are dropped by _place_objects.
        """
        return (*super().placement_groups, "dispenser", "eraser", "usb_a_wall_charger")

    def _add_custom_entities(self) -> None:
        """Shelf task variant: heavy table clutter (eraser, dispenser, charger)."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )
        self._build_shelf_scene(
            coacd_options,
            board_pos=(0.26, 0.30, 0.80),
        )

        # Code Block: eraser
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/361aa36f-d104-465c-afb1-ec68c2f6611f/obj.glb", pattern_is_dir=False),
                scale=1.5,
                pos=(0.33, -0.28, 0.7708),
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=1100.0, friction=1.3),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["eraser"] = {"entity": _e}
        # Code Block: dispenser
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/48cd9352-3020-4c02-aa4a-13f87f712762/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.30, 0.33, 0.8445),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500.0, friction=0.6),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["dispenser"] = {"entity": _e}
        # Code Block: usb a wall charger
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/a758d464-0bb9-490a-a602-4d21d769d086/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.30, -0.40, 0.773),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=400.0, friction=0.5),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["usb_a_wall_charger"] = {"entity": _e}
