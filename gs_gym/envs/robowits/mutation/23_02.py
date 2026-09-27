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
        """Reuse the seed task's groups so the shelf cluster and book still randomize.

        Returning None here left every episode with the identical canonical layout.
        Names the parent lists but this variant lacks are dropped by _place_objects.
        """
        return (*super().placement_groups, "notepad")

    def _add_custom_entities(self) -> None:
        """Shelf task variant: notepad clutter."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )
        self._build_shelf_scene(
            coacd_options,
            board_pos=(0.27, 0.28, 0.80),
        )

        # Code Block: notepad
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/53356b02-b7e3-49a9-b573-13fa5032c1e5/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.30, -0.30, 0.7653),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.5, coup_friction=0.1),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["notepad"] = {"entity": _e}
