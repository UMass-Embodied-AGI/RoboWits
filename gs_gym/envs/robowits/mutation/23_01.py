from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

PlaceBookEnv = _import("gs_gym.envs.robowits.23_place_book").PlaceBookEnv


@register_task("robowits/23_01-v0")
class PlaceBookMut1Env(PlaceBookEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Reuse the seed task's groups so the shelf cluster and book still randomize.

        Returning None here left every episode with the identical canonical layout.
        Names the parent lists but this variant lacks are dropped by _place_objects.
        """
        return (*super().placement_groups, "hammer", "wine_bottle")

    def _add_custom_entities(self) -> None:
        """Pivot scene (unanchored shelf top, no board) with hammer and wine-bottle clutter."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )
        self._build_shelf_scene(
            coacd_options,
            shelf_top_fixed=False,
            include_board=False,
        )

        # Code Block: hammer
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/ab77c172-c69f-41b5-956a-600ea7b64c73/obj.glb", pattern_is_dir=False),
                scale=(0.538, 0.3414, 0.3386),
                pos=(0.30, 0.28, 0.78),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=1000.0, friction=None),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["hammer"] = {"entity": _e}
        # Code Block: wine bottle
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c1ca6be7-d3dc-4817-8df7-6a34c8abda4a/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.62, 0.40, 0.915),
                euler=(0, 0, -10),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.5),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["wine_bottle"] = {"entity": _e}
