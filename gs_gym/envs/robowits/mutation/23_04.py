from importlib import import_module as _import

import genesis as gs

from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

PlaceBookEnv = _import("gs_gym.envs.robowits.23_place_book").PlaceBookEnv


@register_task("robowits/23_04-v0")
class PlaceBookMut4Env(PlaceBookEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Reuse the seed task's groups so the shelf cluster and book still randomize.

        Returning None here left every episode with the identical canonical layout.
        Names the parent lists but this variant lacks are dropped by _place_objects.
        """
        return super().placement_groups

    def _add_custom_entities(self) -> None:
        """Pivot: unanchored shelf top and no board, so the shelf must be opened from above.

        'shelf_top' is spawned movable instead of fixed, so the robot can lift it off and
        reach the row from above; dropping the board removes the only object long enough
        to sweep the leaning books upright from outside the cubby.
        """
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )
        self._build_shelf_scene(
            coacd_options,
            shelf_top_fixed=False,
            include_board=False,
        )
