from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

SealColanderEnv = _import("gs_gym.envs.robowits.21_seal_colander").SealColanderEnv


@register_task("robowits/21_05-v0")
class SealColanderMut5Env(SealColanderEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        # The MPM water sits inside the pitcher, so it has to move *with* the
        # pitcher: grouping it with the perforated container dragged the liquid out
        # of the pitcher while the pitcher was relocated separately, and the
        # resulting interpenetration blew the solver up at every substeps value.
        return ("perforated_container", "curved_holder", "flowerpot")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: perforated container (the box ladle, see 21_seal_colander.py)
        self._add_ladle((0.44, -0.06), yaw_deg=180.0)
        # Code Block: pitcher with its water (task 25's, see 21_seal_colander.py)
        self._add_pitcher_with_water(0.61, 0.0)
        # Code Block: curved holder (the dome, see 21_seal_colander.py)
        self._add_curved_holder((0.50, -0.15))
        # Code Block: flowerpot
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/3aee9353-e21a-40d4-b160-e2a0af1fff7f/obj.glb", pattern_is_dir=False),
                scale=0.165,
                pos=(0.35, 0.12, 0.76 + 0.3327 * 0.165),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=600.0),
            surface=gs.surfaces.Rough(color=(0.7, 0.7, 0.7), double_sided=True),
        )
        self._entities["flowerpot"] = {"entity": _e}
