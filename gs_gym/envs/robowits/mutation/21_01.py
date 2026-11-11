from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

SealColanderEnv = _import("gs_gym.envs.robowits.21_seal_colander").SealColanderEnv


@register_task("robowits/21_01-v0")
class SealColanderMut1Env(SealColanderEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Randomize the container, holder and clutter -- never the pitcher or its water.

        The water is authored inside the pitcher, so grouping it with the perforated
        container (24 cm away) dragged the liquid out of the pitcher on every reset.
        21_05 documents hitting exactly that and blowing up the solver.
        """
        return ("perforated_container", "curved_holder", "flowerpot", "coin_purse", "milk_pack", "wine_bottle")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: perforated container (the box ladle, see 21_seal_colander.py)
        self._add_ladle((0.44, -0.06), yaw_deg=180.0)
        # Code Block: pitcher with its water (task 25's, see 21_seal_colander.py)
        self._add_pitcher_with_water(0.62, -0.1)
        # Code Block: curved holder (the dome, see 21_seal_colander.py)
        self._add_curved_holder((0.50, -0.15))
        # Code Block: flowerpot
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/3aee9353-e21a-40d4-b160-e2a0af1fff7f/obj.glb", pattern_is_dir=False),
                scale=0.15,
                pos=(0.34, -0.38, 0.76 + 0.3327 * 0.15),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["flowerpot"] = {"entity": _e}
        # Code Block: coin purse
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/144f65c2-bc70-4602-8740-9824bf70a928/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.33, 0.15, 0.76 + 0.0188),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["coin_purse"] = {"entity": _e}
        # Code Block: milk pack
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/d5cd5a5e-c548-4acd-8219-d685419731ed/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.70, 0.12, 0.76 + 0.0606),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["milk_pack"] = {"entity": _e}
        # Code Block: wine bottle
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c1ca6be7-d3dc-4817-8df7-6a34c8abda4a/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.48, 0.44, 0.76 + 0.155),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Glass(color=(0.2, 0.2, 0.2), double_sided=True),
        )
        self._entities["wine_bottle"] = {"entity": _e}
