from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

CollectScrewsEnv = _import("gs_gym.envs.robowits.10_collect_screws").CollectScrewsEnv


@register_task("robowits/10_04-v0")
class CollectScrewsMut4Env(CollectScrewsEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("screw", "dustpan", "container", "hand brush")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: container
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/382432e6-4f0d-44b6-98f9-2f3a013a47e2/obj.glb", pattern_is_dir=False),
                scale=1.05,
                pos=(0.70, 0.00, 0.76 + 0.5 * (0.0491 - (-0.0491)) * 1.05),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["container"] = {"entity": _e}
        # Code Block: dustpan
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/dustpan.STL", pattern_is_dir=False),
                scale=0.001,
                pos=(0.68, 0.18, 0.83),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(color=(0.9, 0.8, 0.6), double_sided=True),
        )
        self._entities["dustpan"] = {"entity": _e}
        # Code Block: hand brush
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/f4a6d359-8d33-4a1e-be80-b29142be5135/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.68, 0.42, 0.7786),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(color=(0.6, 0.4, 0.2), double_sided=True),
        )
        self._entities["hand brush"] = {"entity": _e}
        # Code Block: screw
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/1d373cbe-c73d-4929-bf32-e85a98dc4bca/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.62, -0.1, 0.76 + 0.5 * (0.0074 - (-0.0074))),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.6),
            surface=gs.surfaces.Rough(color=(0.6, 0.7, 0.9), double_sided=True),
        )
        self._entities["screw"] = {"entity": _e}
