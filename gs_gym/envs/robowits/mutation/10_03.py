from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

CollectScrewsEnv = _import("gs_gym.envs.robowits.10_collect_screws").CollectScrewsEnv


@register_task("robowits/10_03-v0")
class CollectScrewsMut3Env(CollectScrewsEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("screw", "dustpan", "container", "funnel", "paper sheet")

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
                scale=1.0,
                pos=(0.68, 0.00, 0.8091),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=400.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["container"] = {"entity": _e}
        # Code Block: dustpan
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/dustpan.STL", pattern_is_dir=False),
                scale=0.001,
                pos=(0.67, 0.25, 0.7996),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(color=(0.9, 0.8, 0.6), double_sided=True),
        )
        self._entities["dustpan"] = {"entity": _e}
        # Code Block: screw
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/1d373cbe-c73d-4929-bf32-e85a98dc4bca/obj.glb", pattern_is_dir=False),
                scale=1.5,
                pos=(0.56, -0.16, 0.7674),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(color=(0.6, 0.7, 0.9), double_sided=True),
        )
        self._entities["screw"] = {"entity": _e}
        # Code Block: funnel
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/a664c3de-c8d8-4042-ac6b-2eaf2b86c51e/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.54, -0.19, 0.8379),
                euler=(0.0, 0.0, 15.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=300.0),
            surface=gs.surfaces.Smooth(color=(0.9, 0.5, 0.2), double_sided=True),
        )
        self._entities["funnel"] = {"entity": _e}
        # Code Block: paper sheet
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.47, 0.05, 0.762), euler=(0.0, 0.0, 10.0), size=(0.297, 0.21, 0.002), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.95, 0.9, 0.75), roughness=0.8, ior=1.5),
        )
        self._entities["paper sheet"] = {"entity": _e}
