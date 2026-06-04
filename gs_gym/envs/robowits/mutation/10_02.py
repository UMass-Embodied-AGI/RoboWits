from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

CollectScrewsEnv = _import("gs_gym.envs.robowits.10_collect_screws").CollectScrewsEnv


@register_task("robowits/10_02-v0")
class CollectScrewsMut2Env(CollectScrewsEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("screw", "dustpan", "container", "glass tumbler", "hook stylized", "olive oil bottle")

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
                pos=(0.55, 0.00, 0.76 + 0.0491),
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
        # Code Block: glass tumbler
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/15df4d88-ca29-414a-98e6-4a557b40d8ae/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.40, -0.05, 0.76 + 0.0441),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(color=(0.7, 0.85, 0.95), double_sided=True),
        )
        self._entities["glass tumbler"] = {"entity": _e}
        # Code Block: hook stylized
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/877c1770-de6e-439c-8713-3b0eba76d272/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.66, -0.30, 0.76 + 0.1315),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(color=(0.3, 0.6, 0.3), double_sided=True),
        )
        self._entities["hook stylized"] = {"entity": _e}
        # Code Block: olive oil bottle
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/de0009fd-9494-44de-be80-0e093bae4d9a/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.36, 0.44, 0.76 + 0.161),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(color=(0.7, 0.75, 0.2), double_sided=True),
        )
        self._entities["olive oil bottle"] = {"entity": _e}
