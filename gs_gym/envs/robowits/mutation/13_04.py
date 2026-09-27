from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

CoverWithLidEnv = _import("gs_gym.envs.robowits.13_cover_with_lid").CoverWithLidEnv


@register_task("robowits/13_04-v0")
class CoverWithLidMut4Env(CoverWithLidEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("pot_without_lid", "small_lid", "lid_medium", "lid_large")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: pot without lid
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/2cf9e625-d115-46a9-98af-c771663006db/obj.glb", pattern_is_dir=False),
                scale=0.1212,
                pos=(0.50, 0.00, 0.8454),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["pot_without_lid"] = {"entity": _e}
        # Code Block: small lid
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c1a05823-5cb6-4e64-b3f3-5fd5a86cfc0b/obj.glb", pattern_is_dir=False),
                scale=(0.0718, 0.0718, 0.5),
                pos=(0.66, -0.10, 0.7623),
                euler=(0.0, 0.0, 90.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500, friction=1.5),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["small_lid"] = {"entity": _e}
        # Code Block: lid medium
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c1a05823-5cb6-4e64-b3f3-5fd5a86cfc0b/obj.glb", pattern_is_dir=False),
                scale=(0.0997, 0.0997, 0.5),
                pos=(0.38, 0.10, 0.7639),
                euler=(0.0, 0.0, 90.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500, friction=1.5),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["lid_medium"] = {"entity": _e}
        # Code Block: lid large
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c1a05823-5cb6-4e64-b3f3-5fd5a86cfc0b/obj.glb", pattern_is_dir=False),
                scale=(0.1516, 0.1516, 0.5),
                pos=(0.659, 0.135, 0.7669),
                euler=(0.0, 0.0, 90.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500, friction=1.5),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["lid_large"] = {"entity": _e}
