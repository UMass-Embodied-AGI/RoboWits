from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

CoverWithLidEnv = _import("gs_gym.envs.robowits.13_cover_with_lid").CoverWithLidEnv


@register_task("robowits/13_03-v0")
class CoverWithLidMut3Env(CoverWithLidEnv):
    """Pivot1 mutation of task 13 (CoverWithLid).

    The intended lid is fixed, forcing tool substitution. The agent must reason about circular fit and select a
    different flat, round object with a diameter just larger than the opening to achieve precise coverage.
    """

    # lid_medium is fixed to the table; the plate is the substitute cover.
    COVER_CANDIDATES = (*CoverWithLidEnv.COVER_CANDIDATES, "plate_medium")

    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("pot_without_lid", "small_lid", "lid_medium", "lid_large", "plate_medium")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: pot without lid
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/3aee9353-e21a-40d4-b160-e2a0af1fff7f/obj.glb", pattern_is_dir=False),
                scale=0.1212,
                pos=(0.52, 0.00, 0.8006),
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
                scale=(0.0827, 0.0827, 0.5),
                pos=(0.60, -0.10, 0.7640),
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
                scale=(0.1288, 0.1288, 0.5),
                pos=(0.40, 0.12, 0.7655),
                euler=(0.0, 0.0, 90.0),
                fixed=True,
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
                scale=(0.1835, 0.1835, 0.5),
                pos=(0.68, -0.12, 0.7672),
                euler=(0.0, 0.0, 90.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500, friction=1.5),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["lid_large"] = {"entity": _e}
        # Code Block: plate medium
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/2746f256-610e-4b14-949c-6bd5e2718f5a/obj.glb", pattern_is_dir=False),
                scale=0.3144,
                pos=(0.50, 0.12, 0.7627),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["plate_medium"] = {"entity": _e}
