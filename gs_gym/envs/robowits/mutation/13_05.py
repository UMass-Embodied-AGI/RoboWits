from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

CoverWithLidEnv = _import("gs_gym.envs.robowits.13_cover_with_lid").CoverWithLidEnv


@register_task("robowits/13_05-v0")
class CoverWithLidMut5Env(CoverWithLidEnv):
    """Trap1 mutation of task 13 (CoverWithLid).

    The intended lid is fixed, forcing tool substitution. The agent must reason about circular fit and select a
    different flat, round object with a diameter just larger than the opening to achieve precise coverage.
    """

    # lid_medium is fixed to the table; the plate is the substitute cover.
    COVER_CANDIDATES = (*CoverWithLidEnv.COVER_CANDIDATES, "plate_medium")

    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("pot_without_lid", "small_lid", "lid_medium", "lid_large", "plate_medium", "coaster_medium")

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
                scale=0.16,
                pos=(0.50, 0.00, 0.8132),
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
                scale=(0.1092, 0.1092, 0.5),
                pos=(0.62, -0.05, 0.7646),
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
                scale=(0.1700, 0.1700, 0.5),
                pos=(0.38, 0.08, 0.7677),
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
                scale=(0.2423, 0.2423, 0.5),
                pos=(0.70, 0.30, 0.7707),
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
                scale=0.4149,
                pos=(0.50, 0.12, 0.7635),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["plate_medium"] = {"entity": _e}
        # Code Block: coaster medium
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.50, -0.12, 0.7630), euler=(0.0, 0.0, 0.0), radius=0.054, height=0.006, fixed=True, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.55, 0.36, 0.20), roughness=0.8, ior=1.5),
        )
        self._entities["coaster_medium"] = {"entity": _e}
