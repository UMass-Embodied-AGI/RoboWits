from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

BalanceBoardEnv = _import("gs_gym.envs.robowits.29_balance_board").BalanceBoardEnv


@register_task("robowits/29_01-v0")
class BalanceBoardMut1Env(BalanceBoardEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("small wall", ("board", "support cube"), "paperweight", "glass_tumbler", "dice", "body_lotion")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: small_wall
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.33, 0.00, 0.81), euler=(0, 0, 0), size=(0.02, 0.18, 0.10), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=400.0),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.6),
        )
        self._entities["small_wall"] = {"entity": _e}
        # Code Block: board
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/f923466c-c37d-4744-b197-6089b2899715/obj.glb", pattern_is_dir=False),
                scale=0.6206,
                pos=(0.57, -0.10, 0.7685 + 0.03),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=100),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["board"] = {"entity": _e}
        # Code Block: support cube
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(pos=(0.55, -0.1, 0.78), euler=(0, 0, 0), size=(0.04, 0.04, 0.04), fixed=True, collision=True),
            material=gs.materials.Rigid(rho=200.0, friction=1),
            surface=gs.surfaces.Default(color=(0.95, 0.95, 0.95), roughness=0.5),
        )
        self._entities["support cube"] = {"entity": _e}
        # Code Block: paperweight
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.60, 0.05, 0.78), euler=(0, 0, 0), size=(0.04, 0.04, 0.04), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=8000.0, friction=0.8),
            surface=gs.surfaces.Smooth(color=(0.5, 0.5, 0.55), double_sided=False),
        )
        self._entities["paperweight"] = {"entity": _e}
        # Code Block: glass_tumbler
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/15df4d88-ca29-414a-98e6-4a557b40d8ae/obj.glb", pattern_is_dir=False),
                scale=1.25,
                pos=(0.69, 0.12, 0.815),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=600.0, friction=0.9),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["glass_tumbler"] = {"entity": _e}
        # Code Block: dice
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/1b6970cb-cfe0-44db-8057-115a0476602f/obj.glb", pattern_is_dir=False),
                scale=1.6667,
                pos=(0.36, -0.10, 0.77),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=600.0, friction=0.9),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["dice"] = {"entity": _e}
        # Code Block: body_lotion
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/114f331b-ce87-46c6-85a5-c88f5cb77892/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.70, 0.28, 0.8457),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=600.0, friction=0.9),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["body_lotion"] = {"entity": _e}
