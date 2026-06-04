from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

BalanceBoardEnv = _import("gs_gym.envs.robowits.29_balance_board").BalanceBoardEnv


@register_task("robowits/29_02-v0")
class BalanceBoardMut2Env(BalanceBoardEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("small wall", ("board", "support cube"), "paperweight", "tennis ball", "orange beverage can")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: small wall
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.68, 0.0, 0.82), euler=(0, 0, 0), size=(0.015, 0.22, 0.12), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=400.0),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.6),
        )
        self._entities["small wall"] = {"entity": _e}
        # Code Block: board
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/f923466c-c37d-4744-b197-6089b2899715/obj.glb", pattern_is_dir=False),
                scale=0.6,
                pos=(0.5, 0.0, 0.7668 + 0.03),
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
            gs.morphs.Box(pos=(0.48, 0.0, 0.78), euler=(0, 0, 0), size=(0.04, 0.04, 0.04), fixed=True, collision=True),
            material=gs.materials.Rigid(rho=200.0, friction=1),
            surface=gs.surfaces.Default(color=(0.95, 0.95, 0.95), roughness=0.5),
        )
        self._entities["support cube"] = {"entity": _e}
        # Code Block: paperweight
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.63, 0.15, 0.77), euler=(0, 0, 0), size=(0.04, 0.04, 0.02), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=5000.0, friction=0.8),
            surface=gs.surfaces.Metal(double_sided=True, metal_type="iron"),
        )
        self._entities["paperweight"] = {"entity": _e}
        # Code Block: tennis ball
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb", pattern_is_dir=False),
                scale=0.4467,
                pos=(0.59, -0.15, 0.7935),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.7, coup_restitution=0.3),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["tennis ball"] = {"entity": _e}
        # Code Block: orange beverage can
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/5ca78aac-f60f-43b8-8ab7-a6c668a3527d/obj.glb", pattern_is_dir=False),
                scale=0.9,
                pos=(0.70, 0.42, 0.8304),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["orange beverage can"] = {"entity": _e}
