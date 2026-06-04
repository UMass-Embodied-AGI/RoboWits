from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

BalanceBoardEnv = _import("gs_gym.envs.robowits.29_balance_board").BalanceBoardEnv


@register_task("robowits/29_04-v0")
class BalanceBoardMut4Env(BalanceBoardEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("small wall", ("board", "support cube"), "paperweight")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: small wall
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.68, 0.10, 0.82), euler=(0, 0, 0), size=(0.10, 0.03, 0.12), fixed=False, collision=True
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
                scale=0.8,
                pos=(0.50, -0.09, 0.7709 + 0.05),
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
            gs.morphs.Box(
                pos=(0.48, -0.09, 0.78), euler=(0, 0, 0), size=(0.04, 0.04, 0.04), fixed=True, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1),
            surface=gs.surfaces.Default(color=(0.95, 0.95, 0.95), roughness=0.5),
        )
        self._entities["support cube"] = {"entity": _e}
        # Code Block: paperweight
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.40, 0.15, 0.76 + 0.04 / 2), euler=(0, 0, 0), size=(0.04, 0.04, 0.04), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=2500.0, friction=0.6),
            surface=gs.surfaces.Smooth(color=(0.6, 0.6, 0.6), double_sided=True),
        )
        self._entities["paperweight"] = {"entity": _e}
