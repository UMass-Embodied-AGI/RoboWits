from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

BalanceBoardEnv = _import("gs_gym.envs.robowits.29_balance_board").BalanceBoardEnv


@register_task("robowits/29_03-v0")
class BalanceBoardMut3Env(BalanceBoardEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("small wall", ("board", "support cube"), "paper towel roll", "coin purse", "calendar cube decoration")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: small wall
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.52, 0.00, 0.76 + 0.12 / 2), euler=(0, 0, 0), size=(0.02, 0.24, 0.12), fixed=True, collision=True
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
                scale=0.68,
                pos=(0.334, -0.10, 0.76 + 0.0136 * 0.68 + 0.03),
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
                pos=(0.314, -0.10, 0.78), euler=(0, 0, 0), size=(0.04, 0.04, 0.04), fixed=True, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1),
            surface=gs.surfaces.Default(color=(0.95, 0.95, 0.95), roughness=0.5),
        )
        self._entities["support cube"] = {"entity": _e}
        # Code Block: paper towel roll
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/2efc7778-ceef-45b9-94c7-bed14ea25749/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.64, 0.11, 0.76 + 0.1522),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.8),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["paper towel roll"] = {"entity": _e}
        # Code Block: three tier office filing tray
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/a7f533d5-23e2-4994-8f82-e8b0b0c2700c/obj.glb", pattern_is_dir=False),
                scale=0.6,
                pos=(0.72, -0.09, 0.76 + 0.1288 * 0.6),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=350.0, friction=0.9),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["three tier office filing tray"] = {"entity": _e}
        # Code Block: coin purse
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/144f65c2-bc70-4602-8740-9824bf70a928/obj.glb", pattern_is_dir=False),
                scale=0.7,
                pos=(0.30, 0.17, 0.76 + 0.0188 * 0.7),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=300.0, friction=0.9),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["coin purse"] = {"entity": _e}
        # Code Block: calendar cube decoration
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/1f050ce6-e43d-4864-8fac-01b892274eb2/obj.glb", pattern_is_dir=False),
                scale=0.5,
                pos=(0.46, 0.17, 0.76 + 0.0609 * 0.5),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=400.0, friction=0.9),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["calendar cube decoration"] = {"entity": _e}
