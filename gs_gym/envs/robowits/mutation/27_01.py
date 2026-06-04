from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

RetrieveRollEnv = _import("gs_gym.envs.robowits.27_retrieve_roll").RetrieveRollEnv


@register_task("robowits/27_01-v0")
class RetrieveRollMut1Env(RetrieveRollEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("green target area", "long rod", "hollow roll", "wire hanger", "body lotion", "screwdriver")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: hollow roll
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/a0c77eb5-d5b7-4754-8122-3badaf242b7e/obj.glb", pattern_is_dir=False),
                scale=1.2,
                pos=(0.88, 0.0, 0.76 + 0.0695 * 1.2),
                euler=(0, 90, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=10.0, friction=1.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["hollow roll"] = {"entity": _e}
        # Code Block: long rod
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.52, -0.15, 0.76 + 0.01), euler=(0, 90, 0), size=(0.02, 0.02, 0.40), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=50.0, friction=1.0),
            surface=gs.surfaces.Smooth(color=(0.8, 0.7, 0.5), double_sided=False),
        )
        self._entities["long rod"] = {"entity": _e}
        # Code Block: green target area
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.62, 0, 0.761), euler=(0.0, 0.0, 0.0), size=(0.18, 0.18, 0.002), fixed=True, collision=False
            ),
            material=gs.materials.Rigid(rho=500.0, friction=1.0),
            surface=gs.surfaces.Default(color=(0.1, 0.8, 0.2), roughness=0.6),
        )
        self._entities["green target area"] = {"entity": _e}
        # Code Block: wire hanger
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/d5d470d4-a3ea-47cc-9aa9-618c932fae80/obj.glb", pattern_is_dir=False),
                scale=0.72,
                pos=(0.52, 0.34, 0.76 + 0.0396),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["wire hanger"] = {"entity": _e}
        # Code Block: body lotion
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/114f331b-ce87-46c6-85a5-c88f5cb77892/obj.glb", pattern_is_dir=False),
                scale=1.167,
                pos=(0.78, -0.25, 0.86),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=400.0),
            surface=gs.surfaces.Smooth(color=(0.9, 0.9, 0.9), double_sided=True),
        )
        self._entities["body lotion"] = {"entity": _e}
        # Code Block: screwdriver
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/9d09746f-8f0d-4f93-b009-3d018ef7e3eb/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.60, 0.21, 0.76 - (-0.0102 * 1.0)),
                euler=(0.0, 0.0, 90.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["screwdriver"] = {"entity": _e}
