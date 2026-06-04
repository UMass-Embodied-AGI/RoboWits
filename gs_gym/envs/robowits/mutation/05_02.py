from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

RollUpBallEnv = _import("gs_gym.envs.robowits.05_roll_up_ball").RollUpBallEnv


@register_task("robowits/05_02-v0")
class RollUpBallMut2Env(RollUpBallEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("coffee mug", "box", "high basket", "glass tumbler", "screwdriver")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: high basket
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c6009731-c1d9-48f9-9486-1d5754c336d9/obj.glb", pattern_is_dir=False),
                scale=0.611,
                pos=(0.43, 0.0, 0.806),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["high basket"] = {"entity": _e}
        # Code Block: box
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.40, 0.14, 0.76 + 0.05), euler=(0, 0, 0), size=(0.1, 0.1, 0.1), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1),
            surface=gs.surfaces.Smooth(color=(1.0, 0.5, 0.0), double_sided=True),
        )
        self._entities["box"] = {"entity": _e}
        # Code Block: coffee mug
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/47876be5-5857-400e-85c3-274f171d6a3d/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.7, -0.15, 0.76 - (-0.0392 * 1.0)),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["coffee mug"] = {"entity": _e}
        # Code Block: glass tumbler
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/15df4d88-ca29-414a-98e6-4a557b40d8ae/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.70, 0.10, 0.76 - (-0.0441 * 1.0)),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["glass tumbler"] = {"entity": _e}
        # Code Block: screwdriver
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/9d09746f-8f0d-4f93-b009-3d018ef7e3eb/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.60, 0.18, 0.76 - (-0.0102 * 1.0)),
                euler=(0.0, 0.0, 90.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["screwdriver"] = {"entity": _e}
