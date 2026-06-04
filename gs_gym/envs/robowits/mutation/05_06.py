from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

RollUpBallEnv = _import("gs_gym.envs.robowits.05_roll_up_ball").RollUpBallEnv


@register_task("robowits/05_06-v0")
class RollUpBallMut6Env(RollUpBallEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("long board", "box", "high basket", "dustpan")

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
                scale=0.55,
                pos=(0.517, 0.09, 0.806),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.4),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["high basket"] = {"entity": _e}
        # Code Block: long board
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/08893dc4-bfb1-49ca-9e47-4b3958a21e4b/obj.glb", pattern_is_dir=False),
                scale=0.1411,
                pos=(0.49, 0.37, 0.76 + 0.0468 * 0.1411),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.3),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["long board"] = {"entity": _e}
        # Code Block: dustpan
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/96f2c45a-2a8d-4e72-b241-18e199dc96c7/obj.glb", pattern_is_dir=False),
                scale=0.78,
                pos=(0.585, -0.15, 0.76 + 0.0207 * 0.78),
                euler=(0, 0, -90),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.3),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["dustpan"] = {"entity": _e}
        # Code Block: box
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.32, -0.12, 0.76 + 0.05), euler=(0, 0, 0), size=(0.1, 0.1, 0.1), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1),
            surface=gs.surfaces.Smooth(color=(1.0, 0.5, 0.0), double_sided=True),
        )
        self._entities["box"] = {"entity": _e}
