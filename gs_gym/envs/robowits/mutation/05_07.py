from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

RollUpBallEnv = _import("gs_gym.envs.robowits.05_roll_up_ball").RollUpBallEnv


@register_task("robowits/05_07-v0")
class RollUpBallMut7Env(RollUpBallEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("long board", "box", "high basket", "yoga mat")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: long board
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/08893dc4-bfb1-49ca-9e47-4b3958a21e4b/obj.glb", pattern_is_dir=False),
                scale=(0.1176, 0.2137, 0.5536),
                pos=(0.565, 0.3, 0.77),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["long board"] = {"entity": _e}
        # Code Block: yoga mat
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.6, -0.185, 0.806), euler=(28.0, 0.0, 0.0), size=(0.16, 0.30, 0.01), fixed=True, collision=True
            ),
            material=gs.materials.Rigid(rho=300.0, friction=1.2),
            surface=gs.surfaces.Rough(color=(0.8, 0.1, 0.1), double_sided=True),
        )
        self._entities["yoga mat"] = {"entity": _e}
        # Code Block: box
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.39, -0.09, 0.76 + 0.05), euler=(0, 0, 0), size=(0.1, 0.1, 0.1), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1),
            surface=gs.surfaces.Smooth(color=(1.0, 0.5, 0.0), double_sided=True),
        )
        self._entities["box"] = {"entity": _e}
        # Code Block: high basket
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c6009731-c1d9-48f9-9486-1d5754c336d9/obj.glb", pattern_is_dir=False),
                scale=0.611,
                pos=(0.595, 0.04, 0.806),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=300.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["high basket"] = {"entity": _e}
