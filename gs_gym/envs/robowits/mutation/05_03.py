from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

RollUpBallEnv = _import("gs_gym.envs.robowits.05_roll_up_ball").RollUpBallEnv


@register_task("robowits/05_03-v0")
class RollUpBallMut3Env(RollUpBallEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("long board", "box", "high basket", "metal ruler", "calendar cube decoration")

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
                pos=(0.595, 0.04, 0.806),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["high basket"] = {"entity": _e}
        # Code Block: long board
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/08893dc4-bfb1-49ca-9e47-4b3958a21e4b/obj.glb", pattern_is_dir=False),
                scale=(0.1097, 0.2140, 0.4026),
                pos=(0.32, -0.155, 0.7836),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["long board"] = {"entity": _e}
        # Code Block: box
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.34, 0.0, 0.76 + 0.05), euler=(0, 0, 0), size=(0.1, 0.1, 0.1), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1),
            surface=gs.surfaces.Smooth(color=(1.0, 0.5, 0.0), double_sided=True),
        )
        self._entities["box"] = {"entity": _e}
        # Code Block: metal ruler
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/33679be6-fc3f-40e0-ae2b-c6329d2d0ac8/obj.glb", pattern_is_dir=False),
                scale=1.1,
                pos=(0.64, -0.12, 0.8699),
                euler=(0.0, -35.0, 0.0),
                fixed=False,
                collision=True,
                group_by_material=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.2),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["metal ruler"] = {"entity": _e}
        # Code Block: calendar cube decoration
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/1f050ce6-e43d-4864-8fac-01b892274eb2/obj.glb", pattern_is_dir=False),
                scale=0.5,
                pos=(0.31, 0.16, 0.7904),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["calendar cube decoration"] = {"entity": _e}
