from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

BallIntoJarEnv = _import("gs_gym.envs.robowits.20_ball_into_jar").BallIntoJarEnv


@register_task("robowits/20_01-v0")
class BallIntoJarMut1Env(BallIntoJarEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("glass jar", "foam ball", "pestle", "decorated candle")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: glass jar
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/f4777d21-c966-40dd-872c-8bf28e00d3ee/obj.glb", pattern_is_dir=False),
                scale=0.9006,
                pos=(0.55, 0.00, 0.8403),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=2500.0, friction=0.8),
            surface=gs.surfaces.Glass(color=(0.85, 0.9, 0.95), double_sided=True),
        )
        self._entities["glass jar"] = {"entity": _e}
        # Code Block: foam ball
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                file=get_asset_path("blender_kit/5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb", pattern_is_dir=False),
                scale=0.5667,
                pos=(0.45, -0.12, 0.86),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.MPM.Elastic(E=2e4, nu=0.2, rho=300.0, sampler="pbs"),
            surface=gs.surfaces.Smooth(color=(1.0, 0.9, 0.1), double_sided=True, vis_mode="recon"),
        )
        self._entities["foam ball"] = {"entity": _e}
        # Code Block: pestle
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.015, height=0.17, pos=(0.66, -0.10, 0.845), euler=(0.0, 0.0, 0.0), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=900.0, friction=0.6),
            surface=gs.surfaces.Smooth(color=(0.8, 0.7, 0.6), double_sided=True),
        )
        self._entities["pestle"] = {"entity": _e}
        # Code Block: decorated candle
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/fe32d5de-dba8-4bdb-a878-22101eb9706a/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.70, 0.18, 0.8126),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500.0, friction=0.7),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["decorated candle"] = {"entity": _e}
