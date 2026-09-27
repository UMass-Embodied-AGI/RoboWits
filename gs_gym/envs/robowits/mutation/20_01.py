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
        return ("glass_jar", "foam_ball", "pestle", "decorated_candle")

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
                scale=0.80,
                pos=(0.55, 0.00, 0.8350),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=2500.0, friction=0.8),
            surface=gs.surfaces.Glass(color=(0.85, 0.9, 0.95), double_sided=True),
        )
        self._entities["glass_jar"] = {"entity": _e}
        # Code Block: foam ball
        _e = self._scene.scene.add_entity(
            gs.morphs.Sphere(
                pos=(0.45, -0.12, 0.8420),
                radius=0.06,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.MPM.Elastic(E=5e4, nu=0.2, rho=200.0, sampler="pbs"),
            surface=gs.surfaces.Smooth(color=(1.0, 0.9, 0.1), double_sided=True, vis_mode="recon"),
        )
        self._entities["foam_ball"] = {"entity": _e}
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
        self._entities["decorated_candle"] = {"entity": _e}
