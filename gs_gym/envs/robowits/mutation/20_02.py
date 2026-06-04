from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

BallIntoJarEnv = _import("gs_gym.envs.robowits.20_ball_into_jar").BallIntoJarEnv


@register_task("robowits/20_02-v0")
class BallIntoJarMut2Env(BallIntoJarEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("glass jar", "foam ball", "succulent pot", "cube gift box")

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
                scale=0.90,
                pos=(0.54, 0.00, 0.84),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Glass(color=(0.85, 0.9, 0.95), double_sided=True),
        )
        self._entities["glass jar"] = {"entity": _e}
        # Code Block: foam ball
        _e = self._scene.scene.add_entity(
            gs.morphs.Sphere(pos=(0.60, -0.12, 0.84), euler=(0, 0, 0), radius=0.05, fixed=False, collision=True),
            material=gs.materials.MPM.Elastic(E=5e4, nu=0.2, rho=200.0, sampler="pbs"),
            surface=gs.surfaces.Smooth(color=(1.0, 0.9, 0.1), double_sided=True, vis_mode="recon"),
        )
        self._entities["foam ball"] = {"entity": _e}
        # Code Block: succulent pot
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c483b67d-4d32-4a5b-8bd5-1e8a2c2e59ed/obj.glb", pattern_is_dir=False),
                scale=0.20,
                pos=(0.36, 0.425, 0.787),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["succulent pot"] = {"entity": _e}
        # Code Block: cube gift box
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/500ff99c-a876-44e2-a1c4-a53ee55eb036/obj.glb", pattern_is_dir=False),
                scale=0.41,
                pos=(0.70, 0.15, 0.81),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["cube gift box"] = {"entity": _e}
