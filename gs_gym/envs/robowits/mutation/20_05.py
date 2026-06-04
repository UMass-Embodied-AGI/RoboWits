from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

BallIntoJarEnv = _import("gs_gym.envs.robowits.20_ball_into_jar").BallIntoJarEnv


@register_task("robowits/20_05-v0")
class BallIntoJarMut5Env(BallIntoJarEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("glass jar", "foam ball", "spoon", "paper straw")

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
                scale=0.8,
                pos=(0.52, 0.00, 0.825),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=None),
            surface=gs.surfaces.Glass(color=(0.85, 0.9, 0.95), double_sided=True),
        )
        self._entities["glass jar"] = {"entity": _e}
        # Code Block: foam ball
        _e = self._scene.scene.add_entity(
            gs.morphs.Sphere(pos=(0.46, -0.10, 0.84), radius=0.05, fixed=False, collision=True),
            material=gs.materials.MPM.Elastic(E=5e4, nu=0.2, rho=200.0, sampler="pbs"),
            surface=gs.surfaces.Smooth(color=(1.0, 0.9, 0.1), double_sided=True, vis_mode="recon"),
        )
        self._entities["foam ball"] = {"entity": _e}
        # Code Block: spoon
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/fe575974-2c16-4676-b09d-f79c2ce24812/obj.glb", pattern_is_dir=False),
                scale=1.192,
                pos=(0.66, -0.05, 0.7681),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=7800.0, friction=None),
            surface=gs.surfaces.Metal(color=(0.8, 0.8, 0.8), double_sided=True, metal_type="aluminium"),
        )
        self._entities["spoon"] = {"entity": _e}
        # Code Block: paper straw
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.0045, height=0.20, pos=(0.66, 0.08, 0.785), euler=(0, 90, 0), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=700.0, friction=0.5),
            surface=gs.surfaces.Rough(color=(1.0, 0.9, 0.1), roughness=0.75),
        )
        self._entities["paper straw"] = {"entity": _e}
