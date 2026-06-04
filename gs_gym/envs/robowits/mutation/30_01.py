from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

DifferentiateCubesEnv = _import("gs_gym.envs.robowits.30_differentiate_cubes").DifferentiateCubesEnv


@register_task("robowits/30_01-v0")
class DifferentiateCubesMut1Env(DifferentiateCubesEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (("mug", "wooden cube", "metal cube"), "target area", ("water", "water pitcher"), "cat figurine")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: mug
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/94134b02-c73e-4f2c-ad1d-a00a78160d98/obj.glb", pattern_is_dir=False),
                scale=1.25,
                pos=(0.58, 0.00, 0.76 + 0.0477 * 1.25),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(color=(0.95, 0.8, 0.6), double_sided=True),
        )
        self._entities["mug"] = {"entity": _e}
        # Code Block: wooden cube
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.58, 0.020, 0.845), euler=(0, 0, 0), size=(0.020, 0.020, 0.020), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(1.0, 0.9, 0.1), roughness=0.6),
        )
        self._entities["wooden cube"] = {"entity": _e}
        # Code Block: metal cube
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.58, 0.0, 0.835), euler=(0, 0, 0), size=(0.020, 0.020, 0.020), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=7800.0),
            surface=gs.surfaces.Metal(color=(0.9, 0.1, 0.1), double_sided=False, metal_type="iron"),
        )
        self._entities["metal cube"] = {"entity": _e}
        # Code Block: water pitcher
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c242567e-052d-4561-b2c0-2fed8a5e576b/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.40, -0.05, 0.76 + 0.0721),
                euler=(0, 0, 180),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["water pitcher"] = {"entity": _e}
        # Code Block: target area
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.66, 0.15, 0.7605), euler=(0, 0, 0), size=(0.12, 0.12, 0.001), fixed=True, collision=False
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.1, 0.8, 0.2), roughness=0.9),
        )
        self._entities["target area"] = {"entity": _e}
        # Code Block: cat figurine
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/b349c584-2667-46ff-94e2-3b598d7a362f/obj.glb", pattern_is_dir=False),
                scale=0.35,
                pos=(0.33, 0.44, 0.76 + 0.1086 * 0.35),
                euler=(0, 0, 45),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["cat figurine"] = {"entity": _e}
        # Code Block: pitcher water
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.40, -0.05, 0.83), euler=(0, 0, 0), radius=0.035, height=0.08, fixed=False, collision=True
            ),
            material=gs.materials.MPM.Liquid(rho=1000.0, E=1e6, nu=0.2, viscous=True, sampler="pbs"),
            surface=gs.surfaces.Rough(double_sided=True, vis_mode="recon"),
        )
        self._entities["pitcher water"] = {"entity": _e}
