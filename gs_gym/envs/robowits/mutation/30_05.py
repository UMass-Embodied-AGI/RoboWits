from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

DifferentiateCubesEnv = _import("gs_gym.envs.robowits.30_differentiate_cubes").DifferentiateCubesEnv


@register_task("robowits/30_05-v0")
class DifferentiateCubesMut5Env(DifferentiateCubesEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (("mug", "wooden cube", "metal cube"), "target area", ("water", "water pitcher"), "colander")

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
                scale=1.0,
                pos=(0.55, 0.00, 0.76 + 0.0477),
                euler=(0.0, 0.0, 0.0),
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
                pos=(0.55, 0.0, 0.80), euler=(0.0, 0.0, 0.0), size=(0.02, 0.02, 0.02), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=300.0),
            surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0), roughness=0.6),
        )
        self._entities["wooden cube"] = {"entity": _e}
        # Code Block: metal cube
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.55, 0.03, 0.81), euler=(0.0, 0.0, 0.0), size=(0.02, 0.02, 0.02), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=7800.0),
            surface=gs.surfaces.Metal(color=(1.0, 1.0, 0.0), double_sided=False, metal_type="iron"),
        )
        self._entities["metal cube"] = {"entity": _e}
        # Code Block: water pitcher
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c242567e-052d-4561-b2c0-2fed8a5e576b/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.64, 0.10, 0.8321),
                euler=(0.0, 0.0, 90.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=300.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["water pitcher"] = {"entity": _e}
        # Code Block: colander
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c6009731-c1d9-48f9-9486-1d5754c336d9/obj.glb", pattern_is_dir=False),
                scale=0.5,
                pos=(0.41, 0.10, 0.76 + 0.0739 * 0.5),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=300.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["colander"] = {"entity": _e}
        # Code Block: target area
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.40, -0.10, 0.761), euler=(0.0, 0.0, 0.0), size=(0.14, 0.14, 0.002), fixed=True, collision=False
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.1, 0.8, 0.2), roughness=0.9),
        )
        self._entities["target area"] = {"entity": _e}
        # Code Block: water
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.64, 0.10, 0.835), euler=(0.0, 0.0, 0.0), radius=0.02, height=0.05, fixed=False, collision=True
            ),
            material=gs.materials.SPH.Liquid(
                rho=1000.0, stiffness=50000.0, exponent=7.0, mu=0.005, gamma=0.01, sampler="pbs"
            ),
            surface=gs.surfaces.Default(color=(0.6, 0.8, 1.0), roughness=0.2, vis_mode="recon"),
        )
        self._entities["water"] = {"entity": _e}
