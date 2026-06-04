from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

SealColanderEnv = _import("gs_gym.envs.robowits.21_seal_colander").SealColanderEnv


@register_task("robowits/21_02-v0")
class SealColanderMut2Env(SealColanderEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (
            ("perforated container", "water"),
            "pitcher",
            "curved holder",
            "wire basket",
            "chess knight",
            "pen stand",
        )

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: perforated container
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/slotted_spoon.glb", pattern_is_dir=False),
                scale=0.8847,
                pos=(0.34, -0.06, 0.779),
                euler=(90, 0, 0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.6),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["perforated container"] = {"entity": _e}
        # Code Block: wire basket
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c6009731-c1d9-48f9-9486-1d5754c336d9/obj.glb", pattern_is_dir=False),
                scale=0.40,
                pos=(0.68, 0.05, 0.76 + 0.0739 * 0.40),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["wire basket"] = {"entity": _e}
        # Code Block: pitcher
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c242567e-052d-4561-b2c0-2fed8a5e576b/obj.glb", pattern_is_dir=False),
                scale=1.25,
                pos=(0.60, -0.10, 0.76 + 0.0721 * 1.25),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["pitcher"] = {"entity": _e}
        # Code Block: curved holder
        _e = self._scene.scene.add_entity(
            gs.morphs.Sphere(radius=0.012, pos=(0.45, -0.3, 0.772), euler=(0, 0, 0), fixed=True, collision=True),
            material=gs.materials.Rigid(rho=200.0, friction=0.6),
            surface=gs.surfaces.Smooth(color=(0.85, 0.85, 0.85), double_sided=True),
        )
        self._entities["curved holder"] = {"entity": _e}
        # Code Block: water
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.60, -0.10, 0.82), euler=(0, 0, 0), radius=0.045, height=0.08, fixed=False, collision=True
            ),
            material=gs.materials.MPM.Liquid(E=1e6, nu=0.2, rho=1000.0, viscous=False, sampler="pbs"),
            surface=gs.surfaces.Default(color=(0.6, 0.7, 1.0), double_sided=True, vis_mode="recon"),
        )
        self._entities["water"] = {"entity": _e}
        # Code Block: pen stand
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/eec9c9bc-5967-4096-9274-a6070e92aa6a/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.76, 0.42, 0.76 + 0.015),
                euler=(0, 20, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=600.0, friction=0.6),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["pen stand"] = {"entity": _e}
        # Code Block: chess knight
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/b349c584-2667-46ff-94e2-3b598d7a362f/obj.glb", pattern_is_dir=False),
                scale=0.5,
                pos=(0.76, -0.45, 0.76 + 0.1086 * 0.5),
                euler=(0, 0, 20),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=700.0, friction=0.7),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["chess knight"] = {"entity": _e}
