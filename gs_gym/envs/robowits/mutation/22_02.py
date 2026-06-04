from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

StabilizeBottleEnv = _import("gs_gym.envs.robowits.22_stabilize_bottle").StabilizeBottleEnv


@register_task("robowits/22_02-v0")
class StabilizeBottleMut2Env(StabilizeBottleEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (("sand_container", "dry_sand"), "tube", "bowl", "body_lotion", "packing_peanuts")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: bowl
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/d8dd7f3f-103d-4daf-b579-188178dc4d9e/obj.glb", pattern_is_dir=False),
                scale=1.7,
                pos=(0.60, 0.00, 0.76 - (-0.024) * 1.7),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=400.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["bowl"] = {"entity": _e}
        # Code Block: tube
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/e620c6a5-0a69-4a70-a9a7-62c91931715e/obj.glb", pattern_is_dir=False),
                scale=2.6,
                pos=(0.50, 0.15, 0.76 + 0.0119 * 2.6),
                euler=(0, 90, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=600.0, friction=0.8),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["tube"] = {"entity": _e}
        # Code Block: sand_container
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/94134b02-c73e-4f2c-ad1d-a00a78160d98/obj.glb", pattern_is_dir=False),
                scale=1.15,
                pos=(0.44, -0.10, 0.76 - (-0.0477) * 1.15),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500.0, friction=0.8),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["sand_container"] = {"entity": _e}
        # Code Block: dry_sand
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.44, -0.10, 0.815),
                euler=(0, 0, 0),
                radius=0.018,
                height=0.06,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.MPM.Sand(rho=1200.0, sampler="random", friction_angle=45),
            surface=gs.surfaces.Default(color=(0.9, 0.8, 0.3), vis_mode="particle"),
        )
        self._entities["dry_sand"] = {"entity": _e}
        # Code Block: packing_peanuts
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/0c12fdab-3af8-4827-a605-2a1727e798e0/obj.glb", pattern_is_dir=False),
                scale=0.8,
                pos=(0.7, 0.4, 0.82),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=600.0, friction=0.7),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["packing_peanuts"] = {"entity": _e}
        # Code Block: body_lotion
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/114f331b-ce87-46c6-85a5-c88f5cb77892/obj.glb", pattern_is_dir=False),
                scale=1.05,
                pos=(0.35, -0.05, 0.76 - (-0.0857) * 1.05),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=600.0, friction=0.7),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["body_lotion"] = {"entity": _e}
