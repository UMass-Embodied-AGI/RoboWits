from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

StabilizeBottleEnv = _import("gs_gym.envs.robowits.22_stabilize_bottle").StabilizeBottleEnv


@register_task("robowits/22_01-v0")
class StabilizeBottleMut1Env(StabilizeBottleEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (("sand_container", "dry_sand"), "tube", "bowl", "coins", "rollerball_pen", "cutting_board")

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
                scale=1.0,
                pos=(0.55, 0.00, 0.8032),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["bowl"] = {"entity": _e}
        # Code Block: sand_container
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/94134b02-c73e-4f2c-ad1d-a00a78160d98/obj.glb", pattern_is_dir=False),
                scale=1.1,
                pos=(0.36, -0.10, 0.8125),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["sand_container"] = {"entity": _e}
        # Code Block: dry_sand
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.36, -0.10, 0.82), euler=(0, 0, 0), radius=0.022, height=0.07, fixed=False, collision=True
            ),
            material=gs.materials.MPM.Sand(rho=1000.0, sampler="random", friction_angle=45),
            surface=gs.surfaces.Default(color=(0.9, 0.8, 0.3), vis_mode="particle"),
        )
        self._entities["dry_sand"] = {"entity": _e}
        # Code Block: tube
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/e620c6a5-0a69-4a70-a9a7-62c91931715e/obj.glb", pattern_is_dir=False),
                scale=2.5,
                pos=(0.66, 0.15, 0.7898),
                euler=(0, 90, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.8),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["tube"] = {"entity": _e}
        # Code Block: coins
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.48, -0.18, 0.763), euler=(0, 0, 0), radius=0.018, height=0.006, fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=7800.0, friction=0.3),
            surface=gs.surfaces.Smooth(color=(0.85, 0.7, 0.25), double_sided=True),
        )
        self._entities["coins"] = {"entity": _e}
        # Code Block: rollerball_pen
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/981f426a-18e8-4732-a8dd-141f2acde7a6/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.42, 0.15, 0.7657),
                euler=(0, 0, 90),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["rollerball_pen"] = {"entity": _e}
        # Code Block: cutting_board
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/f923466c-c37d-4744-b197-6089b2899715/obj.glb", pattern_is_dir=False),
                scale=0.35,
                pos=(0.66, 0.45, 0.7648),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["cutting_board"] = {"entity": _e}
