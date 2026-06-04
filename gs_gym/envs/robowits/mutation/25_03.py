from importlib import import_module as _import

import genesis as gs
import numpy as np

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

_wim = _import("gs_gym.envs.robowits.25_water_into_mug")
WaterIntoMugEnv = _wim.WaterIntoMugEnv
mug_pos_relative_to_pitcher = _wim.mug_pos_relative_to_pitcher


@register_task("robowits/25_03-v0")
class WaterIntoMugMut3Env(WaterIntoMugEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (("pitcher", "water", "mug"), "heavy_large_object", "colanderladle", "sponge", "kitchen_towel")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: pitcher
        _pitcher_yaw_deg = 90.0
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/pitcher.glb", pattern_is_dir=False),
                scale=1.525,
                pos=(0.46, 0.0, 0.8715),
                euler=(0.0, 0.0, _pitcher_yaw_deg),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=800.0, friction=1.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        _px, _py = 0.46, 0.0
        self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(_px, _py - 0.02, 0.82),
                radius=0.11,
                height=0.05,
                fixed=True,
                visualization=False,
                collision=True,
            ),
        )
        for rot in [0, 30, 60, 120, 150, 180, 210, 240, 270, 300, 330]:
            sin_rot, cos_rot = np.sin(np.deg2rad(rot + 180)), np.cos(np.deg2rad(rot + 180))
            self._scene.scene.add_entity(
                gs.morphs.Box(
                    pos=(_px + 0.09 * cos_rot, _py - 0.015 + 0.09 * sin_rot, 0.97 - 0.08),
                    euler=(0.0, 0.0, rot),
                    size=(0.01, 0.05, 0.15),
                    fixed=True,
                    visualization=False,
                    collision=True,
                )
            )
        sin_rot, cos_rot = np.sin(np.deg2rad(90 + 180)), np.cos(np.deg2rad(90 + 180))
        self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(_px + 0.09 * cos_rot, _py - 0.015 + 0.09 * sin_rot, 0.95 - 0.08),
                euler=(0.0, 0.0, 90.0),
                size=(0.01, 0.05, 0.1),
                fixed=True,
                visualization=False,
                collision=True,
            )
        )
        self._entities["pitcher"] = {"entity": _e}
        # Code Block: mug (offset from pitcher rotates with pitcher mesh euler Z)
        _pz = 0.8715
        _mug_x, _mug_y, _mug_z = mug_pos_relative_to_pitcher(_px, _py, _pz, _pitcher_yaw_deg)
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/94134b02-c73e-4f2c-ad1d-a00a78160d98/obj.glb", pattern_is_dir=False),
                scale=0.85,
                pos=(_mug_x, _mug_y, _mug_z),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=2400.0, friction=1.2),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["mug"] = {"entity": _e}
        # Code Block: heavy_large_object
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/rockman.glb", pattern_is_dir=False),
                scale=1.3,
                pos=(0.72, -0.38, 0.81),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500.0, friction=0.8),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["heavy_large_object"] = {"entity": _e}
        # Code Block: ladle
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/d6d7aa32-40ba-4c63-9ef4-1f33138cc47d/obj.glb", pattern_is_dir=False),
                scale=0.7,
                pos=(0.34, -0.01, 0.76 + 0.0347),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=7800.0, friction=0.6),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["ladle"] = {"entity": _e}
        # Code Block: colander
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c6009731-c1d9-48f9-9486-1d5754c336d9/obj.glb", pattern_is_dir=False),
                scale=0.46,
                pos=(0.35, -0.38, 0.794),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=900.0, friction=0.8),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["colander"] = {"entity": _e}
        # Code Block: kitchen_towel
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/ae9c4515-7b90-4555-a2a2-4c33b5bb274d/obj.glb", pattern_is_dir=False),
                scale=0.7,
                pos=(0.38, 0.30, 0.787),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=300.0, friction=0.9),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["kitchen_towel"] = {"entity": _e}
        # Code Block: sponge
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/a939eb10-a16c-4e6e-b766-0805109143d4/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.72, -0.12, 0.779),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.9),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["sponge"] = {"entity": _e}
        # Code Block: water
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.06, height=0.3, pos=(0.46, 0.01, 0.9915), euler=(0.0, 0.0, 0.0), fixed=False, collision=True
            ),
            material=gs.materials.SPH.Liquid(
                rho=1000.0, stiffness=5000.0, exponent=7.0, mu=0.01, gamma=0.02, sampler="pbs"
            ),
            surface=gs.surfaces.Glass(color=(0.6, 0.85, 1.0), double_sided=True, vis_mode="recon"),
        )
        self._entities["water"] = {"entity": _e}
