from importlib import import_module as _import

import genesis as gs
import numpy as np

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

_wim = _import("gs_gym.envs.robowits.25_water_into_mug")
WaterIntoMugEnv = _wim.WaterIntoMugEnv
mug_pos_relative_to_pitcher = _wim.mug_pos_relative_to_pitcher


@register_task("robowits/25_02-v0")
class WaterIntoMugMut2Env(WaterIntoMugEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (("pitcher", "water", "mug"), "heavy_large_object", "chinese chopsticks", "ping pong ball")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: pitcher
        _pitcher_yaw_deg = 0.0
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/pitcher.glb", pattern_is_dir=False),
                scale=1.525,
                pos=(0.55, 0.0, 0.8715),
                euler=(0.0, 0.0, _pitcher_yaw_deg),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=1000.0, friction=0.8, coup_friction=0.2, coup_softness=0.01),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        # Δeuler = 0 - (-90) = +90, so rel offsets in +y rotate onto -x
        _px, _py = 0.55, 0.0
        self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(_px - 0.02, _py, 0.82),
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
                    pos=(_px - 0.015 + 0.09 * cos_rot, _py + 0.09 * sin_rot, 0.97 - 0.08),
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
                pos=(_px - 0.015 + 0.09 * cos_rot, _py + 0.09 * sin_rot, 0.95 - 0.08),
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
            material=gs.materials.Rigid(rho=800.0, friction=0.9),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["mug"] = {"entity": _e}
        # Code Block: heavy large object
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/rockman.glb", pattern_is_dir=False),
                scale=1,
                pos=(0.36, -0.13, 0.793),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500.0, friction=0.8),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["heavy large object"] = {"entity": _e}
        # Code Block: ping pong ball
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb", pattern_is_dir=False),
                scale=0.267,
                pos=(0.46, -0.11, 0.76 + 0.075 * 0.267),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=120.0, friction=0.4),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["ping pong ball"] = {"entity": _e}
        # Code Block: chinese chopsticks
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/471281c7-80f0-4cfa-a991-6013f30794de/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.68, -0.41, 0.7677),
                euler=(0, 0, 90),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=700.0, friction=0.7),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["chinese chopsticks"] = {"entity": _e}
        # Code Block: water
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.06,
                height=0.3,
                pos=(0.55 + 0.01, 0.0, 0.9915),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.SPH.Liquid(
                rho=1000.0, stiffness=5000.0, exponent=7.0, mu=0.01, gamma=0.02, sampler="pbs"
            ),
            surface=gs.surfaces.Glass(color=(0.6, 0.85, 1.0), double_sided=True, vis_mode="recon"),
        )
        self._entities["water"] = {"entity": _e}
