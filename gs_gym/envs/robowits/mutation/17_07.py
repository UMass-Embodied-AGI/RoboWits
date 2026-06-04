from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

BallOntoTowerEnv = _import("gs_gym.envs.robowits.17_ball_onto_tower").BallOntoTowerEnv


@register_task("robowits/17_07-v0")
class BallOntoTowerMut7Env(BallOntoTowerEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (("tower base", "large ring", "medium ring"), "small ring", "ball", "bottle cap", "coaster")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: tower base
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/tower_base.glb", pattern_is_dir=False),
                scale=(1.0, 1.0, 1.5),
                pos=(0.605, 0.0, 0.803),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
                decimate=True,
                decimate_face_num=100,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["tower base"] = {"entity": _e}
        # Code Block: large ring
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/large_ring.glb", pattern_is_dir=False),
                scale=(1.2, 1.2, 1.5),
                pos=(0.605, 0.0, 0.802),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
                decimate=True,
                decimate_face_num=100,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["large ring"] = {"entity": _e}
        # Code Block: medium ring
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/medium_ring.glb", pattern_is_dir=False),
                scale=(1.2, 1.2, 1.5),
                pos=(0.605, 0.0, 0.835),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
                decimate=True,
                decimate_face_num=100,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["medium ring"] = {"entity": _e}
        # Code Block: small ring
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/small_ring.glb", pattern_is_dir=False),
                scale=(1.2, 1.2, 1.5),
                pos=(0.69, 0.08, 0.7704),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
                decimate=True,
                decimate_face_num=100,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["small ring"] = {"entity": _e}
        # Code Block: ball
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb", pattern_is_dir=False),
                scale=0.2,
                pos=(0.55, 0.34, 0.76 + 0.075),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["ball"] = {"entity": _e}
        # Code Block: bottle cap
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.62, -0.18, 0.7635),
                euler=(0.0, 0.0, 0.0),
                radius=0.016,
                height=0.007,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(color=(0.2, 0.4, 0.9), double_sided=True),
        )
        self._entities["bottle cap"] = {"entity": _e}
        # Code Block: coaster
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.045,
                height=0.005,
                pos=(0.75, -0.08, 0.76 + 0.005 / 2.0),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=700.0),
            surface=gs.surfaces.Default(color=(0.72, 0.55, 0.35), roughness=0.6, ior=1.4),
        )
        self._entities["coaster"] = {"entity": _e}
