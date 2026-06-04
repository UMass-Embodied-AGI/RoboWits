from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

BallOntoTowerEnv = _import("gs_gym.envs.robowits.17_ball_onto_tower").BallOntoTowerEnv


@register_task("robowits/17_03-v0")
class BallOntoTowerMut3Env(BallOntoTowerEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (
            ("tower base", "large ring", "medium ring"),
            "small ring",
            "ball",
            "funnel",
            "mug",
            "organizer bag",
            "glass tumbler",
            "brown leather wallet",
        )

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
        # Code Block: funnel
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/a664c3de-c8d8-4042-ac6b-2eaf2b86c51e/obj.glb", pattern_is_dir=False),
                scale=0.6,
                pos=(0.40, 0.00, 0.96 + (0.0769 * 2) / 2.0),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=300.0, friction=0.7),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["funnel"] = {"entity": _e}
        # Code Block: mug
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/94134b02-c73e-4f2c-ad1d-a00a78160d98/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.81, -0.15, 0.76 + 0.0477),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=600.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["mug"] = {"entity": _e}
        # Code Block: organizer bag
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/de650dd1-95e3-48ae-9368-2635d625b546/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.33, -0.44, 0.76),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=400.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["organizer bag"] = {"entity": _e}
        # Code Block: glass tumbler
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/15df4d88-ca29-414a-98e6-4a557b40d8ae/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.81, 0.0, 0.76 + 0.0441),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=800.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["glass tumbler"] = {"entity": _e}
        # Code Block: brown leather wallet
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/9c66604f-ba98-458c-8542-87e6852f64bc/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.58, 0.44, 0.76 + 0.0024),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=700.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["brown leather wallet"] = {"entity": _e}
