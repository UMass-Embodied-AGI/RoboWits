from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

RaisePlatformEnv = _import("gs_gym.envs.robowits.24_raise_platform").RaisePlatformEnv


@register_task("robowits/24_02-v0")
class RaisePlatformMut2Env(RaisePlatformEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (
            ("cup support 1", "cup support 2"),
            ("support cube 1"),
            ("heavy book", "support cube 2"),
            "cardboard tube",
            "kitchen towel",
            "chocolate bar",
            "dice",
            "brown leather wallet",
            "pentel eraser",
        )

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: heavy book
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/196d92f1-10b5-4563-ae7f-0b26b615ce51/obj.glb", pattern_is_dir=False),
                scale=0.7,
                pos=(0.70, -0.32, 0.7742 + 0.03),
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=100),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["heavy book"] = {"entity": _e}
        # Code Block: cup support 1
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/94134b02-c73e-4f2c-ad1d-a00a78160d98/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.64, -0.10, 0.8082),
                euler=(0, 180, 0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["cup support 1"] = {"entity": _e}
        # Code Block: cup support 2
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/94134b02-c73e-4f2c-ad1d-a00a78160d98/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.64, 0.10, 0.8082),
                euler=(180, 0, 0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["cup support 2"] = {"entity": _e}
        # Code Block: support cube 1
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(pos=(0.57, 0.0, 0.3), euler=(0, 0, 0), size=(0.02, 0.02, 0.03), fixed=True, collision=True),
            material=gs.materials.Rigid(rho=200.0, friction=1),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5),
        )
        self._entities["support cube 1"] = {"entity": _e}
        # Code Block: support cube 2
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.68, -0.32, 0.76 + 0.015), euler=(0, 0, 0), size=(0.02, 0.02, 0.03), fixed=True, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5),
        )
        self._entities["support cube 2"] = {"entity": _e}
        # Code Block: cardboard tube
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.02,
                height=0.13,
                pos=(0.65, 0.28, 0.76 + 0.065),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.8),
            surface=gs.surfaces.Default(color=(0.74, 0.66, 0.52), roughness=0.8, ior=1.5),
        )
        self._entities["cardboard tube"] = {"entity": _e}
        # Code Block: kitchen towel
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/ae9c4515-7b90-4555-a2a2-4c33b5bb274d/obj.glb", pattern_is_dir=False),
                scale=0.8,
                pos=(0.32, -0.32, 0.7816),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=150.0, friction=0.9),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["kitchen towel"] = {"entity": _e}
        # Code Block: chocolate bar
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/4b504df0-2e47-4db7-83f7-e8e045dfcbd7/obj.glb", pattern_is_dir=False),
                scale=0.8,
                pos=(0.70, 0.44, 0.7658),
                euler=(0, 0, 20),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.8),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["chocolate bar"] = {"entity": _e}
        # Code Block: dice
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/1b6970cb-cfe0-44db-8057-115a0476602f/obj.glb", pattern_is_dir=False),
                scale=1.333,
                pos=(0.76, 0.49, 0.768),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=400.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["dice"] = {"entity": _e}
        # Code Block: brown leather wallet
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/9c66604f-ba98-458c-8542-87e6852f64bc/obj.glb", pattern_is_dir=False),
                scale=1,
                pos=(0.3, 0.28, 0.76288),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["brown leather wallet"] = {"entity": _e}
        # Code Block: pentel eraser
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/361aa36f-d104-465c-afb1-ec68c2f6611f/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.8, 0.54, 0.7672),
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=600.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["pentel eraser"] = {"entity": _e}
