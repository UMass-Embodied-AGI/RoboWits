from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

RaisePlatformEnv = _import("gs_gym.envs.robowits.24_raise_platform").RaisePlatformEnv


@register_task("robowits/24_03-v0")
class RaisePlatformMut3Env(RaisePlatformEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (
            ("cup support 1", "cup support 2"),
            ("board deck", "support cube 1"),
            ("heavy book", "support cube 2"),
            "support can",
        )

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: cup support 1
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/94134b02-c73e-4f2c-ad1d-a00a78160d98/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.34, 0.08, 0.8090),
                euler=(180, 0, 0),
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
                pos=(0.54, 0.08, 0.8090),
                euler=(180, 0, 0),
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["cup support 2"] = {"entity": _e}
        # Code Block: board deck
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.57, -0.2, 0.763 + 0.06), euler=(0, 0, 0), size=(0.30, 0.20, 0.006), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=1000.0, friction=0.7),
            surface=gs.surfaces.Rough(roughness=0.7),
        )
        self._entities["board deck"] = {"entity": _e}
        # Code Block: support cube 1
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.57, -0.2, 0.79), euler=(0, 0, 0), size=(0.06, 0.06, 0.06), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5),
        )
        self._entities["support cube 1"] = {"entity": _e}
        # Code Block: support can
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/5ca78aac-f60f-43b8-8ab7-a6c668a3527d/obj.glb", pattern_is_dir=False),
                scale=0.7,
                pos=(0.44, 0.16, 0.8147),
                euler=(0, 0, 0),
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["support can"] = {"entity": _e}
        # Code Block: heavy book
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/196d92f1-10b5-4563-ae7f-0b26b615ce51/obj.glb", pattern_is_dir=False),
                scale=0.7,
                pos=(0.76, 0.00, 0.7742 + 0.06),
                euler=(90.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=700.0, friction=0.9),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["heavy book"] = {"entity": _e}
        # Code Block: support cube 2
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.76, 0.00, 0.79), euler=(0, 0, 0), size=(0.06, 0.06, 0.06), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5),
        )
        self._entities["support cube 2"] = {"entity": _e}
