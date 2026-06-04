from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

AlignBlocksEnv = _import("gs_gym.envs.robowits.01_align_blocks").AlignBlocksEnv


@register_task("robowits/01_01-v0")
class AlignBlocksMut1Env(AlignBlocksEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (
            "first target cube",
            "second target cube",
            "third target cube",
            "long rigid ruler",
            "paper cup",
            "toy duck",
        )

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: first target cube
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.54, 0.02, 0.785), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.6, 0.7, 0.9), roughness=0.5),
        )
        self._entities["first target cube"] = {"entity": _e}
        # Code Block: second target cube
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.60, -0.10, 0.785), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.6, 0.7, 0.9), roughness=0.5),
        )
        self._entities["second target cube"] = {"entity": _e}
        # Code Block: third target cube
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.48, 0.14, 0.785), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.6, 0.7, 0.9), roughness=0.5),
        )
        self._entities["third target cube"] = {"entity": _e}
        # Code Block: long rigid ruler
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/33679be6-fc3f-40e0-ae2b-c6329d2d0ac8/obj.glb", pattern_is_dir=False),
                scale=1.2,
                pos=(0.38, 0.0, 0.76 + 0.0097 * 1.2),
                euler=(0, 0, 90),
                fixed=False,
                collision=True,
                group_by_material=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["long rigid ruler"] = {"entity": _e}
        # Code Block: paper cup
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/69730348-4786-4ef3-b7a5-6d4572e43811/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.68, -0.05, 0.8216),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["paper cup"] = {"entity": _e}
        # Code Block: toy duck
        _e = self._scene.scene.add_entity(
            gs.morphs.Sphere(pos=(0.34, 0.12, 0.79), euler=(0, 0, 0), radius=0.03, fixed=False, collision=True),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.95, 0.9, 0.2), roughness=0.4, ior=1.5),
        )
        self._entities["toy duck"] = {"entity": _e}
