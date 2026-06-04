from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

AlignChopsticksEnv = _import("gs_gym.envs.robowits.26_align_chopsticks").AlignChopsticksEnv


@register_task("robowits/26_02-v0")
class AlignChopsticksMut2Env(AlignChopsticksEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (
            "pencil",
            (
                "board",
                "first chopstick",
                "second chopstick",
                "third chopstick",
                "fourth chopstick",
                "fifth chopstick",
                "sixth chopstick",
            ),
        )

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: pencil
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/981f426a-18e8-4732-a8dd-141f2acde7a6/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.63, 0.44, 0.7657),
                euler=(0, 0, 90),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["pencil"] = {"entity": _e}
        # Code Block: board
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/e8afda3b-6dea-4bfc-859f-88a35bb623a0/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.54, 0.0, 0.76 + 0.01),
                euler=(0, 0, 0),
                fixed=False,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["board"] = {"entity": _e}
        # Code Block: first chopstick
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/chopstick.glb", pattern_is_dir=False),
                scale=(1.2, 2.8, 2.8),
                pos=(0.335, -0.12, 0.85),
                euler=(0, 14, 0),
                fixed=False,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.7),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["first chopstick"] = {"entity": _e}
        # Code Block: second chopstick
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/chopstick.glb", pattern_is_dir=False),
                scale=(1.2, 2.8, 2.8),
                pos=(0.335, -0.07, 0.85),
                euler=(0, -14, 180),
                fixed=False,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.7),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["second chopstick"] = {"entity": _e}
        # Code Block: third chopstick
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/chopstick.glb", pattern_is_dir=False),
                scale=(1.2, 2.8, 2.8),
                pos=(0.335, -0.02, 0.85),
                euler=(0, 12, 0),
                fixed=False,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.7),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["third chopstick"] = {"entity": _e}
        # Code Block: fourth chopstick
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/chopstick.glb", pattern_is_dir=False),
                scale=(1.2, 2.8, 2.8),
                pos=(0.335, 0.03, 0.80),
                euler=(0, -12, 180),
                fixed=False,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.7),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["fourth chopstick"] = {"entity": _e}
        # Code Block: fifth chopstick
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/chopstick.glb", pattern_is_dir=False),
                scale=(1.2, 2.8, 2.8),
                pos=(0.335, 0.08, 0.80),
                euler=(0, 10, 0),
                fixed=False,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.7),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["fifth chopstick"] = {"entity": _e}
        # Code Block: sixth chopstick
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/chopstick.glb", pattern_is_dir=False),
                scale=(1.2, 2.8, 2.8),
                pos=(0.335, 0.13, 0.80),
                euler=(0, -10, 180),
                fixed=False,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.7),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["sixth chopstick"] = {"entity": _e}
