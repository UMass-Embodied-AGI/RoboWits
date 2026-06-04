from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

StabilizeBottleEnv = _import("gs_gym.envs.robowits.22_stabilize_bottle").StabilizeBottleEnv


@register_task("robowits/22_03-v0")
class StabilizeBottleMut3Env(StabilizeBottleEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("block A", "tube", "bowl", "block B", "block C")

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
                scale=1.551,
                pos=(0.66, 0.0, 0.797224),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["bowl"] = {"entity": _e}
        # Code Block: tube
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/e620c6a5-0a69-4a70-a9a7-62c91931715e/obj.glb", pattern_is_dir=False),
                scale=2.124,
                pos=(0.50, 0.35, 0.76 + 0.0119 * 2.124),
                euler=(0.0, 90.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["tube"] = {"entity": _e}
        # Code Block: block A
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.36, -0.435, 0.7675),
                euler=(0.0, 0.0, 0.0),
                size=(0.075, 0.025, 0.015),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(color=(0.76, 0.60, 0.42), double_sided=True),
        )
        self._entities["block A"] = {"entity": _e}
        # Code Block: block B
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.44, 0.435, 0.7675),
                euler=(0.0, 0.0, 0.0),
                size=(0.075, 0.025, 0.015),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(color=(0.76, 0.60, 0.42), double_sided=True),
        )
        self._entities["block B"] = {"entity": _e}
        # Code Block: block C
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.62, -0.14, 0.7675),
                euler=(0.0, 0.0, 0.0),
                size=(0.075, 0.025, 0.015),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(color=(0.76, 0.60, 0.42), double_sided=True),
        )
        self._entities["block C"] = {"entity": _e}
