from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

RetrieveCubeEnv = _import("gs_gym.envs.robowits.02_retrieve_cube").RetrieveCubeEnv


@register_task("robowits/02_02-v0")
class RetrieveCubeMut2Env(RetrieveCubeEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (("deep narrow container", "cube"), "target area", "lemon", "paper cup")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: target area
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.68, 0.30, 0.761), euler=(0, 0, 0), size=(0.12, 0.12, 0.002), fixed=True, collision=False
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(color=(0.1, 0.8, 0.1), double_sided=True),
        )
        self._entities["target area"] = {"entity": _e}
        # Code Block: deep narrow container
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/ddfb928e-c26d-4805-b41a-ab1545269e99/obj.glb", pattern_is_dir=False),
                scale=0.60,
                pos=(0.55, 0.00, 0.76 + 0.1332 * 0.60),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=400.0, friction=0.8),
            surface=gs.surfaces.Glass(color=(0.9, 0.95, 1.0), double_sided=True),
        )
        self._entities["deep narrow container"] = {"entity": _e}
        # Code Block: cube
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.55, 0.0, 0.86), euler=(0, 0, 0), size=(0.015, 0.015, 0.015), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=300.0, friction=0.7),
            surface=gs.surfaces.Rough(color=(0.8, 0.6, 0.4), double_sided=True),
        )
        self._entities["cube"] = {"entity": _e}
        # Code Block: lemon
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c45b108b-2163-469c-8ed4-dcb82260d83f/obj.glb", pattern_is_dir=False),
                scale=0.8,
                pos=(0.68, -0.35, 0.76 + 0.0374 * 0.8),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=900.0, friction=0.9),
            surface=gs.surfaces.Smooth(color=(1.0, 0.95, 0.1), double_sided=True),
        )
        self._entities["lemon"] = {"entity": _e}
        # Code Block: paper cup
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/69730348-4786-4ef3-b7a5-6d4572e43811/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.40, 0.10, 0.76 + 0.0616),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=300.0, friction=0.7),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["paper cup"] = {"entity": _e}
