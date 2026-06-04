from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

GapRetrieveEnv = _import("gs_gym.envs.robowits.03_gap_retrieve").GapRetrieveEnv


@register_task("robowits/03_04-v0")
class GapRetrieveMut4Env(GapRetrieveEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (("lemon", "right boundary block"), "target area", "plastic ruler")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: right boundary block
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.40, 0.031 + 0.025, 0.785 + 0.075),
                euler=(0.0, 0.0, 0.0),
                size=(0.14, 0.05, 0.2),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500.0, friction=1.0),
            surface=gs.surfaces.Rough(color=(0.6, 0.6, 0.6), double_sided=False),
        )
        self._entities["right boundary block"] = {"entity": _e}
        # Code Block: target area
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.67, 0.00, 0.7605),
                euler=(0.0, 0.0, 0.0),
                size=(0.12, 0.10, 0.001),
                fixed=True,
                collision=False,
            ),
            material=gs.materials.Rigid(rho=100.0, friction=0.5),
            surface=gs.surfaces.Default(
                color=(0.1, 0.8, 0.1), roughness=0.9, ior=1.4, double_sided=True, vis_mode="visual"
            ),
        )
        self._entities["target area"] = {"entity": _e}
        # Code Block: lemon
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c45b108b-2163-469c-8ed4-dcb82260d83f/obj.glb", pattern_is_dir=False),
                scale=0.6,
                pos=(0.40, 0.0, 0.76064 + 0.019),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=1200.0, friction=0.8),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["lemon"] = {"entity": _e}
        # Code Block: plastic ruler
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.66, 0.20, 0.7615), euler=(0, 0, 90), size=(0.30, 0.12, 0.003), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=300.0, friction=0.4),
            surface=gs.surfaces.Smooth(color=(0.7, 0.85, 1.0), double_sided=False),
        )
        self._entities["plastic ruler"] = {"entity": _e}
