from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

RetrieveRollEnv = _import("gs_gym.envs.robowits.27_retrieve_roll").RetrieveRollEnv


@register_task("robowits/27_05-v0")
class RetrieveRollMut5Env(RetrieveRollEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("green target area", "long rod", "hollow roll", "short ruler")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: green_target_area
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.64, 0.0, 0.7625), euler=(0, 0, 0), size=(0.16, 0.16, 0.005), fixed=True, collision=False
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1.0),
            surface=gs.surfaces.Default(color=(0.1, 0.8, 0.2), roughness=0.6),
        )
        self._entities["green_target_area"] = {"entity": _e}
        # Code Block: short_ruler
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/33679be6-fc3f-40e0-ae2b-c6329d2d0ac8/obj.glb", pattern_is_dir=False),
                scale=0.5,
                pos=(0.45, 0.15, 0.7657),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                group_by_material=True,
            ),
            material=gs.materials.Rigid(rho=300.0, friction=0.9),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["short_ruler"] = {"entity": _e}
        # Code Block: hollow roll
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/a0c77eb5-d5b7-4754-8122-3badaf242b7e/obj.glb", pattern_is_dir=False),
                scale=1.2,
                pos=(0.88, 0.0, 0.76 + 0.0695 * 1.2),
                euler=(0, 90, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=10.0, friction=1.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["hollow roll"] = {"entity": _e}
        # Code Block: long rod
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.52, -0.15, 0.76 + 0.01), euler=(0, 90, 0), size=(0.02, 0.02, 0.40), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=50.0, friction=1.0),
            surface=gs.surfaces.Smooth(color=(0.8, 0.7, 0.5), double_sided=False),
        )
        self._entities["long rod"] = {"entity": _e}
