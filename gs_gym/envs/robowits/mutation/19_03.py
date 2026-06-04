from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

StackBowlsEnv = _import("gs_gym.envs.robowits.19_stack_bowls").StackBowlsEnv


@register_task("robowits/19_03-v0")
class StackBowlsMut3Env(StackBowlsEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("large bowl", "medium bowl", "fry pan", "square bowl")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: large bowl
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c652cf0f-d2eb-44bd-9e68-a2ceca698591/obj.glb", pattern_is_dir=False),
                scale=1.69,
                pos=(0.60, 0.00, 0.76 + 0.024 * 1.69),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["large bowl"] = {"entity": _e}
        # Code Block: medium bowl
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c652cf0f-d2eb-44bd-9e68-a2ceca698591/obj.glb", pattern_is_dir=False),
                scale=0.776,
                pos=(0.666, 0.30, 0.76 + 0.024 * 0.776),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["medium bowl"] = {"entity": _e}
        # Code Block: square bowl
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.67, -0.23, 0.785), euler=(0, 0, 0), size=(0.108, 0.12, 0.05), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.95, 0.95, 0.95), roughness=0.5, ior=1.5),
        )
        self._entities["square bowl"] = {"entity": _e}
        # Code Block: fry pan
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/08772cb0-334d-4254-b72c-1abd326751d8/obj.glb", pattern_is_dir=False),
                scale=0.7,
                pos=(0.372, 0.02, 0.76 + 0.0433 * 0.7),
                euler=(0, 0, 15),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["fry pan"] = {"entity": _e}
