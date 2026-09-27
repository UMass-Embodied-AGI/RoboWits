from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

StandPagesEnv = _import("gs_gym.envs.robowits.07_stand_pages").StandPagesEnv


@register_task("robowits/07_02-v0")
class StandPagesMut2Env(StandPagesEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("page_A_with_eyelets", "page_B_with_eyelets", "high_friction_mat", "tall_support_object")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: high friction mat
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.58, 0.0, 0.7625), euler=(0, 0, 0), size=(0.25, 0.15, 0.005), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(
                rho=200.0, friction=1.3, coup_friction=0.8, coup_softness=0.002, coup_restitution=0.0
            ),
            surface=gs.surfaces.Rough(color=(0.1, 0.1, 0.1), double_sided=True),
        )
        self._entities["high_friction_mat"] = {"entity": _e}
        # Code Block: tall support object
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.62, 0.0, 0.865), euler=(0, 0, 0), size=(0.20, 0.05, 0.20), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(
                rho=200.0, friction=1.0, coup_friction=0.7, coup_softness=0.002, coup_restitution=0.0
            ),
            surface=gs.surfaces.Smooth(color=(0.7, 0.5, 0.3), double_sided=True),
        )
        self._entities["tall_support_object"] = {"entity": _e}
        # Code Block: page A with eyelets
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("toolbench/pageA.glb", pattern_is_dir=False),
                scale=0.6,
                pos=(0.37, 0.0, 0.796),
                euler=(0, 90, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(
                rho=200.0, friction=0.8, coup_friction=0.6, coup_softness=0.002, coup_restitution=0.0
            ),
            surface=gs.surfaces.Smooth(color=(0.7, 0.7, 0.7), double_sided=True),
        )
        self._entities["page_A_with_eyelets"] = {"entity": _e}
        # Code Block: page B with eyelets
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("toolbench/pageB.glb", pattern_is_dir=False),
                scale=0.6,
                pos=(0.665, 0.26, 0.796),
                euler=(0, 90, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(
                rho=200.0, friction=0.8, coup_friction=0.6, coup_softness=0.002, coup_restitution=0.0
            ),
            surface=gs.surfaces.Smooth(color=(0.7, 0.7, 0.7), double_sided=True),
        )
        self._entities["page_B_with_eyelets"] = {"entity": _e}
