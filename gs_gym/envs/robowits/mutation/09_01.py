from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

HoldCupEnv = _import("gs_gym.envs.robowits.09_hold_cup").HoldCupEnv


@register_task("robowits/09_01-v0")
class HoldCupMut1Env(HoldCupEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("cup", "coaster", ("slope", "target area"), "automotive funnel", "calendar cube decoration")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: slope
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/ab77c172-c69f-41b5-956a-600ea7b64c73/obj.glb", pattern_is_dir=False),
                scale=(0.5, 1.5, 0.6),
                pos=(0.58, 0.0, 0.76 + 0.0586 * 0.5),
                euler=(0.0, 0.0, 90.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.3),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["slope"] = {"entity": _e}
        # Code Block: cup
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/3d998505-6bbb-4cc2-8359-c147ac531430/obj.glb", pattern_is_dir=False),
                scale=1.1,
                pos=(0.35, 0, 0.76 + 0.0459 * 1.1),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.2, coup_friction=0.05),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["cup"] = {"entity": _e}
        # Code Block: coaster
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.42, 0.14, 0.76 + 0.01),
                euler=(0.0, 0.0, 0.0),
                radius=0.04,
                height=0.02,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1.5),
            surface=gs.surfaces.Default(color=(0.6, 0.4, 0.2), roughness=0.7),
        )
        self._entities["coaster"] = {"entity": _e}
        # Code Block: target area
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.60, 0.0, 0.76 + 0.0586 * 0.8),
                euler=(-12.0, 0.0, 0.0),
                size=(0.13, 0.16, 0.002),
                fixed=True,
                collision=False,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.2, 0.8, 0.3), roughness=0.2),
        )
        self._entities["target area"] = {"entity": _e}
        # Code Block: automotive funnel
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/a664c3de-c8d8-4042-ac6b-2eaf2b86c51e/obj.glb", pattern_is_dir=False),
                scale=0.576,
                pos=(0.44, -0.15, 0.76 + 0.0769 * 0.576),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(color=(0.95, 0.95, 0.95), double_sided=True),
        )
        self._entities["automotive funnel"] = {"entity": _e}
        # Code Block: calendar cube decoration
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/1f050ce6-e43d-4864-8fac-01b892274eb2/obj.glb", pattern_is_dir=False),
                scale=0.611,
                pos=(0.75, 0.2, 0.76 + (0.0609 * 0.611)),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(color=(0.8, 0.7, 0.5), double_sided=True),
        )
        self._entities["calendar cube decoration"] = {"entity": _e}
