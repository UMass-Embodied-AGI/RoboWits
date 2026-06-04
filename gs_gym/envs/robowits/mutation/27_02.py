from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

RetrieveRollEnv = _import("gs_gym.envs.robowits.27_retrieve_roll").RetrieveRollEnv


@register_task("robowits/27_02-v0")
class RetrieveRollMut2Env(RetrieveRollEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (
            "green target area",
            "long rod",
            "hollow roll",
            "short ruler",
            "rollerball pen",
            "pen holder",
            "basic funnel",
            "calendar cube decoration",
        )

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: green target area
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.44, 0.0, 0.7625), euler=(0, 0, 0), size=(0.20, 0.15, 0.005), fixed=True, collision=False
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.1, 0.8, 0.2), roughness=0.6),
        )
        self._entities["green target area"] = {"entity": _e}
        # Code Block: long rod
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.51, -0.12, 0.7725), euler=(0, 90, 0), size=(0.02, 0.02, 0.40), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=50.0, friction=1.0),
            surface=gs.surfaces.Smooth(color=(0.8, 0.7, 0.5), double_sided=False),
        )
        self._entities["long rod"] = {"entity": _e}
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
        # Code Block: short ruler
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/33679be6-fc3f-40e0-ae2b-c6329d2d0ac8/obj.glb", pattern_is_dir=False),
                scale=0.47,
                pos=(0.50, -0.18, 0.7697),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                group_by_material=True,
            ),
            material=gs.materials.Rigid(rho=250.0, friction=0.9),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["short ruler"] = {"entity": _e}
        # Code Block: rollerball pen
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/981f426a-18e8-4732-a8dd-141f2acde7a6/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.62, 0.30, 0.7657),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=300.0, friction=0.8),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["rollerball pen"] = {"entity": _e}
        # Code Block: pen holder
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/ad0782ee-ec85-4357-8adc-54271822aa84/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.68, 0.15, 0.8635),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=400.0, friction=0.9),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["pen holder"] = {"entity": _e}
        # Code Block: basic funnel
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/5fbaeb99-7700-45b8-b4f6-61b67fba52c7/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.70, -0.33, 0.8286),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=250.0, friction=0.7),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["basic funnel"] = {"entity": _e}
        # Code Block: calendar cube decoration
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/1f050ce6-e43d-4864-8fac-01b892274eb2/obj.glb", pattern_is_dir=False),
                scale=0.5,
                pos=(0.31, 0.16, 0.7904),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["calendar cube decoration"] = {"entity": _e}
