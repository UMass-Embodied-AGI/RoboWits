from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

DominosEnv = _import("gs_gym.envs.robowits.06_dominos").DominosEnv


@register_task("robowits/06_01-v0")
class DominosMut1Env(DominosEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (
            "white block 4",
            ("white block 3", "white block 2", "white block 1", "red block", "target area"),
            "ruler",
            "tv remote control",
        )

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: red block
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.915, 0.00, 0.805), euler=(0, 0, 0), size=(0.02, 0.05, 0.09), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=500.0),
            surface=gs.surfaces.Default(color=(0.9, 0.1, 0.1), roughness=0.6),
        )
        self._entities["red block"] = {"entity": _e}
        # Code Block: target area
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.960, 0.00, 0.761), euler=(0, 0, 0), size=(0.06, 0.10, 0.002), fixed=True, collision=False
            ),
            material=gs.materials.Rigid(rho=100.0),
            surface=gs.surfaces.Default(color=(0.1, 0.7, 0.2), roughness=0.8),
        )
        self._entities["target area"] = {"entity": _e}
        # Code Block: white block 1
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.855, 0.00, 0.805), euler=(0, 0, 0), size=(0.02, 0.05, 0.09), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=500.0),
            surface=gs.surfaces.Default(color=(0.95, 0.95, 0.95), roughness=0.4),
        )
        self._entities["white block 1"] = {"entity": _e}
        # Code Block: white block 2
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.795, 0.00, 0.805), euler=(0, 0, 0), size=(0.02, 0.05, 0.09), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=500.0),
            surface=gs.surfaces.Default(color=(0.95, 0.95, 0.95), roughness=0.4),
        )
        self._entities["white block 2"] = {"entity": _e}
        # Code Block: white block 3
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.735, 0.0, 0.805), euler=(0, 0, 0), size=(0.02, 0.05, 0.09), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=500.0),
            surface=gs.surfaces.Default(color=(0.95, 0.95, 0.95), roughness=0.4),
        )
        self._entities["white block 3"] = {"entity": _e}
        # Code Block: white block 4
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.50, 0.00, 0.805), euler=(0, 0, 0), size=(0.02, 0.05, 0.09), fixed=True, collision=True
            ),
            material=gs.materials.Rigid(rho=500.0),
            surface=gs.surfaces.Default(color=(0.95, 0.95, 0.95), roughness=0.4),
        )
        self._entities["white block 4"] = {"entity": _e}
        # Code Block: ruler
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/33679be6-fc3f-40e0-ae2b-c6329d2d0ac8/obj.glb", pattern_is_dir=False),
                scale=(0.9375, 1.0, 1.0),
                pos=(0.50, -0.05, 0.7707),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                group_by_material=True,
            ),
            material=gs.materials.Rigid(rho=700.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["ruler"] = {"entity": _e}
        # Code Block: tv remote control
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/e3a03694-8872-4428-b864-836b1dfa6614/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.46, -0.15, 0.7735),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=600.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["tv remote control"] = {"entity": _e}
