from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

RetrieveCubeEnv = _import("gs_gym.envs.robowits.02_retrieve_cube").RetrieveCubeEnv


@register_task("robowits/02_01-v0")
class RetrieveCubeMut1Env(RetrieveCubeEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (
            ("wide open container", "cube"),
            "target area",
            "ikea plate",
            "tv remote control",
            "stylized copper kettle",
        )

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: wide open container
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/d8dd7f3f-103d-4daf-b579-188178dc4d9e/obj.glb", pattern_is_dir=False),
                scale=1.27,
                pos=(0.42, 0.00, 0.7905),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["wide open container"] = {"entity": _e}
        # Code Block: target area
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.62, 0.00, 0.761), euler=(0.0, 0.0, 0.0), size=(0.12, 0.12, 0.002), fixed=True, collision=False
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(color=(0.1, 0.8, 0.1), double_sided=True),
        )
        self._entities["target area"] = {"entity": _e}
        # Code Block: cube
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.42, 0.00, 0.79), euler=(0.0, 0.0, 0.0), size=(0.015, 0.015, 0.015), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(color=(0.8, 0.6, 0.4), double_sided=True),
        )
        self._entities["cube"] = {"entity": _e}
        # Code Block: ikea plate
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/a0970e73-f2c7-4c82-a28a-a5e47e70c96f/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.34, -0.44, 0.7694),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["ikea plate"] = {"entity": _e}
        # Code Block: tv remote control
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/e3a03694-8872-4428-b864-836b1dfa6614/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.66, -0.30, 0.7735),
                euler=(0.0, 0.0, 90.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["tv remote control"] = {"entity": _e}
        # Code Block: stylized copper kettle
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/9db8695f-8b03-4013-95f2-00ac41a8fa49/obj.glb", pattern_is_dir=False),
                scale=0.8,
                pos=(0.70, 0.22, 0.835),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Metal(color=(0.95, 0.6, 0.2), double_sided=True, metal_type="copper"),
        )
        self._entities["stylized copper kettle"] = {"entity": _e}
