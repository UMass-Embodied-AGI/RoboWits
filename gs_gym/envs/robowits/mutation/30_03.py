from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

DifferentiateCubesEnv = _import("gs_gym.envs.robowits.30_differentiate_cubes").DifferentiateCubesEnv


@register_task("robowits/30_03-v0")
class DifferentiateCubesMut3Env(DifferentiateCubesEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (("mug", "wooden cube", "metal cube"), "target area", "eraser", "ruler")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: target area
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.40, 0.00, 0.761), euler=(0, 0, 0), size=(0.16, 0.10, 0.002), fixed=True, collision=False
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.1, 0.8, 0.2), roughness=0.9),
        )
        self._entities["target area"] = {"entity": _e}
        # Code Block: mug
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/94134b02-c73e-4f2c-ad1d-a00a78160d98/obj.glb", pattern_is_dir=False),
                scale=1.26,
                pos=(0.66, 0.00, 0.8201),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=600.0, friction=1.0),
            surface=gs.surfaces.Smooth(color=(0.95, 0.8, 0.6), double_sided=True),
        )
        self._entities["mug"] = {"entity": _e}
        # Code Block: wooden cube
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.66, 0.0, 0.865), euler=(0, 0, 0), size=(0.028, 0.028, 0.028), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=300.0, friction=0.8),
            surface=gs.surfaces.Default(color=(0.9, 0.1, 0.1), roughness=0.6),
        )
        self._entities["wooden cube"] = {"entity": _e}
        # Code Block: metal cube
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.66, 0.03, 0.865), euler=(0, 0, 0), size=(0.028, 0.028, 0.028), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=7800.0, friction=0.6),
            surface=gs.surfaces.Metal(color=(0.95, 0.9, 0.1), double_sided=False, metal_type="iron"),
        )
        self._entities["metal cube"] = {"entity": _e}
        # Code Block: eraser
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/32b6af99-cf89-47b3-9807-7de783bf2d9f/obj.glb", pattern_is_dir=False),
                scale=1.2,
                pos=(0.36, 0.12, 0.7653),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=1100.0, friction=0.9),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["eraser"] = {"entity": _e}
        # Code Block: ruler
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.46, -0.12, 0.763), euler=(0, 0, 0), size=(0.30, 0.03, 0.006), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=800.0, friction=0.6),
            surface=gs.surfaces.Smooth(color=(0.85, 0.75, 0.5), double_sided=True),
        )
        self._entities["ruler"] = {"entity": _e}
