from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

StandBulbEnv = _import("gs_gym.envs.robowits.16_stand_bulb").StandBulbEnv


@register_task("robowits/16_07-v0")
class StandBulbMut7Env(StandBulbEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("target patch", "plastic_lid", "tall object with curved bottom", "bottle cap")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: target patch
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.50, 0.00, 0.761), euler=(0, 0, 0), radius=0.12, height=0.002, fixed=True, collision=False
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.1, 0.8, 0.1), roughness=0.6),
        )
        self._entities["target patch"] = {"entity": _e}
        # Code Block: plastic lid
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/d8dd7f3f-103d-4daf-b579-188178dc4d9e/obj.glb", pattern_is_dir=False),
                scale=0.85,
                pos=(0.64, 0.00, 0.7804),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["plastic lid"] = {"entity": _e}
        # Code Block: bottle cap
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.50, -0.15, 0.764), euler=(0, 0, 0), radius=0.015, height=0.008, fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=500.0),
            surface=gs.surfaces.Smooth(color=(0.2, 0.2, 0.2), double_sided=True),
        )
        self._entities["bottle cap"] = {"entity": _e}
        # Code Block: tall object with curved bottom
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/4c5e9552-87c0-46e3-8e40-8c736fee4ff1/obj.glb", pattern_is_dir=False),
                scale=2.0,
                pos=(0.40, 0.15, 0.758),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                decimate=True,
                decimate_face_num=100,
                group_by_material=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["tall object with curved bottom"] = {"entity": _e}
