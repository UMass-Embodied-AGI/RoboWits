from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

StandBulbEnv = _import("gs_gym.envs.robowits.16_stand_bulb").StandBulbEnv


@register_task("robowits/16_05-v0")
class StandBulbMut5Env(StandBulbEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("target patch", "stabilizing ring", "tall object with curved bottom", "ceramic_saucer", "small bowl")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: target_patch
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.56, 0.0, 0.761), euler=(0, 0, 0), radius=0.08, height=0.002, fixed=True, collision=False
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.1, 0.8, 0.1), roughness=0.4, ior=1.5),
        )
        self._entities["target_patch"] = {"entity": _e}
        # Code Block: stabilizing ring
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/adbecefa-2e19-4438-b6e1-59e6f4390122/obj.glb", pattern_is_dir=False),
                scale=(0.7, 0.7, 0.8),
                pos=(0.70, 0.43, 0.81),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
                decimate=True,
                decimate_face_num=100,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["stabilizing ring"] = {"entity": _e}
        # Code Block: small_bowl
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/d8dd7f3f-103d-4daf-b579-188178dc4d9e/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.48, -0.10, 0.7840),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["small_bowl"] = {"entity": _e}
        # Code Block: ceramic_saucer
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/9eeac844-2584-473c-bb5f-1908f3e3dbaa/obj.glb", pattern_is_dir=False),
                scale=0.5,
                pos=(0.76, -0.18, 0.790),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["ceramic_saucer"] = {"entity": _e}
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
