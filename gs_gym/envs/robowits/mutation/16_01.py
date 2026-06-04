from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

StandBulbEnv = _import("gs_gym.envs.robowits.16_stand_bulb").StandBulbEnv


@register_task("robowits/16_01-v0")
class StandBulbMut1Env(StandBulbEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (
            "stabilizing ring",
            "tall object with curved bottom",
            "target patch",
            "three tier office filing tray",
            "scale ruler",
            "small bowl",
            "body lotion",
        )

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: target patch
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.50, 0.00, 0.761), euler=(0, 0, 0), radius=0.10, height=0.002, fixed=True, collision=False
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.1, 0.8, 0.2), roughness=0.6),
        )
        self._entities["target patch"] = {"entity": _e}
        # Code Block: small bowl
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/d8dd7f3f-103d-4daf-b579-188178dc4d9e/obj.glb", pattern_is_dir=False),
                scale=0.78,
                pos=(0.62, -0.08, 0.76 + 0.5 * (0.048 * 1.2)),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["small bowl"] = {"entity": _e}
        # Code Block: tall object with curved bottom
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/4c5e9552-87c0-46e3-8e40-8c736fee4ff1/obj.glb", pattern_is_dir=False),
                scale=2.0,
                pos=(0.60, 0.2, 0.738),
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
        # Code Block: body lotion
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/114f331b-ce87-46c6-85a5-c88f5cb77892/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.68, -0.32, 0.76 + 0.5 * (0.1714 * 1.0)),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.2, 0.8, 0.3), roughness=0.5),
        )
        self._entities["body lotion"] = {"entity": _e}
        # Code Block: scale ruler
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/33679be6-fc3f-40e0-ae2b-c6329d2d0ac8/obj.glb", pattern_is_dir=False),
                scale=0.8,
                pos=(0.34, -0.22, 0.76 + 0.5 * (0.0194 * 1.0)),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                group_by_material=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["scale ruler"] = {"entity": _e}
        # Code Block: three tier office filing tray
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/a7f533d5-23e2-4994-8f82-e8b0b0c2700c/obj.glb", pattern_is_dir=False),
                scale=0.45,
                pos=(0.3, -0.05, 0.76 + 0.5 * (0.2576 * 1.0)),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["three tier office filing tray"] = {"entity": _e}
