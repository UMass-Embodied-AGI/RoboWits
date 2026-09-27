from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

CoverWithLidEnv = _import("gs_gym.envs.robowits.13_cover_with_lid").CoverWithLidEnv


@register_task("robowits/13_02-v0")
class CoverWithLidMut2Env(CoverWithLidEnv):
    @property
    def _object_reachable_areas(self) -> dict:
        # The parent exposes this as a property, so extend it rather than assigning.
        # Distractors only have to sit on the table, not within arm reach; "lid_large"
        # is task-critical but its 0.243 m footprint left only 15 mm of slack in x.
        clutter_area = {"x_min": 0.25, "x_max": 0.88, "y_min": -0.62, "y_max": 0.62}
        areas = dict(super()._object_reachable_areas)
        for name in ("coaster_medium", "glass_mug", "olive_oil_bottle", "plate_medium", "table_lamp"):
            areas[name] = clutter_area
        areas["lid_large"] = {"x_min": 0.2, "x_max": 0.75, "y_min": -0.45, "y_max": 0.45}
        return areas

    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (
            "pot_without_lid",
            "small_lid",
            "lid_medium",
            "lid_large",
            "plate_medium",
            "coaster_medium",
            "olive_oil_bottle",
            "table_lamp",
            "glass_mug",
        )

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: pot without lid
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/3aee9353-e21a-40d4-b160-e2a0af1fff7f/obj.glb", pattern_is_dir=False),
                scale=0.2538,
                pos=(0.50, 0.00, 0.8444),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["pot_without_lid"] = {"entity": _e}
        # Code Block: small lid
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c1a05823-5cb6-4e64-b3f3-5fd5a86cfc0b/obj.glb", pattern_is_dir=False),
                scale=(0.1743, 0.1743, 0.5),
                pos=(0.36, -0.10, 0.7684),
                euler=(0.0, 0.0, 90.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500, friction=1.5),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["small_lid"] = {"entity": _e}
        # Code Block: lid medium
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c1a05823-5cb6-4e64-b3f3-5fd5a86cfc0b/obj.glb", pattern_is_dir=False),
                scale=(0.2694, 0.2694, 0.5),
                pos=(0.70, 0.04, 0.7730),
                euler=(0.0, 0.0, 90.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500, friction=1.5),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["lid_medium"] = {"entity": _e}
        # Code Block: lid large
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c1a05823-5cb6-4e64-b3f3-5fd5a86cfc0b/obj.glb", pattern_is_dir=False),
                scale=(0.3803, 0.3803, 0.5),
                pos=(0.70, -0.34, 0.7783),
                euler=(0.0, 0.0, 90.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500, friction=1.5),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["lid_large"] = {"entity": _e}
        # Code Block: plate medium
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/2746f256-610e-4b14-949c-6bd5e2718f5a/obj.glb", pattern_is_dir=False),
                scale=0.6914,
                pos=(0.62, 0.22, 0.7659),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["plate_medium"] = {"entity": _e}
        # Code Block: coaster medium
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.64, -0.02, 0.763), euler=(0, 0, 0), radius=0.085, height=0.006, fixed=True, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(color=(0.55, 0.36, 0.2), double_sided=True),
        )
        self._entities["coaster_medium"] = {"entity": _e}
        # Code Block: olive oil bottle
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/de0009fd-9494-44de-be80-0e093bae4d9a/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.54, 0.44, 0.9210),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["olive_oil_bottle"] = {"entity": _e}
        # Code Block: table lamp
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/b709a919-0de6-4587-a54f-bbb9df553f3c/obj.glb", pattern_is_dir=False),
                scale=0.25,
                pos=(0.62, -0.18, 0.8329),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["table_lamp"] = {"entity": _e}
        # Code Block: glass mug
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/ddfb928e-c26d-4805-b41a-ab1545269e99/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.44, -0.44, 0.8932),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["glass_mug"] = {"entity": _e}
