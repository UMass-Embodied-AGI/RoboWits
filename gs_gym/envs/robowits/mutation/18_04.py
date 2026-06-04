from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

CylinderThroughHoleEnv = _import("gs_gym.envs.robowits.18_cylinder_through_hole").CylinderThroughHoleEnv


@register_task("robowits/18_04-v0")
class CylinderThroughHoleMut4Env(CylinderThroughHoleEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("cylindrical_peg", ("collection_zone", "hole_plate"), "hammer")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: collection_zone
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.55, 0.0, 0.761), euler=(0, 0, 0), size=(0.22, 0.18, 0.002), fixed=True, collision=False
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.1, 0.8, 0.2), roughness=0.6),
        )
        self._entities["collection_zone"] = {"entity": _e}
        # Code Block: hole_plate
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=gs.options.CoacdOptions(
                    threshold=0.01, preprocess_resolution=200, max_convex_hull=50, decimate=True
                ),
                file=get_asset_path("hf_assets/holeplate.glb", pattern_is_dir=False),
                scale=(1.2, 1.0, 1.2),
                pos=(0.55, 0.0, 0.7823),
                euler=(0, 90, -90),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(color=(0.95, 0.95, 0.95), double_sided=True),
        )
        self._entities["hole_plate"] = {"entity": _e}
        # Code Block: cylindrical_peg
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/ecfc80a0-4318-4ac8-8ec0-fa1e355d1521/obj.glb", pattern_is_dir=False),
                scale=0.9,
                pos=(0.66, -0.25, 0.8278),
                euler=(0, 90, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.8),
            surface=gs.surfaces.Smooth(color=(0.8, 0.8, 0.0), double_sided=True),
        )
        self._entities["cylindrical_peg"] = {"entity": _e}
        # Code Block: hammer
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/ab77c172-c69f-41b5-956a-600ea7b64c73/obj.glb", pattern_is_dir=False),
                scale=0.32,
                pos=(0.68, -0.12, 0.78),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=800.0, friction=0.6),
            surface=gs.surfaces.Metal(color=(0.6, 0.6, 0.6), double_sided=True, metal_type="iron"),
        )
        self._entities["hammer"] = {"entity": _e}
