from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

CoverWithLidEnv = _import("gs_gym.envs.robowits.13_cover_with_lid").CoverWithLidEnv


@register_task("robowits/13_01-v0")
class CoverWithLidMut1Env(CoverWithLidEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("pot without lid", "small lid", "lid medium", "lid large", "milk pack", "teaspoon")

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
                scale=0.2077,
                pos=(0.60, 0.00, 0.76 + 0.3327 * 0.2077),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["pot without lid"] = {"entity": _e}
        # Code Block: small lid
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c1a05823-5cb6-4e64-b3f3-5fd5a86cfc0b/obj.glb", pattern_is_dir=False),
                scale=(0.1576, 0.1576, 0.5),
                pos=(0.67, -0.145, 0.76 + 0.0482 * 0.1576),
                euler=(0.0, 0.0, 90.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500, friction=1.5),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["small lid"] = {"entity": _e}
        # Code Block: lid medium
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c1a05823-5cb6-4e64-b3f3-5fd5a86cfc0b/obj.glb", pattern_is_dir=False),
                scale=(0.2048, 0.2048, 0.5),
                pos=(0.64, 0.14, 0.76 + 0.0482 * 0.2048),
                euler=(0.0, 0.0, 90.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500, friction=1.5),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["lid medium"] = {"entity": _e}
        # Code Block: lid large
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c1a05823-5cb6-4e64-b3f3-5fd5a86cfc0b/obj.glb", pattern_is_dir=False),
                scale=(0.3152, 0.3152, 0.5),
                pos=(0.40, 0.095, 0.76 + 0.0482 * 0.3152),
                euler=(0.0, 0.0, 90.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500, friction=1.5),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["lid large"] = {"entity": _e}
        # Code Block: milk pack
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/d5cd5a5e-c548-4acd-8219-d685419731ed/obj.glb", pattern_is_dir=False),
                scale=1.321,
                pos=(0.44, -0.06, 0.76 + 0.0606 * 1.321),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["milk pack"] = {"entity": _e}
        # Code Block: teaspoon
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/bec2248d-0f97-4683-89af-ecbb50934af2/obj.glb", pattern_is_dir=False),
                scale=1.139,
                pos=(0.52, -0.17, 0.76 + 0.004 * 1.139),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["teaspoon"] = {"entity": _e}
