from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

PinchCardEnv = _import("gs_gym.envs.robowits.04_pinch_card").PinchCardEnv


@register_task("robowits/04_03-v0")
class PinchCardMut3Env(PinchCardEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (
            "bank card",
            "eraser",
            "microfiber cloth",
            "metal ruler",
        )

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: bank card
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/9b4245ca-361f-4940-a23d-090b1e547a52/obj.glb", pattern_is_dir=False),
                scale=0.8,
                pos=(0.46, 0.0, 0.843),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.4),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["bank card"] = {"entity": _e}

        # Code Block: eraser
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/333832bb-68f0-4f72-a300-658c4fdccfdf/obj.glb", pattern_is_dir=False),
                scale=2,
                pos=(0.48, 0.08, 0.85),  # 1.12338),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1.2),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["eraser"] = {"entity": _e}
        # Code Block: small table
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/17674e05-b713-4b2a-a04c-49aee4d4d401/obj.glb", pattern_is_dir=False),
                scale=0.5,
                pos=(0.46, 0, 0.76338),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1.2),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["small table"] = {"entity": _e}
        # Code Block: target cube
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.7, -0.3, 0.86), euler=(0.0, 0.0, 0.0), size=(0.15, 0.15, 0.2), fixed=True, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.4, ior=1.5),
        )
        self._entities["target cube"] = {"entity": _e}
        # Code Block: target area
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.7, -0.3, 0.96), euler=(0.0, 0.0, 0.0), size=(0.15, 0.15, 0.002), fixed=True, collision=False
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.2, 0.8, 0.2), roughness=0.4, ior=1.5),
        )
        self._entities["target area"] = {"entity": _e}
        # Code Block: microfiber cloth
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/ae9c4515-7b90-4555-a2a2-4c33b5bb274d/obj.glb", pattern_is_dir=False),
                scale=(0.6691, 0.7094, 0.07407),
                pos=(0.52, 0.26, 0.7622),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=None),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["microfiber cloth"] = {"entity": _e}
        # Code Block: metal ruler
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.74, 0.00, 0.7615), euler=(0.0, 0.0, 0.0), size=(0.03, 0.30, 0.003), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=None),
            surface=gs.surfaces.Metal(color=(0.7, 0.7, 0.7), double_sided=False, metal_type="aluminium"),
        )
        self._entities["metal ruler"] = {"entity": _e}
