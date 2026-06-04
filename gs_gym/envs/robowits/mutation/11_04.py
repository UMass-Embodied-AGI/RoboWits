from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

PlaceTallBoxEnv = _import("gs_gym.envs.robowits.11_place_tall_box").PlaceTallBoxEnv


@register_task("robowits/11_04-v0")
class PlaceTallBoxMut4Env(PlaceTallBoxEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (
            "tall cardboard box",
            ("support block left", "support block right", "beam plank", "target area"),
            "plastic cutting board",
            "cooling rack",
        )

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: support block left
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.55, -0.15, 0.8025), euler=(0, 0, 0), size=(0.10, 0.06, 0.085), fixed=True, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.6, 0.6, 0.6), roughness=0.6, ior=1.5),
        )
        self._entities["support block left"] = {"entity": _e}
        # Code Block: support block right
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.55, 0.15, 0.8025), euler=(0, 0, 0), size=(0.10, 0.06, 0.085), fixed=True, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.6, 0.6, 0.6), roughness=0.6, ior=1.5),
        )
        self._entities["support block right"] = {"entity": _e}
        # Code Block: beam plank
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(pos=(0.55, 0.0, 0.855), euler=(0, 0, 0), size=(0.12, 0.36, 0.02), fixed=True, collision=True),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.7, 0.55, 0.3), roughness=0.7, ior=1.5),
        )
        self._entities["beam plank"] = {"entity": _e}
        # Code Block: target area
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.55, 0.0, 0.761), euler=(0, 0, 0), size=(0.10, 0.12, 0.002), fixed=True, collision=False
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.2, 0.8, 0.2), roughness=0.4, ior=1.5),
        )
        self._entities["target area"] = {"entity": _e}
        # Code Block: tall cardboard box
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/8fb31a9b-33d5-4246-997f-84307520c1a0/obj.glb", pattern_is_dir=False),
                scale=1.2,
                pos=(0.46, 0.08, 0.76 + 0.0685 * 1.2),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["tall cardboard box"] = {"entity": _e}
        # Code Block: plastic cutting board
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/fbf1c6fc-4045-44a3-9f45-882d134ccb0c/obj.glb", pattern_is_dir=False),
                scale=0.7,
                pos=(0.39, 0.25, 0.76 + (0.0059 * 0.7)),
                euler=(0, 0, 90),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["plastic cutting board"] = {"entity": _e}
        # Code Block: cooling rack
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.4, -0.10, 0.77), euler=(0, 0, 0), size=(0.18, 0.18, 0.02), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(color=(0.85, 0.85, 0.85), double_sided=True),
        )
        self._entities["cooling rack"] = {"entity": _e}
