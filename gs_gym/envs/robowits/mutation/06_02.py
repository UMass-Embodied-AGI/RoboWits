from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

DominosEnv = _import("gs_gym.envs.robowits.06_dominos").DominosEnv


@register_task("robowits/06_02-v0")
class DominosMut2Env(DominosEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (
            "white block 4",
            ("white block 3", "white block 2", "white block 1", "red block", "target area"),
            "screw",
            "japanese vase",
            "coffee mug",
            "tennis ball",
        )

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: white block 3
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.735, 0.00, 0.805), euler=(0, 0, 0), size=(0.02, 0.05, 0.09), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.95, 0.95, 0.95), roughness=0.6, ior=1.4),
        )
        self._entities["white block 3"] = {"entity": _e}
        # Code Block: white block 2
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.795, 0.00, 0.805), euler=(0, 0, 0), size=(0.02, 0.05, 0.09), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.95, 0.95, 0.95), roughness=0.6, ior=1.4),
        )
        self._entities["white block 2"] = {"entity": _e}
        # Code Block: white block 1
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.855, 0.00, 0.805), euler=(0, 0, 0), size=(0.02, 0.05, 0.09), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.95, 0.95, 0.95), roughness=0.6, ior=1.4),
        )
        self._entities["white block 1"] = {"entity": _e}
        # Code Block: red block
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.915, 0.00, 0.805), euler=(0, 0, 0), size=(0.02, 0.05, 0.09), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.9, 0.1, 0.1), roughness=0.5, ior=1.4),
        )
        self._entities["red block"] = {"entity": _e}
        # Code Block: target area
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.960, 0.00, 0.761),
                euler=(0, 0, 0),
                size=(0.06, 0.10, 0.002),
                fixed=True,
                collision=False,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(color=(0.1, 0.7, 0.2), double_sided=True),
        )
        self._entities["target area"] = {"entity": _e}
        # Code Block: japanese vase
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/b6cde4f1-bd96-4bd4-84d5-228c86d9d8ff/obj.glb", pattern_is_dir=False),
                scale=0.753,
                pos=(0.55, -0.15, 0.76 + (0.239 * 0.753) / 2),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["japanese vase"] = {"entity": _e}
        # Code Block: coffee mug
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/47876be5-5857-400e-85c3-274f171d6a3d/obj.glb", pattern_is_dir=False),
                scale=1.276,
                pos=(0.35, 0.05, 0.76 + (0.0784 * 1.276) / 2),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["coffee mug"] = {"entity": _e}
        # Code Block: screw
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/1d373cbe-c73d-4929-bf32-e85a98dc4bca/obj.glb", pattern_is_dir=False),
                scale=2.703,
                pos=(0.40, 0.15, 0.76 + (0.0148 * 2.703) / 2),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["screw"] = {"entity": _e}
        # Code Block: tennis ball
        _e = self._scene.scene.add_entity(
            gs.morphs.Sphere(
                pos=(0.60, 0.00, 0.76 + 0.0335), euler=(0, 0, 0), radius=0.0335, fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(color=(0.7, 0.9, 0.2), double_sided=True),
        )
        self._entities["tennis ball"] = {"entity": _e}
