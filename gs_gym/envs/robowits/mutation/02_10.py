from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

RetrieveCubeEnv = _import("gs_gym.envs.robowits.02_retrieve_cube").RetrieveCubeEnv


@register_task("robowits/02_10-v0")
class RetrieveCubeMut10Env(RetrieveCubeEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (("deep narrow container", "cube"), "target area", "wooden skewer")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: deep narrow container
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/ddfb928e-c26d-4805-b41a-ab1545269e99/obj.glb", pattern_is_dir=False),
                scale=0.6,
                pos=(0.55, 0.0, 0.76 + 0.1332 * 0.6),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(
                rho=200.0, friction=None, coup_friction=0.15, coup_softness=0.002, coup_restitution=0.0
            ),
            surface=gs.surfaces.Glass(color=(1.0, 1.0, 1.0), double_sided=True),
        )
        self._entities["deep narrow container"] = {"entity": _e}
        # Code Block: cube
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.55, 0.0, 0.90), euler=(0.0, 0.0, 0.0), size=(0.015, 0.015, 0.015), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=300.0),
            surface=gs.surfaces.Rough(color=(0.8, 0.6, 0.4), double_sided=True),
        )
        self._entities["cube"] = {"entity": _e}
        # Code Block: target area
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.38, -0.05, 0.761),
                euler=(0.0, 0.0, 0.0),
                size=(0.12, 0.12, 0.002),
                fixed=True,
                collision=False,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(color=(0.1, 0.8, 0.1), double_sided=True),
        )
        self._entities["target area"] = {"entity": _e}
        # Code Block: wooden skewer
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.55, 0.16, 0.76 + 0.003),
                euler=(0.0, 90.0, 0.0),
                radius=0.003,
                height=0.28,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500.0),
            surface=gs.surfaces.Rough(color=(0.86, 0.76, 0.52), double_sided=False),
        )
        self._entities["wooden skewer"] = {"entity": _e}
