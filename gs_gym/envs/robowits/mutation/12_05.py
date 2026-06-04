from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

BallIntoBottleEnv = _import("gs_gym.envs.robowits.12_ball_into_bottle").BallIntoBottleEnv


@register_task("robowits/12_05-v0")
class BallIntoBottleMut5Env(BallIntoBottleEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (("ball container", "small ball"), "red bottle", "funnel", "kitchen strainer")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: red bottle
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/ffb3fbe7-1355-465f-8750-475210d8c949/obj.glb", pattern_is_dir=False),
                scale=0.81,
                pos=(0.52, 0.00, 0.76 + 0.81 * 0.1422),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1.5),
            surface=gs.surfaces.Glass(double_sided=True, color=(0.8, 0.2, 0.2)),
        )
        self._entities["red bottle"] = {"entity": _e}
        # Code Block: funnel
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/funnel.glb", pattern_is_dir=False),
                scale=0.8,
                pos=(0.52, 0.16, 0.76 + 0.8 * 0.0682),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["funnel"] = {"entity": _e}
        # Code Block: ball container
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/3d998505-6bbb-4cc2-8359-c147ac531430/obj.glb", pattern_is_dir=False),
                scale=0.70,
                pos=(0.65, -0.15, 0.76 + 0.1332 * 0.70),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1.2),
            surface=gs.surfaces.Glass(color=(0.8, 0.9, 1.0), opacity=0.3, double_sided=True),
        )
        self._entities["ball container"] = {"entity": _e}
        # Code Block: kitchen strainer
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c6009731-c1d9-48f9-9486-1d5754c336d9/obj.glb", pattern_is_dir=False),
                scale=0.36,
                pos=(0.66, 0.28, 0.76 + 0.36 * 0.0739),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["kitchen strainer"] = {"entity": _e}
        # Code Block: small ball
        _e = self._scene.scene.add_entity(
            gs.morphs.Sphere(pos=(0.65, -0.15, 0.82), radius=0.01, fixed=False, collision=True),
            material=gs.materials.Rigid(rho=500.0),
            surface=gs.surfaces.Smooth(color=(0.2, 0.6, 0.9), double_sided=True),
        )
        self._entities["small ball"] = {"entity": _e}
