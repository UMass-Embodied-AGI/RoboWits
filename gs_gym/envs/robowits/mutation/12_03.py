from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

BallIntoBottleEnv = _import("gs_gym.envs.robowits.12_ball_into_bottle").BallIntoBottleEnv


@register_task("robowits/12_03-v0")
class BallIntoBottleMut3Env(BallIntoBottleEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (
            ("ball container", "small ball"),
            "red bottle",
            "funnel",
            "cat figurine",
            "bananaa",
            "dispenser",
            "tableware",
            "chasen",
        )

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
                scale=0.84,
                pos=(0.50, 0.00, 0.76 + 0.1422 * 0.84),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Glass(double_sided=True, color=(0.8, 0.2, 0.2)),
        )
        self._entities["red bottle"] = {"entity": _e}
        # Code Block: ball container
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/3d998505-6bbb-4cc2-8359-c147ac531430/obj.glb", pattern_is_dir=False),
                scale=0.60,
                pos=(0.50, -0.18, 0.76 + 0.1332 * 0.60),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Glass(color=(0.8, 0.9, 1.0), opacity=0.3, double_sided=True),
        )
        self._entities["ball container"] = {"entity": _e}
        # Code Block: funnel
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/funnel.glb", pattern_is_dir=False),
                scale=0.53,
                pos=(0.62, 0.10, 0.76 + 0.1133 * 0.53),
                euler=(-90, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["funnel"] = {"entity": _e}
        # Code Block: small ball
        _e = self._scene.scene.add_entity(
            gs.morphs.Sphere(radius=0.0075, pos=(0.50, -0.18, 0.86), euler=(0, 0, 0), fixed=False, collision=True),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(color=(0.2, 0.6, 0.9), double_sided=True),
        )
        self._entities["small ball"] = {"entity": _e}
        # Code Block: tableware
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/5acad02b-19b1-4b2a-a015-1e21b95b6d2d/obj.glb", pattern_is_dir=False),
                scale=0.40,
                pos=(0.619, 0.33, 0.76 + 0.0529 * 0.40),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["tableware"] = {"entity": _e}
        # Code Block: bananas
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/871ca434-8876-4d33-82f7-324b517ff67b/obj.glb", pattern_is_dir=False),
                scale=0.80,
                pos=(0.62, -0.33, 0.76 + 0.0893 * 0.80),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["bananas"] = {"entity": _e}
        # Code Block: cat figurine
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/b349c584-2667-46ff-94e2-3b598d7a362f/obj.glb", pattern_is_dir=False),
                scale=0.46,
                pos=(0.41, 0.12, 0.76 + 0.1086 * 0.46),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["cat figurine"] = {"entity": _e}
        # Code Block: dispenser
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/48cd9352-3020-4c02-aa4a-13f87f712762/obj.glb", pattern_is_dir=False),
                scale=0.83,
                pos=(0.36, -0.06, 0.76 + 0.0845 * 0.83),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["dispenser"] = {"entity": _e}
        # Code Block: chasen
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/03a1d5fd-4793-43fc-b009-df7a13b78f8e/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.44, -0.12, 0.76 + 0.0391 * 1.0),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["chasen"] = {"entity": _e}
