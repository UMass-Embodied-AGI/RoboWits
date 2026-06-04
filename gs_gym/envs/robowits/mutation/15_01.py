from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

SeparateMarblesAndSandEnv = _import("gs_gym.envs.robowits.15_separate_marbles_and_sand").SeparateMarblesAndSandEnv


@register_task("robowits/15_01-v0")
class SeparateMarblesAndSandMut1Env(SeparateMarblesAndSandEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (
            ("sand", "marble 1", "marble 2", "marble 3", "marble 4", "marble 5", "jar", "jar_inner"),
            "colander",
            "bowl",
            "wooden pencil case",
            "sunglasses",
            "pentel eraser",
        )

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: colander
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/633e9459-f7ed-4507-ad97-ce2783b06a02/obj.glb", pattern_is_dir=False),
                scale=0.764,
                pos=(0.38, -0.18, 0.76 + 0.0739 * 0.585),
                euler=(180, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["colander"] = {"entity": _e}
        # Code Block: bowl
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c652cf0f-d2eb-44bd-9e68-a2ceca698591/obj.glb", pattern_is_dir=False),
                scale=1.227,
                pos=(0.38, 0.18, 0.76 + 0.0739 * 0.585),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["bowl"] = {"entity": _e}
        # Code Block: jar
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/simplify_red_cup.obj", pattern_is_dir=False),
                scale=1.5,
                pos=(0.58, 0.0, 0.855),
                euler=(90.0, 0.0, 0.0),
                fixed=True,
                collision=True,
                convexify=False,
            ),
            material=gs.materials.Rigid(rho=300.0, friction=0.9, sdf_max_res=256),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["jar"] = {"entity": _e}
        # Code Block: marble 1
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb", pattern_is_dir=False),
                scale=0.1467,
                pos=(0.565, 0.0, 0.811118 - 0.015),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["marble 1"] = {"entity": _e}
        # Code Block: marble 2
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb", pattern_is_dir=False),
                scale=0.1467,
                pos=(0.595, 0.0, 0.811118 - 0.015),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["marble 2"] = {"entity": _e}
        # Code Block: marble 3
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb", pattern_is_dir=False),
                scale=0.1467,
                pos=(0.58, 0.015, 0.811118 - 0.015),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["marble 3"] = {"entity": _e}
        # Code Block: marble 4
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb", pattern_is_dir=False),
                scale=0.1467,
                pos=(0.58 - 0.015, -0.015, 0.811118 - 0.003),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["marble 4"] = {"entity": _e}
        # Code Block: marble 5
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb", pattern_is_dir=False),
                scale=0.1467,
                pos=(0.58 + 0.015, 0.015, 0.811118 - 0.003),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["marble 5"] = {"entity": _e}
        # Code Block: sand
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.58, 0.0, 0.788), euler=(0, 0, 0), radius=0.028, height=0.015, fixed=False, collision=True
            ),
            material=gs.materials.MPM.Sand(rho=1000.0, sampler="random", friction_angle=45),
            surface=gs.surfaces.Default(color=(0.9, 0.8, 0.2), vis_mode="recon", double_sided=True),
        )
        self._entities["sand"] = {"entity": _e}
        # Code Block: wooden pencil case
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/4de06b64-1660-4775-818b-4935e4cc8cf3/obj.glb", pattern_is_dir=False),
                scale=0.6,
                pos=(0.70, -0.34, 0.82),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["wooden pencil case"] = {"entity": _e}
        # Code Block: sunglasses
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/426300b9-fa86-4ac6-bb73-f3b9d2daf731/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.70, 0.34, 0.76 + 0.0303),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["sunglasses"] = {"entity": _e}
        # Code Block: pentel eraser
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/361aa36f-d104-465c-afb1-ec68c2f6611f/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.72, -0.18, 0.76 + 0.0211),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["pentel eraser"] = {"entity": _e}
        # Code Block: jar_inner
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.58, 0.0, 0.76 + 0.045), euler=(0, 0, 0), radius=0.042, height=0.09, fixed=True, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1.0),
            surface=gs.surfaces.Default(color=(0.8, 0.8, 0.8), vis_mode="collision", double_sided=True),
        )
        self._entities["jar_inner"] = {"entity": _e}
