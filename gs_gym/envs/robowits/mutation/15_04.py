from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

SeparateMarblesAndSandEnv = _import("gs_gym.envs.robowits.15_separate_marbles_and_sand").SeparateMarblesAndSandEnv


@register_task("robowits/15_04-v0")
class SeparateMarblesAndSandMut4Env(SeparateMarblesAndSandEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (
            ("sand", "marble 1", "marble 2", "marble 3", "marble 4", "marble 5", "jar"),
            "colander",
            "bowl",
            "ladle",
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
                scale=0.9508,
                pos=(0.40, -0.15, 0.76 + 0.0502 * 0.9508),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=300.0, friction=0.8),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["colander"] = {"entity": _e}
        # Code Block: bowl
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c652cf0f-d2eb-44bd-9e68-a2ceca698591/obj.glb", pattern_is_dir=False),
                scale=1.534,
                pos=(0.40, 0.09, 0.76 + 0.0352 * 1.534),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=300.0, friction=0.8),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["bowl"] = {"entity": _e}
        # Code Block: jar
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/simplify_red_cup.obj", pattern_is_dir=False),
                scale=1.5,
                pos=(0.60, 0.09, 0.855),
                euler=(90.0, 0.0, 0.0),
                fixed=True,
                collision=True,
                convexify=False,
            ),
            material=gs.materials.Rigid(rho=300.0, friction=0.9, sdf_max_res=256),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["jar"] = {"entity": _e}
        # Code Block: ladle
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/d6d7aa32-40ba-4c63-9ef4-1f33138cc47d/obj.glb", pattern_is_dir=False),
                scale=0.847,
                pos=(0.60, -0.18, 0.76 + 0.0547 * 0.847),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500.0, friction=0.6),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["ladle"] = {"entity": _e}
        # Code Block: marble 1
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb", pattern_is_dir=False),
                scale=0.0933,
                pos=(0.588, 0.078, 0.842),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=2500.0, friction=0.6),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["marble 1"] = {"entity": _e}
        # Code Block: marble 2
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb", pattern_is_dir=False),
                scale=0.0933,
                pos=(0.612, 0.078, 0.842),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=2500.0, friction=0.6),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["marble 2"] = {"entity": _e}
        # Code Block: marble 3
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb", pattern_is_dir=False),
                scale=0.0933,
                pos=(0.588, 0.102, 0.842),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=2500.0, friction=0.6),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["marble 3"] = {"entity": _e}
        # Code Block: marble 4
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb", pattern_is_dir=False),
                scale=0.0933,
                pos=(0.612, 0.102, 0.842),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=2500.0, friction=0.6),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["marble 4"] = {"entity": _e}
        # Code Block: marble 5
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb", pattern_is_dir=False),
                scale=0.0933,
                pos=(0.60, 0.09, 0.838),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=2500.0, friction=0.6),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["marble 5"] = {"entity": _e}
        # Code Block: sand
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.026, height=0.030, pos=(0.60, 0.09, 0.797), euler=(0.0, 0.0, 0.0), fixed=False, collision=True
            ),
            material=gs.materials.MPM.Sand(rho=1000.0, sampler="random", friction_angle=45),
            surface=gs.surfaces.Default(color=(0.9, 0.82, 0.3), vis_mode="recon", double_sided=True),
        )
        self._entities["sand"] = {"entity": _e}
