from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

SeparateMarblesAndSandEnv = _import("gs_gym.envs.robowits.15_separate_marbles_and_sand").SeparateMarblesAndSandEnv


@register_task("robowits/15_03-v0")
class SeparateMarblesAndSandMut3Env(SeparateMarblesAndSandEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (
            ("sand", "marble 1", "marble 2", "marble 3", "marble 4", "marble 5", "jar"),
            "colander",
            ("bowl", "mesh strainer"),
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
                scale=0.864,
                pos=(0.64, 0.15, 0.8038),
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
                pos=(0.42, -0.10, 0.76 + 0.0739 * 0.585),
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
                pos=(0.44, 0.10, 0.855),
                euler=(90.0, 0.0, 0.0),
                fixed=False,
                collision=True,
                convexify=False,
            ),
            material=gs.materials.Rigid(rho=300.0, friction=0.9, sdf_max_res=256),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["jar"] = {"entity": _e}
        # Code Block: mesh strainer
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c6009731-c1d9-48f9-9486-1d5754c336d9/obj.glb", pattern_is_dir=False),
                scale=0.66,
                pos=(0.42, -0.10, 0.895),
                euler=(0, 0, 0),
                fixed=False,
                collision=False,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["mesh strainer"] = {"entity": _e}
        # Code Block: sand
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.44, 0.10, 0.807), euler=(0, 0, 0), radius=0.032, height=0.05, fixed=False, collision=True
            ),
            material=gs.materials.MPM.Sand(rho=1000.0, sampler="random", friction_angle=45),
            surface=gs.surfaces.Default(color=(0.95, 0.85, 0.2), double_sided=True, vis_mode="recon"),
        )
        self._entities["sand"] = {"entity": _e}
        # Code Block: marble 1
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb", pattern_is_dir=False),
                scale=0.147,
                pos=(0.44, 0.10, 0.848),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=800.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["marble 1"] = {"entity": _e}
        # Code Block: marble 2
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb", pattern_is_dir=False),
                scale=0.147,
                pos=(0.46, 0.12, 0.842),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=800.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["marble 2"] = {"entity": _e}
        # Code Block: marble 3
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb", pattern_is_dir=False),
                scale=0.147,
                pos=(0.46, 0.08, 0.842),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=800.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["marble 3"] = {"entity": _e}
        # Code Block: marble 4
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb", pattern_is_dir=False),
                scale=0.147,
                pos=(0.42, 0.12, 0.842),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=800.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["marble 4"] = {"entity": _e}
        # Code Block: marble 5
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb", pattern_is_dir=False),
                scale=0.147,
                pos=(0.42, 0.08, 0.842),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=800.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["marble 5"] = {"entity": _e}
