from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

SeparateMarblesAndSandEnv = _import("gs_gym.envs.robowits.15_separate_marbles_and_sand").SeparateMarblesAndSandEnv


@register_task("robowits/15_05-v0")
class SeparateMarblesAndSandMut5Env(SeparateMarblesAndSandEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (
            ("sand", "marble 1", "marble 2", "marble 3", "marble 4", "marble 5", "jar"),
            "colander",
            "bowl",
            "coffee filter",
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
                pos=(0.50, 0.0, 0.76 + 0.0739 * 0.585),
                euler=(180, 0, 0),
                fixed=False,
                collision=False,
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
                pos=(0.30, 0.0, 0.76 + 0.0352 * 1.227),
                euler=(0, 0, 0),
                fixed=False,
                collision=False,
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
                pos=(0.70, 0.0, 0.855),
                euler=(90.0, 0.0, 0.0),
                fixed=False,
                collision=False,
                convexify=False,
            ),
            material=gs.materials.Rigid(rho=300.0, friction=0.9, sdf_max_res=256),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["jar"] = {"entity": _e}
        # Code Block: coffee filter
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/4b5c4da2-2e9c-40aa-9542-711c163d9d80/obj.glb", pattern_is_dir=False),
                scale=1,
                pos=(0.72, -0.15, 0.76 + (0.0459 - (-0.0459)) * 1.3 / 2),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["coffee filter"] = {"entity": _e}
        # Code Block: sand
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.03,
                height=0.05,
                pos=(0.70, 0.0, 0.76 + 0.022 + 0.05 / 2),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.MPM.Sand(rho=900.0, friction_angle=45),
            surface=gs.surfaces.Default(roughness=0.6, ior=1.5, vis_mode="recon", color=(0.95, 0.85, 0.4)),
        )
        self._entities["sand"] = {"entity": _e}
        # Code Block: marble 1
        _e = self._scene.scene.add_entity(
            gs.morphs.Sphere(radius=0.008, pos=(0.70, 0.0, 0.82), euler=(0, 0, 0), fixed=False, collision=True),
            material=gs.materials.Rigid(rho=1500.0, friction=0.6),
            surface=gs.surfaces.Smooth(color=(0.2, 0.6, 1.0), double_sided=True),
        )
        self._entities["marble 1"] = {"entity": _e}
        # Code Block: marble 2
        _e = self._scene.scene.add_entity(
            gs.morphs.Sphere(radius=0.008, pos=(0.688, 0.012, 0.832), euler=(0, 0, 0), fixed=False, collision=True),
            material=gs.materials.Rigid(rho=1500.0, friction=0.6),
            surface=gs.surfaces.Smooth(color=(0.9, 0.3, 0.3), double_sided=True),
        )
        self._entities["marble 2"] = {"entity": _e}
        # Code Block: marble 3
        _e = self._scene.scene.add_entity(
            gs.morphs.Sphere(radius=0.008, pos=(0.712, -0.012, 0.832), euler=(0, 0, 0), fixed=False, collision=True),
            material=gs.materials.Rigid(rho=1500.0, friction=0.6),
            surface=gs.surfaces.Smooth(color=(0.3, 0.9, 0.4), double_sided=True),
        )
        self._entities["marble 3"] = {"entity": _e}
        # Code Block: marble 4
        _e = self._scene.scene.add_entity(
            gs.morphs.Sphere(radius=0.008, pos=(0.688, -0.012, 0.816), euler=(0, 0, 0), fixed=False, collision=True),
            material=gs.materials.Rigid(rho=1500.0, friction=0.6),
            surface=gs.surfaces.Smooth(color=(0.95, 0.8, 0.2), double_sided=True),
        )
        self._entities["marble 4"] = {"entity": _e}
        # Code Block: marble 5
        _e = self._scene.scene.add_entity(
            gs.morphs.Sphere(radius=0.008, pos=(0.712, 0.012, 0.816), euler=(0, 0, 0), fixed=False, collision=True),
            material=gs.materials.Rigid(rho=1500.0, friction=0.6),
            surface=gs.surfaces.Smooth(color=(0.6, 0.4, 1.0), double_sided=True),
        )
        self._entities["marble 5"] = {"entity": _e}
