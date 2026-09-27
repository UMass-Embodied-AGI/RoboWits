from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

SeparateMarblesAndSandEnv = _import("gs_gym.envs.robowits.15_separate_marbles_and_sand").SeparateMarblesAndSandEnv


@register_task("robowits/15_03-v0")
class SeparateMarblesAndSandMut3Env(SeparateMarblesAndSandEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Distractors only have to sit on the table, not within arm reach, so give
        # them the wider surface: with everything squeezed into the default
        # reachable box, placement could not fit them and fell back to the
        # canonical layout on nearly every reset.
        clutter_area = {"x_min": 0.25, "x_max": 0.88, "y_min": -0.62, "y_max": 0.62}
        for name in ("mesh_strainer",):
            self._object_reachable_areas[name] = clutter_area
        # The bowl competes with the jar/marble cluster and the colander; give it
        # the full default reachable x-range so a free spot exists.
        self._object_reachable_areas["bowl"] = {"x_min": 0.2, "x_max": 0.75, "y_min": -0.45, "y_max": 0.45}

    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (
            *super().placement_groups,
            # The strainer is clutter, not part of the task: grouping it with the bowl
            # forced the pair into the bowl's area as one 0.261 m cluster, which never fit.
            "mesh_strainer",
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
            material=gs.materials.Rigid(rho=300.0, friction=1.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["colander"] = {"entity": _e}
        # Code Block: bowl
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c652cf0f-d2eb-44bd-9e68-a2ceca698591/obj.glb", pattern_is_dir=False),
                scale=1.38,
                pos=(0.42, -0.10, 0.808576),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=300.0, friction=1.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["bowl"] = {"entity": _e}
        # Code Block: jar
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/simplify_red_cup.obj", pattern_is_dir=False),
                scale=1,
                pos=(0.44, 0.10, 0.824),
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
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["mesh_strainer"] = {"entity": _e}
        # Code Block: sand
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.45, 0.11, 1.19), euler=(0, 0, 0), radius=0.02, height=0.2, fixed=False, collision=True
            ),
            material=gs.materials.MPM.Sand(rho=800.0, sampler="pbs", friction_angle=30),
            surface=gs.surfaces.Default(color=(0.9, 0.8, 0.3), vis_mode="particle", double_sided=True),
        )
        self._entities["sand"] = {"entity": _e}
        # Code Block: marbles
        marble_positions = [
            (0.44, 0.1, 0.775),
            (0.44, 0.1, 0.795),
            (0.44, 0.1, 0.815),
            (0.44, 0.1, 0.835),
            (0.44, 0.1, 0.855),
        ]
        for i, pos in enumerate(marble_positions, start=1):
            _e = self._scene.scene.add_entity(
                gs.morphs.Mesh(
                    coacd_options=coacd_options,
                    file=get_asset_path(
                        "blender_kit/5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb", pattern_is_dir=False
                    ),
                    scale=0.12,
                    pos=pos,
                    euler=(0.0, 0.0, 0.0),
                    fixed=False,
                    collision=True,
                ),
                material=gs.materials.Rigid(rho=1200.0, friction=0.6),
                surface=gs.surfaces.Smooth(double_sided=True),
            )
            self._entities[f"marble_{i}"] = {"entity": _e}
